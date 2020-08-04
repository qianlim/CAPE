import numpy as np
import os
import smplx
import torch
import trimesh
from os.path import join, exists
from lib.utils import filter_cloth_pose

np.random.seed(123)

class demo_full(object):
    def __init__(self, bodydata, model, name, gender, dataset, data_dir, datadir_root, n_sample, save_obj,
                 smpl_model_folder='body_models', vis=True):
        self.n_sample = n_sample
        self.name = name
        self.data_dir = data_dir
        self.datadir_root = datadir_root
        self.bodydata = bodydata
        self.model = model
        self.dataset = dataset
        self.save_obj = save_obj
        self.vis = vis

        from psbody.mesh import Mesh

        self.smpl_model = smplx.body_models.create(model_type='smpl',
                                                   model_path=smpl_model_folder,
                                                   gender=gender)

        self.clo_type_readable = np.array(['shortlong', 'shortshort', 'longshort', 'longlong'])

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.clothing_verts_idx = np.load(join(script_dir, 'data', 'clothing_verts_idx.npy'))
        self.ref_mesh = Mesh(filename=join(script_dir, 'data', 'template_mesh.obj'))
        self.vpe = np.load(os.path.join(script_dir, 'data', 'edges_smpl.npy'))  # vertex per edge
        self.minimal_shape = self.ref_mesh.v

        self.rot = np.load(join(script_dir, 'data', 'demo_data', 'demo_pose_params.npz'))['rot'] # 216 dim pose vector
        self.pose = np.load(join(script_dir, 'data', 'demo_data', 'demo_pose_params.npz'))['pose']

        train_stats = np.load(join(script_dir, 'data', 'demo_data', 'trainset_stats.npz'))
        self.train_mean = train_stats['mean']
        self.train_std = train_stats['std']

        self.results_dir = join(script_dir, 'results', name)
        os.makedirs(self.results_dir, exist_ok=True)


    def test_model(self):
        '''
        test the auto-encoding errors of the model
        '''
        print('\n=============== Running demo: test reconstruction ===============')

        obj_dir = join(self.results_dir, 'test_reconstruction_objs_{}'.format(self.dataset))

        vertices = self.bodydata.vertices_test
        condition = self.bodydata.cond1_test
        if hasattr(self.bodydata, 'cond1_test_full'):
            pose_params_full = self.bodydata.cond1_test_full
        condition2 = self.bodydata.cond2_test

        print("\nTesting on test set, {} examples...\n".format(len(vertices)))

        predictions, recon_loss, latent_loss, edge_loss = self.model.predict(data=vertices,
                                                                             cond=condition,
                                                                             cond2=condition2,
                                                                             labels=vertices,
                                                                             phase='test')
        predictions = predictions * self.bodydata.std + self.bodydata.mean
        gt = vertices * self.bodydata.std + self.bodydata.mean

        # compute the test errors that belong to clothing-related vertices
        diff = predictions - gt
        diff = diff[:, self.clothing_verts_idx, :]

        euclidean_err = np.sqrt(np.sum(diff ** 2, axis=2))
        euclidean_err_mean = np.mean(euclidean_err)
        euclidean_err_std = np.std(euclidean_err)
        euclidean_err_median = np.median(euclidean_err)

        test_result_str = "\nResults from {}: \n" \
                          "L1 {:.5f}, KL {:.5f}, Edge {:.5f}\n" \
                          "Eucledian err mean {:.5f}, std {:.5f}, median {:.5f}.\n".format(self.name,
                                recon_loss, latent_loss, edge_loss,
                                euclidean_err_mean, euclidean_err_std, euclidean_err_median)
        print(test_result_str)
        with open(os.path.join(self.results_dir, 'test_results_{}.txt'.format(self.dataset)), 'a+') as fp:
            fp.write(test_result_str)

        with open(os.path.join(self.results_dir, '../all_test_results_{}.txt'.format(self.dataset)), 'a+') as fp:
            fp.write(test_result_str)

        # visualize / save results
        disp_masked = np.zeros_like(predictions) # only add cloth_related disps to body
        disp_masked[:, self.clothing_verts_idx, :] = predictions[:, self.clothing_verts_idx, :]

        predictions_fullbody = disp_masked + self.minimal_shape
        gt_fullbody = gt + self.minimal_shape

        if pose_params_full.shape[-1] == 216:
            # if we use rotation matrices (24*9=216 dim) as condition,
            # need to process it to become pose params (24*3=72 dim)
            from lib.utils import rot2pose
            pose_params_full = rot2pose(pose_params_full)

        if self.save_obj or self.vis:
            if hasattr(self.bodydata, 'cond1_test_full'):
                # only save / vis exemplars of test set, to save time and disk space
                predictions_fullbody_sliced = predictions_fullbody[::int(len(gt_fullbody)/self.n_sample)]
                pose_params_full_sliced = pose_params_full[::int(len(gt_fullbody)/self.n_sample)]
                predictions_fullbody_posed = self.pose_result(predictions_fullbody_sliced, pose_params_full_sliced,
                                                              save_obj=self.save_obj, obj_dir=obj_dir)
                if self.vis:
                    gt_fullbody_sliced = gt_fullbody[::int(len(gt_fullbody)/self.n_sample), :, :]
                    gt_fullbody_posed = self.pose_result(gt_fullbody_sliced, pose_params_full_sliced, save_obj=False, obj_dir=obj_dir)
                    minimal_shape_posed = self.pose_result(np.array([self.minimal_shape]), pose_params_full_sliced, save_obj=False)

            elif self.vis:
                    minimal_shape_repeated = np.repeat(self.minimal_shape[np.newaxis, :], gt_fullbody.shape[0], axis=0)

        if self.vis:
            if hasattr(self.bodydata, 'cond1_test_full'):
                self.vis_meshviewer(predictions_fullbody_posed, gt_fullbody_posed, minimal_shape_posed, self.n_sample)
            else:
                self.vis_meshviewer(predictions_fullbody, gt_fullbody, minimal_shape_repeated, self.n_sample)


    def sample_vary_pose(self):
        '''
        fix clothing type, sample several poses, under each pose sample latent code N times
        '''
        full_pose = self.pose # take the corresponding full 72-dim pose params, for later reposing
        rot = filter_cloth_pose(self.rot) # only keep pose params from clo-related joints; then take one pose instance
        clotype = (self.clo_type_readable == 'shortlong').astype(int) # fix one clothing type
        clotype_repeated = np.repeat(clotype[np.newaxis, :], len(rot), axis=0)

        # get latent embedding of the conditions
        pose_emb, clotype_emb = self.model.encode_only_condition(rot, clotype_repeated)
        clotype_emb = clotype_emb[0]

        obj_dir = join(self.results_dir, 'sample_vary_pose')

        print('\n=============== Running demo: fix z, clotype, change pose ===============')
        print('\nFound {} different pose, for each we generate {} samples\n'.format(len(rot), self.n_sample))

        # sample latent space
        z_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.n_sample, self.model.nz))

        for idx, pose_emb_i in enumerate(pose_emb):
            full_pose_repeated = np.repeat(full_pose[np.newaxis, idx, :], self.n_sample, axis=0)
            # concat z with conditions
            z_sample_c = np.array([np.concatenate([sample.reshape(1, -1), pose_emb_i.reshape(1, -1), clotype_emb.reshape(1, -1)], axis=1)
                                                    for sample in z_samples]).reshape(self.n_sample, -1)

            predictions = self.model.decode(z_sample_c, cond=pose_emb_i.reshape(1, -1), cond2=clotype_emb.reshape(1, -1))
            predictions = predictions * self.train_std + self.train_mean

            # exclude head, fingers and toes
            disp_masked = np.zeros_like(predictions)
            disp_masked[:, self.clothing_verts_idx, :] = predictions[:, self.clothing_verts_idx, :]

            predictions_fullbody = disp_masked + self.minimal_shape

            predictions_fullbody_posed = self.pose_result_onepose_multisample(predictions_fullbody, full_pose_repeated, pose_idx=idx,
                                                          save_obj=self.save_obj, obj_dir=obj_dir)
            if self.vis:
                minimal_shape_posed = self.pose_result_onepose_multisample(np.array([self.minimal_shape]), full_pose_repeated, pose_idx=idx,
                                                   save_obj=False)
                self.vis_meshviewer(mesh1=predictions_fullbody_posed, mesh2=minimal_shape_posed, mesh3=None,
                            n_sample=self.n_sample, titlebar='Sample vary pose')


    def sample_vary_clotype(self):
        '''
        fix body pose, sample 4 clothing types, under each clothing type sample latent code N times
        '''
        full_pose = self.pose[2] # take the corresponding full 72-dim pose params, for later reposing
        full_pose_repeated = np.repeat(full_pose[np.newaxis,:], self.n_sample, axis=0)

        clotype = np.unique(self.bodydata.cond2_test, axis=0)
        rot = filter_cloth_pose(self.rot)[0]  # only keep pose params from clo-related joints; then take one pose instance
        rot_repeated = np.repeat(rot[np.newaxis,:], len(clotype), axis=0) # repeat to pair with clotype

        # get latent embedding of the conditions
        pose_emb, clotype_emb = self.model.encode_only_condition(rot_repeated, clotype)
        pose_emb = pose_emb[0] # since it's repeated so only take one

        print('\n=============== Running demo: fix z, pose, change clothing type ===============')
        print('Found {} different clothing types, for each we generate {} samples\n'.format(len(clotype), self.n_sample))

        obj_dir = join(self.results_dir, 'sample_vary_clotype')

        # sample z from latent space
        z_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.n_sample, self.model.nz))

        for i in range(len(clotype)):
            clotype_i = clotype[i]
            clotype_emb_i =clotype_emb[i]

            clotype_name = self.clo_type_readable[np.argmax(clotype_i)] # get the human-readable clothing types from one-hot vecs

            # concat z with conditions
            z_sample_c = np.array([np.concatenate([sample.reshape(1, -1), pose_emb.reshape(1, -1),
                                clotype_emb_i.reshape(1, -1)], axis=1) for sample in z_samples]).reshape(self.n_sample, -1)

            predictions = self.model.decode(z_sample_c, cond=pose_emb.reshape(1,-1), cond2=clotype_emb_i.reshape(1,-1))

            predictions = predictions * self.train_std + self.train_mean

            # exclude head, fingers and toes
            disp_masked = np.zeros_like(predictions)
            disp_masked[:, self.clothing_verts_idx, :] = predictions[:, self.clothing_verts_idx, :]

            predictions_fullbody = disp_masked + self.minimal_shape

            predictions_fullbody_posed = self.pose_result(predictions_fullbody, full_pose_repeated,
                                                          cloth_type='clotype_{}'.format(clotype_name),
                                                          save_obj=self.save_obj, obj_dir=obj_dir)
            minimal_shape_posed = self.pose_result(np.array([self.minimal_shape]), full_pose_repeated,
                                                   save_obj=False)
            if self.vis:
                self.vis_meshviewer(mesh1=predictions_fullbody_posed, mesh2=minimal_shape_posed, mesh3=None,
                                n_sample=self.n_sample, titlebar='Sample vary clothtype, clothing type: {}'.format(clotype_name))


    def vis_meshviewer(self, mesh1, mesh2, mesh3, n_sample, titlebar='titlebar', disp_value=False, values_to_disp=None):
        from psbody.mesh import Mesh, MeshViewers

        if mesh3 is not None:
            viewer = MeshViewers(shape=(1, 3), titlebar=titlebar)
            for x in range(n_sample):
                viewer[0][0].static_meshes = [Mesh(mesh1[x], self.ref_mesh.f)]
                viewer[0][1].static_meshes = [Mesh(mesh2[x], self.ref_mesh.f)]
                viewer[0][2].static_meshes = [Mesh(mesh3[x], self.ref_mesh.f)]
                if disp_value is False:
                    input('frame {}, Press key for next'.format(x))
                else:
                    input('Current value: {}'.format(values_to_disp[x]))
        else:
            viewer = MeshViewers(shape=(1, 2), titlebar=titlebar)
            for x in range(n_sample):
                viewer[0][0].static_meshes = [Mesh(mesh1[x], self.ref_mesh.f)]
                viewer[0][1].static_meshes = [Mesh(mesh2[x], self.ref_mesh.f)]
                if disp_value is False:
                    input('frame {}, press key for next'.format(x))
                else:
                    input('Current value: {}'.format(values_to_disp[x]))


    def pose_result(self, verts, pose_params, save_obj, cloth_type=None, obj_dir=None):
        '''
        :param verts: [N, 6890, 3]
        :param pose_params: [N, 72]
        '''
        from psbody.mesh import Mesh, MeshViewers

        if verts.shape[0] != 1: # minimal shape: pose it to every pose
            assert verts.shape[0] == pose_params.shape[0] # otherwise the number of results should equal the number of pose identities

        verts_posed = []

        if save_obj:
            if not exists(obj_dir):
                os.makedirs(obj_dir)
            print('saving results as .obj files to {}...'.format(obj_dir))

        if verts.shape[0] == 1:
            self.smpl_model.v_template[:] = torch.from_numpy(verts[0])
            for i in range(len(pose_params)):
                self.smpl_model.body_pose[:] = torch.from_numpy(pose_params[i][3:])
                self.smpl_model.global_orient[:] = torch.from_numpy(pose_params[i][:3])
                verts_out = self.smpl_model().vertices.detach().cpu().numpy()
                verts_posed.append(verts_out)
                if save_obj:
                    if cloth_type is not None:
                        Mesh(verts_out.squeeze(), self.smpl_model.faces).write_obj(join(obj_dir, '{}_{:0>4d}.obj').format(cloth_type, i))
                    else:
                        Mesh(verts_out.squeeze(), self.smpl_model.faces).write_obj(join(obj_dir, '{:0>4d}.obj').format(i))
        else:
            for i in range(len(verts)):
                self.smpl_model.v_template[:] = torch.from_numpy(verts[i])
                self.smpl_model.body_pose[:] = torch.from_numpy(pose_params[i][3:])
                self.smpl_model.global_orient[:] = torch.from_numpy(pose_params[i][:3])
                verts_out = self.smpl_model().vertices.detach().cpu().numpy()
                verts_posed.append(verts_out)
                if save_obj:
                    if cloth_type is not None:
                        Mesh(verts_out.squeeze(), self.smpl_model.faces).write_obj(join(obj_dir, '{}_{:0>4d}.obj').format(cloth_type, i))
                    else:
                        Mesh(verts_out.squeeze(), self.smpl_model.faces).write_obj(join(obj_dir, '{:0>4d}.obj').format(i))

        return verts_posed


    def pose_result_onepose_multisample(self, verts, pose_params, pose_idx, save_obj, obj_dir=None):
        '''
        :param verts: [N, 6890, 3]
        :param pose_params: [N, 72]
        '''
        from psbody.mesh import Mesh, MeshViewers

        if verts.shape[0] != 1: # minimal shape: pose it to every pose
            assert verts.shape[0] == pose_params.shape[0] # otherwise the number of results should equal the number of pose identities

        verts_posed = []

        if save_obj:
            if not exists(obj_dir):
                os.makedirs(obj_dir)
            print('saving results as .obj files to {}...'.format(obj_dir))

        if verts.shape[0] == 1:
            self.smpl_model.v_template[:] = torch.from_numpy(verts[0])
            for i in range(len(pose_params)):
                self.smpl_model.body_pose[:] = torch.from_numpy(pose_params[i][3:])
                self.smpl_model.global_orient[:] = torch.from_numpy(pose_params[i][:3])
                verts_out = self.smpl_model().vertices.detach().cpu().numpy()
                verts_posed.append(verts_out)
                if save_obj:
                    Mesh(verts_out, self.smpl_model.faces).write_obj(join(obj_dir, 'pose{}_{:0>4d}.obj').format(pose_idx, i))

        else:
            for i in range(len(verts)):
                self.smpl_model.v_template[:] = torch.from_numpy(verts[i])
                self.smpl_model.body_pose[:] = torch.from_numpy(pose_params[i][3:])
                self.smpl_model.global_orient[:] = torch.from_numpy(pose_params[i][:3])
                verts_out = self.smpl_model().vertices.detach().cpu().numpy()
                verts_posed.append(verts_out)
                if save_obj:
                    Mesh(verts_out.squeeze(), self.smpl_model.faces).write_obj(join(obj_dir, 'pose{}_{:0>4d}.obj').format(pose_idx, i))

        return verts_posed


    def run(self):
        self.test_model()
        self.sample_vary_pose()
        self.sample_vary_clotype()


class demo_simple(object):
    def __init__(self, model, name):
        self.name = name
        self.model = model
        self.n_sample = 3
        self.save_obj = True

        import trimesh

        self.clo_type_readable = np.array(['shortlong', 'shortshort', 'longshort', 'longlong'])

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.clothing_verts_idx = np.load(join(script_dir, 'data', 'clothing_verts_idx.npy'))
        self.ref_mesh = trimesh.load(join(script_dir, 'data', 'template_mesh.obj'), process=False)
        self.minimal_shape = self.ref_mesh.vertices

        self.rot = np.load(join(script_dir, 'data', 'demo_data', 'demo_pose_params.npz'))['rot']
        self.pose = np.load(join(script_dir, 'data', 'demo_data', 'demo_pose_params.npz'))['pose']

        train_stats = np.load(join(script_dir, 'data', 'demo_data', 'trainset_stats.npz'))
        self.train_mean = train_stats['mean']
        self.train_std = train_stats['std']

        self.results_dir = join(script_dir, 'results', 'demo_results')
        os.makedirs(self.results_dir, exist_ok=True)

    def sample_vary_clotype(self):
        '''
        fix body pose, sample 4 clothing types, under each clothing type sample latent code N times
        '''
        clotype = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]) # one-hot encoding of 4 clothing types
        rot = filter_cloth_pose(self.rot)[0]
        rot_repeated = np.repeat(rot[np.newaxis,:], len(clotype), axis=0)

        # get latent embedding of the conditions
        pose_emb, clotype_emb = self.model.encode_only_condition(rot_repeated, clotype)
        pose_emb = pose_emb[0]

        print('\n=============== Running demo: fix z, pose, change clothing type ===============')
        print('Found {} different clothing types, for each we generate {} samples\n'.format(len(clotype), self.n_sample))

        # sample z from latent space
        z_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.n_sample, self.model.nz))

        for i in range(len(clotype)):
            clotype_i = clotype[i]
            clotype_emb_i =clotype_emb[i]

            clotype_name = self.clo_type_readable[np.argmax(clotype_i)] # get the human-readable clothing types from one-hot vecs

            # concat z with conditions
            z_sample_c = np.array([np.concatenate([sample.reshape(1, -1), pose_emb.reshape(1, -1),
                                clotype_emb_i.reshape(1, -1)], axis=1) for sample in z_samples]).reshape(self.n_sample, -1)

            predictions = self.model.decode(z_sample_c, cond=pose_emb.reshape(1,-1), cond2=clotype_emb_i.reshape(1,-1))

            predictions = predictions * self.train_std + self.train_mean

            # exclude head, fingers and toes
            disp_masked = np.zeros_like(predictions)
            disp_masked[:, self.clothing_verts_idx, :] = predictions[:, self.clothing_verts_idx, :]

            predictions_fullbody = disp_masked + self.minimal_shape

            for j in range(self.n_sample):
                mm = trimesh.Trimesh(vertices=predictions_fullbody[j], faces=self.ref_mesh.faces)
                mm.export(join(self.results_dir, '{}_{:0>4d}.obj').format(clotype_name, j))