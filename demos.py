import numpy as np
import copy
import os
import smplx
import torch
from os.path import join, exists
from psbody.mesh import Mesh, MeshViewer, MeshViewers
from lib.utils import filter_cloth_pose

np.random.seed(123)

class demo(object):
    def __init__(self, model, name, dataset, data_dir, datadir_root, n_sample, save_obj,
                 sample_option='normal', smpl_model_folder='', vis=True):
        self.n_sample = n_sample
        self.sample_option = sample_option
        self.name = name
        self.data_dir = data_dir
        self.datadir_root = datadir_root
        self.model = model
        self.dataset = dataset
        self.save_obj = save_obj
        self.vis = vis

        self.smpl_model = smplx.body_models.create(model_type='smpl',
                                                   model_path=smpl_model_folder,
                                                   gender='neutral')

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.clothing_verts_idx = np.load(join(script_dir, 'data', 'clothing_verts_idx.npy'))
        self.ref_mesh = Mesh(filename=join(script_dir, 'data', 'template_mesh.obj'))
        self.minimal_shape = self.ref_mesh.v

        self.rot = np.load(join(script_dir, 'data', 'demo_data', 'demo_pose_params.npz'))['rot'] # 216 dim pose vector
        self.pose = np.load(join(script_dir, 'data', 'demo_data', 'demo_pose_params.npz'))['pose']

        train_stats = np.load(join(script_dir, 'data', 'demo_data', 'trainset_stats.npz'))
        self.train_mean = train_stats['mean']
        self.train_std = train_stats['std']

        self.results_dir = join(script_dir, 'results', name)
        if not exists(self.results_dir):
            os.makedirs(self.results_dir)

    def sample_vary_pose(self):
        '''
        fix clothing type, sample sevearl poses, under each pose sample latent code N times
        '''
        full_pose = self.pose # take the corresponding full 72-dim pose params, for later reposing
        rot = filter_cloth_pose(self.rot) # only keep pose params from clo-related joints; then take one pose instance
        clotype = np.array([1, 0, 0, 0]) # one-hot clothing type label
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

    def vis_meshviewer(self, mesh1, mesh2, mesh3, n_sample, titlebar='titlebar', disp_value=False, values_to_disp=None):
        from psbody.mesh import Mesh, MeshViewer, MeshViewers

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
                # model.pose[:] = pose_params[i]
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
        self.sample_vary_pose()

