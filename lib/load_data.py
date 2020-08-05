import os
import numpy as np
import time
from lib.utils import filter_cloth_pose
from lib.mesh_sampling import laplacian

def load_graph_mtx(project_dir, load_for_demo=False):
    print('loading pre-saved transform matrices...')
    A_ds2 = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/ds2/A.npy'), encoding='latin1'))
    D_ds2 = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/ds2/D.npy'), encoding='latin1'))
    U_ds2 = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/ds2/U.npy'), encoding='latin1'))

    A_ds2 = list(map(lambda x: x.astype('float32'), A_ds2))
    D_ds2 = list(map(lambda x: x.astype('float32'), D_ds2))
    U_ds2 = list(map(lambda x: x.astype('float32'), U_ds2))

    L_ds2 = [laplacian(a, normalized=True) for a in A_ds2]

    if not load_for_demo:
        return L_ds2, D_ds2, U_ds2
    else:
        A = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/for_demo/A.npy'), encoding='latin1'))
        D = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/for_demo/D.npy'), encoding='latin1'))
        U = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/for_demo/U.npy'), encoding='latin1'))

        p = list(map(lambda x: x.shape[0], A))
        A = list(map(lambda x: x.astype('float32'), A))
        D = list(map(lambda x: x.astype('float32'), D))
        U = list(map(lambda x: x.astype('float32'), U))

        L = [laplacian(a, normalized=True) for a in A]
        return L, D, U, p, L_ds2, D_ds2, U_ds2


class BodyData(object):
    def __init__(self, nVal, train_mesh_fn, train_cond1_fn, test_mesh_fn, test_cond1_fn, reference_mesh_file,
                 train_cond2_fn=None, test_cond2_fn=None):
        self.nVal = nVal
        self.train_mesh_fn = train_mesh_fn
        self.train_cond1_fn = train_cond1_fn
        self.train_cond2_fn = train_cond2_fn
        self.test_mesh_fn = test_mesh_fn
        self.test_cond1_fn = test_cond1_fn
        self.test_cond2_fn = test_cond2_fn
        self.vertices_train, self.cond1_train = None, None
        self.vertices_val, self.cond1_val = None, None
        self.vertices_test, self.cond1_test = None, None
        self.N = None
        self.n_vertex = None

        self.load()
        from psbody.mesh import Mesh
        self.reference_mesh = Mesh(filename=reference_mesh_file)

        self.mean = np.mean(self.vertices_train, axis=0)
        self.std = np.std(self.vertices_train, axis=0)

        self.normalize()
        self.change_dtype()

    def load(self):
        vertices_train = np.load(self.train_mesh_fn)

        self.vertices_train = vertices_train[:-self.nVal]
        self.vertices_val = vertices_train[-self.nVal:]

        cond1_train = np.load(self.train_cond1_fn)
        if len(cond1_train.shape) > 2: #pose param not flattened
            cond1_train = cond1_train.reshape(len(cond1_train), -1)
        self.cond1_train = cond1_train[:-self.nVal]
        self.cond1_val = cond1_train[-self.nVal:]

        if self.train_cond2_fn is not None:
            cond2_train = np.load(self.train_cond2_fn)
            self.cond2_train = cond2_train[:-self.nVal]
            self.cond2_val = cond2_train[-self.nVal:]

        self.n_vertex = self.vertices_train.shape[1]

        self.vertices_test = np.load(self.test_mesh_fn)
        self.cond1_test = np.load(self.test_cond1_fn)

        if self.test_cond2_fn is not None:
            self.cond2_test = np.load(self.test_cond2_fn)

        if len(self.cond1_test.shape) > 2: #pose param not flattened
            self.cond1_test = self.cond1_test.reshape(len(self.cond1_test), -1)

        '''
        Remove pose parameters of joints irrelevant to clothing (e.g. head)
        while keeping the full pose parameters for test time use such as reposing the model
        14 is the number of clothing-related joints, just to make sure the data is not already pose-filtered
        '''
        if self.cond1_test.shape[-1] % 14 != 0:
            self.cond1_test_full = self.cond1_test
            self.cond1_train_full = self.cond1_train
            self.cond1_val_full = self.cond1_val

            self.cond1_train, self.cond1_val, self.cond1_test = list(map(filter_cloth_pose, [self.cond1_train, self.cond1_val, self.cond1_test]))
        print('Data loaded, {} train, {} val, {} test examples.\n'.format(len(self.vertices_train),
                                                                       len(self.vertices_val), len(self.vertices_test)))

    def normalize(self):
        self.vertices_train -= self.mean
        self.vertices_train /= self.std

        self.vertices_val -= self.mean
        self.vertices_val /= self.std

        self.vertices_test -= self.mean
        self.vertices_test /= self.std

        print('Vertices normalized.\n')

    def change_dtype(self):
        self.vertices_train = self.vertices_train.astype('float32')
        self.vertices_val = self.vertices_val.astype('float32')
        self.vertices_test = self.vertices_test.astype('float32')

        self.cond1_train = self.cond1_train.astype('float32')
        self.cond1_val = self.cond1_val.astype('float32')
        self.cond1_test = self.cond1_test.astype('float32')

        if self.train_cond2_fn is not None:
            self.cond2_train = self.cond2_train.astype('float32')
            self.cond2_val = self.cond2_val.astype('float32')
            self.cond2_test = self.cond2_test.astype('float32')

    def vec2mesh(self, vec):
        from psbody.mesh import Mesh
        vec = vec.reshape((self.n_vertex, 3)) * self.std + self.mean
        return Mesh(v=vec, f=self.reference_mesh.f)

    def show_mesh(self, viewer, mesh_vecs, figsize):
        for i in range(figsize[0]):
            for j in range(figsize[1]):
                mesh_vec = mesh_vecs[i * (figsize[0] - 1) + j]
                mesh_mesh = self.vec2mesh(mesh_vec)
                viewer[i][j].set_dynamic_meshes([mesh_mesh])
        time.sleep(0.1)  # pause 0.5 seconds
        return 0

    def get_normalized_meshes(self, mesh_paths):
        from psbody.mesh import Mesh
        meshes = []
        for mesh_path in mesh_paths:
            mesh = Mesh(filename=mesh_path)
            mesh_v = (mesh.v - self.mean) / self.std
        meshes.append(mesh_v)
        return np.array(meshes)
