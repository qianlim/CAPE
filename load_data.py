import glob
import os
import numpy as np
import time
import random
from copy import deepcopy
from tqdm import tqdm
from psbody.mesh import Mesh, MeshViewer, MeshViewers
from lib.utils import filter_cloth_pose
from lib.mesh_sampling import laplacian

def load_graph_mtx(project_dir):
    print('loading pre-saved transform matrices for downsampling factor=2..')
    A_ds2 = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/ds2/A.npy'), encoding='latin1'))
    D_ds2 = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/ds2/D.npy'), encoding='latin1'))
    U_ds2 = list(np.load(os.path.join(project_dir, 'data', 'transform_matrices/ds2/U.npy'), encoding='latin1'))

    A_ds2 = list(map(lambda x: x.astype('float32'), A_ds2))
    D_ds2 = list(map(lambda x: x.astype('float32'), D_ds2))
    U_ds2 = list(map(lambda x: x.astype('float32'), U_ds2))

    L_ds2 = [laplacian(a, normalized=True) for a in A_ds2]

    return L_ds2, D_ds2, U_ds2


class BodyData(object):
    def __init__(self, nVal, train_file_A, train_file_C, test_file_A, test_file_C, reference_mesh_file,
                 train_file_C2=None, test_file_C2=None):
        self.nVal = nVal
        self.train_file_A = train_file_A
        self.train_file_C = train_file_C
        self.train_file_C2 = train_file_C2
        self.test_file_A = test_file_A
        self.test_file_C = test_file_C
        self.test_file_C2 = test_file_C2
        self.vertices_train_A, self.data_train_C = None, None
        self.vertices_val_A, self.data_val_C = None, None
        self.vertices_test_A, self.data_test_C = None, None
        self.N = None
        self.n_vertex = None

        self.load()
        self.reference_mesh = Mesh(filename=reference_mesh_file)

        self.mean_A = np.mean(self.vertices_train_A, axis=0)
        self.std_A = np.std(self.vertices_train_A, axis=0)

        self.mean_C = np.mean(self.data_train_C, axis=0)
        self.std_C = np.std(self.data_train_C, axis=0)

        self.normalize()
        self.change_dtype()

    def load(self):
        vertices_train_A = np.load(self.train_file_A)

        self.vertices_train_A = vertices_train_A[:-self.nVal]
        self.vertices_val_A = vertices_train_A[-self.nVal:]
        self.vertices_train = self.vertices_train_A

        data_train_C = np.load(self.train_file_C)
        if len(data_train_C.shape) > 2: #pose param not flattened
            data_train_C = data_train_C.reshape(len(data_train_C), -1)
        self.data_train_C = data_train_C[:-self.nVal]
        self.data_val_C = data_train_C[-self.nVal:]

        if self.train_file_C2 is not None:
            data_train_C2 = np.load(self.train_file_C2)
            self.data_train_C2 = data_train_C2[:-self.nVal]
            self.data_val_C2 = data_train_C2[-self.nVal:]

        self.n_vertex = self.vertices_train.shape[1]

        self.vertices_test_A = np.load(self.test_file_A)
        self.data_test_C = np.load(self.test_file_C)

        if self.test_file_C2 is not None:
            self.data_test_C2 = np.load(self.test_file_C2)

        if len(self.data_test_C.shape) > 2: #pose param not flattened
            self.data_test_C = self.data_test_C.reshape(len(self.data_test_C), -1)

        '''
        Remove pose parameters of joints irrelevant to clothing (e.g. head)
        while keeping the full pose parameters for test time use such as reposing the model
        14 is the number of clothing-related joints, just to make sure the data is not already pose-filtered
        '''
        if self.data_test_C.shape[-1] % 14 != 0:
            self.data_test_C_full = self.data_test_C
            self.data_train_C_full = self.data_train_C
            self.data_val_C_full = self.data_val_C

            self.data_train_C, self.data_val_C, self.data_test_C = list(map(filter_cloth_pose, [self.data_train_C, self.data_val_C, self.data_test_C]))

    def normalize(self):
        self.vertices_train_A -= self.mean_A
        self.vertices_train_A /= self.std_A

        self.vertices_val_A -= self.mean_A
        self.vertices_val_A /= self.std_A

        self.vertices_test_A -= self.mean_A
        self.vertices_test_A /= self.std_A

        print('Vertices normalized')

    def change_dtype(self):
        self.vertices_train_A = self.vertices_train_A.astype('float32')
        self.vertices_val_A = self.vertices_val_A.astype('float32')
        self.vertices_test_A = self.vertices_test_A.astype('float32')

        self.data_train_C = self.data_train_C.astype('float32')
        self.data_val_C = self.data_val_C.astype('float32')
        self.data_test_C = self.data_test_C.astype('float32')

        if self.train_file_C2 is not None:
            self.data_train_C2 = self.data_train_C2.astype('float32')
            self.data_val_C2 = self.data_val_C2.astype('float32')
            self.data_test_C2 = self.data_test_C2.astype('float32')

    def vec2mesh(self, vec):
        vec = vec.reshape((self.n_vertex, 3)) * self.std_A + self.mean_A
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
        meshes = []
        for mesh_path in mesh_paths:
            mesh = Mesh(filename=mesh_path)
            mesh_v = (mesh.v - self.mean) / self.std
        meshes.append(mesh_v)
        return np.array(meshes)
