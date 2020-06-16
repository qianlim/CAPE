import matplotlib.pyplot as plt
import numpy as np
import cv2
import time, re
import tensorflow as tf

# joint names of SMPL
J_names = { 0: 'Pelvis',
    1: 'L_Hip',
    4: 'L_Knee',
    7: 'L_Ankle',
    10: 'L_Foot',

    2: 'R_Hip',
    5: 'R_Knee',
    8: 'R_Ankle',
    11: 'R_Foot',

    3: 'Spine1',
    6: 'Spine2',
    9: 'Spine3',
    12: 'Neck',
    15: 'Head',

    13: 'L_Collar',
    16: 'L_Shoulder',
    18: 'L_Elbow',
    20: 'L_Wrist',
    22: 'L_Hand',
    14: 'R_Collar',
    17: 'R_Shoulder',
    19: 'R_Elbow',
    21: 'R_Wrist',
    23: 'R_Hand',
}

# indices of SMPL joints related to clothing
useful_joints_idx = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19]

def filter_cloth_pose(pose_vec):
    '''
    Remove SMPL pose params from clothing-unrelated joints
    args:
        pose_vec: flattened 72-dim pose vector or 216-dim rotational matrix
    returns:
        42(=14*3)-dim pose vector or 126(=14*9) dim rot matrix that is relevant to clothing
    '''
    num_examples = pose_vec.shape[0]
    dim = pose_vec.shape[-1]

    # 24x3, SMPL pose parameters
    if dim == 72:
        pose_array = pose_vec.reshape(num_examples, -1, 3)

    # 24x9, rotational matrices of the pose parameters
    elif dim == 216:
        pose_array = pose_vec.reshape(num_examples, -1, 9)
    else:
        print('please provide either 72-dim pose vector or 216-dim rot matrix')
        return
    useful_pose_array = pose_array[:, useful_joints_idx, :]
    return useful_pose_array.reshape(num_examples, -1)

def row(A):
    return tf.reshape(A, (1, -1))

def col(A):
    return tf.reshape(A, (-1, 1))

def sparse2tfsparse(sparse_matrix):
    '''
    turn a scipy sparse csr_matrix into a tensorflow sparse matrix
    '''
    sparse_matrix = sparse_matrix.tocoo()
    indices = np.column_stack((sparse_matrix.row, sparse_matrix.col))
    sparse_matrix = tf.SparseTensor(indices, sparse_matrix.data, sparse_matrix.shape)
    sparse_matrix = tf.sparse_reorder(sparse_matrix)
    return sparse_matrix

def pose2rot(pose):
    '''
    use Rodrigues transformation to turn pose vector to rotation matrix
    args:
        pose: [num_examples, 72], unraveled version of the pose vector in axis angle representation (24*3)
    returns:
        rot_all: rot matrix, [num_examples, 216], (216=24*3*3)
    '''
    from cv2 import Rodrigues
    num_examples = pose.shape[0]
    pose = pose.reshape(num_examples, -1, 3)

    rot_all = [np.array([Rodrigues(pp[i, :])[0] for i in range(pp.shape[0])]).ravel() for pp in pose]
    rot_all = np.array(rot_all)
    return rot_all

def rot2pose(rot):
    '''
    use Rodrigues transformation to turn rotation matrices into pose vector
    args:
        rot: [num_examples, 216], unraveled version of the 3x3 rot matrix (216=24 joints * 3*3)
    returns:
        pose_vec: pose vector [num_examples, 72], pose vector in axis angle representation (72=24*3)
    '''
    from cv2 import Rodrigues
    num_examples = rot.shape[0]
    rot = rot.reshape(num_examples, -1, 9)
    pose_vec = [np.array([Rodrigues(rr[i, :].reshape(3,3))[0] for i in range(rr.shape[0])]).ravel() for rr in rot]
    pose_vec = np.array(pose_vec)

    return pose_vec


'''

Following: TensorFlow implementation of psbody.mesh.geometry.tri_normals

'''

def TriNormals(v, f):
    return NormalizedNx3(TriNormalsScaled(v, f))

def TriNormalsScaled(v, f):
    edge_vec1 = tf.reshape(TriEdges(v, f, 1, 0), (-1, 3))
    edge_vec2 = tf.reshape(TriEdges(v, f, 2, 0), (-1, 3))
    return tf.cross(edge_vec1, edge_vec2)

def NormalizedNx3(v):
    v = tf.reshape(v, (-1, 3))
    ss = tf.reduce_sum(tf.square(v), axis=1)
    # prevent zero division
    indices = tf.equal(ss, 0.)
    mask = tf.cast(indices, ss.dtype)  # a mask, 1 where norms==0, 0 otherwise
    norms = tf.add(ss, mask)
    s = tf.sqrt(norms)
    return tf.reshape(tf.divide(v, col(s)), [-1])

def TriEdges(v, f, cplus, cminus):
    assert(cplus >= 0 and cplus <= 2 and cminus >= 0 and cminus <= 2)
    return _edges_for(v, f, cplus, cminus)

def _edges_for(v, f, cplus, cminus):
    # return (
    #     v.reshape(-1, 3)[f[:, cplus], :] -
    #     v.reshape(-1, 3)[f[:, cminus], :]).ravel()

    ind_plus, ind_minus = f[:, cplus], f[:, cminus]
    ind_plus = tf.expand_dims(ind_plus, 1)
    ind_minus = tf.expand_dims(ind_minus, 1)
    v = tf.reshape(v, (-1, 3))

    t = tf.gather_nd(v, ind_plus) - tf.gather_nd(v, ind_minus)
    return tf.reshape(t, [-1])