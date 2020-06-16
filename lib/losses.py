'''
implemnetation of edge and normal losses
'''

import numpy as np
import tensorflow as tf
from .utils import sparse2tfsparse, TriNormalsScaled, TriNormals

def edge_loss_calc(pred, gt, vpe):
    '''
    Calculate edge loss measured by difference in the length
    args:
        pred: prediction, [batch size, num_verts (6890), 3]
        gt: ground truth, [batch size, num_verts (6890), 3]
        vpe: SMPL vertex-edges correspondence table, [20664, 2]
    returns:
        edge_obj, an array of size [batch_size, 20664], each element of the second dimension is the
        length of the difference vector between corresponding edges in GT and in pred
    '''
    # get vectors of all edges, have size (batch_size, 20664, 3)
    edges_vec = lambda x: tf.gather(x,vpe[:,0],axis=1) -  tf.gather(x,vpe[:,1],axis=1)
    edge_diff = edges_vec(pred) -edges_vec(gt) # elwise diff between the set of edges in the gt and set of edges in pred
    edge_obj = tf.norm(edge_diff, ord='euclidean', axis=-1)

    return tf.reduce_mean(edge_obj)

def face_normal_loss_calc(pred, gt, f):
    '''
    This version works as long as the values feed into pred is *not all zeros* (otherwise will produce nan),
    but this situation should never happen in evaluation/test phase, because if a batch is all filled with 0 placeholders,
    this batch will not be created.
    '''
    batch_size = int(gt.get_shape()[0])
    f_tensor = tf.cast(f.astype('int32'), tf.int32)
    normals_pred = tf.stack([estimate_face_normals(pred[x], f, f_tensor) for x in range(batch_size)])
    normals_gt = tf.stack([estimate_face_normals(gt[x], f, f_tensor) for x in range(batch_size)])

    zeros = tf.zeros(shape=tf.shape(normals_gt))

    idx_nonzeros = tf.where(tf.norm(tf.norm(normals_gt - zeros, axis=-1), axis=-1) > 1e-14)

    cos = tf.reduce_sum(tf.multiply(normals_pred, normals_gt), axis=-1)
    cos_abs = tf.abs(cos)
    normal_loss = 1 - cos_abs

    result = tf.gather_nd(normal_loss, idx_nonzeros)
    return tf.reduce_mean(result)

def estimate_face_normals(v, f_arr, f_tensor):
    face_normals = tf.reshape(TriNormals(v, f_tensor),(-1, 3)) # this is *unnormalized* face normals (abs != 1), if you need normalized norms, use TriNormals()
    face_normals = tf.cast(face_normals, tf.float32)
    return face_normals

def estimate_vertex_normals(v, f_arr, f_tensor):
    '''
    TF implementation of psbody.mesh.mesh.estimate_vertex_normals
    calculate *scaled* (norm==1) normals of all the vertices v
    args:
        v: vertices
        f_arr: faces, nparray
        f_tensor: same faces as f_arr, but stored in a tensor
    returns:
        array of surface normals, shape of (#verts, 3), already centered and scaled (abs=1),
        i.e. only indicates surface normal directions
    '''
    face_normals = tf.reshape(TriNormals(v, f_tensor),(-1, 3)) # this is *unnormalized* face normals (abs != 1), if you need normalized norms, use TriNormals()
    face_normals = tf.cast(face_normals, tf.float32)
    ftov = faces_by_vertex(v, f_arr, as_sparse_matrix=True) # scipy sparse tensor
    ftov = ftov.astype('float32')
    ftov = sparse2tfsparse(ftov)
    non_scaled_normals = tf.sparse_tensor_dense_matmul(ftov, face_normals)
    norms = tf.norm(non_scaled_normals, axis=1)

    indices = tf.equal(norms, 0.0)
    mask = tf.cast(indices, norms.dtype) # a mask, 1 where norms==0, 0 otherwise
    norms = tf.add(norms, mask)

    normalized_vert_normals = tf.transpose(tf.divide(tf.transpose(non_scaled_normals), norms))

    return normalized_vert_normals


def faces_by_vertex(v, f, as_sparse_matrix=False):
    # Returns a list of #num_verts elements, each element is again a list, indicating which faces are using this vertex
    import scipy.sparse as sp
    if not as_sparse_matrix:
        faces_by_vertex = [[] for i in range(len(v))]
        for i, face in enumerate(f):
            faces_by_vertex[face[0]].append(i)
            faces_by_vertex[face[1]].append(i)
            faces_by_vertex[face[2]].append(i)
    else:
        row = f.flatten()
        col = np.array([range(f.shape[0])] * 3).T.flatten()
        data = np.ones(len(col))
        faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(v.shape[0], f.shape[0]))
    return faces_by_vertex

