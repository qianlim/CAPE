import tensorflow as tf
import scipy.sparse
import numpy as np
import os, time, collections, shutil
from . import losses, utils
from .mesh_sampling import rescale_L
import tqdm
import trimesh

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class base_model(object):
    # a set of basic methods
    def __init__(self, L, D, U, F=None, K=None, p=None, nz=18, loss='l1', nn_input_channel=3,
                 filter='chebyshev5', activation='b1leakyrelu', pool='poolwT',
                 unpool='poolwT', num_epochs=60, lr=0.008, decay_rate=0.99,
                 optimizer='sgd', decay_steps=None, momentum=0.9, cond_dim=0, nz_cond=0,
                 regularization=0, batch_size=32, seed=123,
                 lambda_recon=1.0, lambda_edge=0.0, lambda_latent=1e-3,
                 restart=False, name='', loss_mask=None):
        try:
            tf.random.set_random_seed(seed)
        except:
            tf.set_random_seed(seed)
        self.input_num_verts = L[0].shape[0]
        self.nn_input_channel = nn_input_channel

        self.name = name
        self.restart = restart
        self.Laplacian, self.Downsample_mtx, self.Upsample_mtx, self.p = L, D, U, p
        self.out_channels = F
        self.poly_order = K
        self.which_loss = loss
        self.num_epochs, self.learning_rate = num_epochs, lr
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization = regularization
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.fc_regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization)
        self.plot_latent = False

        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.verts_ref = trimesh.load(os.path.join(self.script_path, '..', 'data/template_mesh.obj'), process=False).vertices
        self.vpe = np.load(os.path.join(self.script_path, '..','data/edges_smpl.npy')) # vertex per edge

        if loss_mask == 'binary':
            print('Using loss mask on clothing related body parts...')
            mask = np.load(os.path.join(self.script_path, 'data','loss_mask_binary.npy'))
            self.loss_mask = np.repeat(mask[np.newaxis, :,:], batch_size, axis=0)
        else:
            self.loss_mask = 1.0

        self.nz = nz
        self.cond_dim = cond_dim
        self.nz_cond = nz_cond

        self.filter = getattr(self, filter)
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, activation)
        self.pool = getattr(self, pool)
        self.unpool = getattr(self, unpool)

        self.lambda_l1, self.lambda_edge, self.lambda_latent = lambda_recon, lambda_edge, lambda_latent

    '''
    following: building components for the network
    '''
    def chebyshev5(self, x, L, Fout, K, trainable=True):
        # L here is the Laplacian
        N, M, Fin = x.get_shape()  # batch_size, num_vertices, channels_in
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        # recurrent relationship of Chebyshev polynomial
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)  # L: M x M; therefore shape kept, x1 shape = [M, Fin * N]
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin * K, Fout], trainable=trainable)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1leakyrelu(self, x, trainable=True):
        """Bias and leaky ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], trainable=trainable)
        return tf.nn.leaky_relu(x + b)

    def b1tanh(self, x):
        """Bias and tanh. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)])
        return tf.nn.tanh(x + b)

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)])
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)])
        return tf.nn.relu(x + b)

    def poolwT(self, x, L):
        '''
        pooling and unpooling layer using the pre-computed up/downsampling matrices, introduced in the CoMA paper
        args:
            x: input to the pool/unpool layer, shape:[batch_size, num_vertices, feature_dim]
            L: the up- /down-sampling matrix (U or D) computed from mesh_sampling using qslim algorithm,
                  of shape [#verts next layer, #verts this layer]
        '''
        Mp = L.shape[0]  # L: number of outputs vertices
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin * N])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin
        return x

    def cnp(self, x, i, name):
        '''
        Convolution, Non-linearity, Pooling (downsampling)
        args:
            x: input to the layer, [batch_size, num_vertices, feature_dim]
            i: layer_index, input layer to the whole network has index 0
            name: layer name for creating variable scope
        '''
        with tf.variable_scope(name):
            with tf.name_scope('filter'):
                x = self.filter(x, self.Laplacian[i], self.out_channels[i], self.poly_order[i])
            with tf.name_scope('bias_relu'):
                x = self.brelu(x)
            with tf.name_scope('pooling'):
                x = self.pool(x, self.Downsample_mtx[i])
            print('{}: ({}, {}), K={}'.format(name, int(x.get_shape()[1]), self.out_channels[i],
                                              self.poly_order[i]))
        return x

    def udn(self, x, out_channels, i, name):
        '''
        Unpool (upsampling), Deconv, Nonlinearity
        args:
            x: input to the layer, [batch_size, num_vertices, feature_dim]
            i: layer_index, input layer to the whole network has index 0
            out_channels: list of output channels
            name: layer name
        '''
        with tf.variable_scope(name):
            with tf.name_scope('unpooling'):
                x = self.unpool(x, self.Upsample_mtx[-i-1])
            with tf.name_scope('filter'):
                x = self.filter(x, self.Laplacian[-i-2], out_channels[-i-1], self.poly_order[-i-1])
            with tf.name_scope('bias_relu'):
                x = self.brelu(x)
            print('{}: ({}, {}), K={}'.format(name, int(x.get_shape()[1]), out_channels[-(i + 1)],
                                              self.poly_order[-(i + 1)]))
        return x

    def vae_sampling(self, z_mean, z_logvar):
        eps = tf.random_normal([self.batch_size, int(self.nz)], 0, 1, dtype=tf.float32)
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps))
        return z

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, trainable=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial, trainable=trainable)
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, trainable=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial,trainable=trainable)
        tf.summary.histogram(var.op.name, var)
        return var


class CAPE(base_model):
    '''
    Mesh CVAE + discriminator
    Takes 2 conditions: body pose and clothing type (one-hot encoding)
    '''
    def __init__(self, L, D, U, L_d, D_d, lr_scaler, lambda_gan, use_res_block, use_res_block_dec, nz_cond2,
                 cond2_dim, Kd, n_layer_cond=1, cond_encoder=True, reduce_dim=True, affine=False,
                 lr_warmup=False, optim_condnet=True, **kwargs):
        super(CAPE, self).__init__(L, D, U, **kwargs)
        self.Laplacian_d, self.Downsample_mtx_d = L_d, D_d
        self.Laplacian, self.Downsample_mtx, self.Upsample_mtx= L, D, U
        self.poly_order_d= [Kd] * len(self.out_channels) # small K -> small receptive field for each patch
        self.use_res_block = use_res_block # use residual block instead of simple graph conv for encoder
        self.use_res_block_dec = use_res_block_dec # user residual block for decoder

        self.nz_cond2 = nz_cond2
        self.cond2_dim = cond2_dim
        self.n_layer_cond = n_layer_cond
        self.cond_encoder = cond_encoder
        self.optim_condnet = optim_condnet

        self.reduce_dim = reduce_dim
        self.affine = affine

        if self.reduce_dim > 0:
            self.reduce_rate = self.out_channels[-1] // self.reduce_dim
        elif self.reduce_dim == 0:
            self.reduce_rate = 1
        else:
            raise ValueError('reduce dim must be greater than 0!')

        self.lr_g = self.learning_rate
        self.lr_d = self.learning_rate * lr_scaler
        self.lambda_gan = lambda_gan
        self.lr_warmup = lr_warmup


    def build_graph(self, input_num_verts, nn_input_channel, phase='train'):
        """Build the computational graph of the model, incl. the architecture, loss, train op etc."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.ph_data_g = tf.placeholder(tf.float32, (self.batch_size, input_num_verts, nn_input_channel), 'data_g')
                self.ph_data_d = tf.placeholder(tf.float32, (self.batch_size, input_num_verts, nn_input_channel), 'data_d')

                self.ph_cond_g = tf.placeholder(tf.float32, (self.batch_size, self.cond_dim), 'condition_g')
                self.ph_cond2_g = tf.placeholder(tf.float32, (self.batch_size, self.cond2_dim), 'condition2_g')

                self.ph_cond_d = tf.placeholder(tf.float32, (self.batch_size, self.cond_dim), 'condition_d')
                self.ph_cond2_d = tf.placeholder(tf.float32, (self.batch_size, self.cond2_dim), 'condition2_d')

                # recon gt for vae
                self.ph_gt = tf.placeholder(tf.float32, (self.batch_size, input_num_verts, nn_input_channel),'gt')
            # a simple NN for condition vector
            self.y_latent_g = self.condition(self.ph_cond_g, nz_cond=self.nz_cond, name='pose', nlayers=2, reuse=False)
            self.y2_latent_g = self.condition(self.ph_cond2_g, nz_cond=self.nz_cond2, name='clo_label',
                                              nlayers=self.n_layer_cond, reuse=False)

            self.y_latent_d = self.condition(self.ph_cond_d, nz_cond=self.nz_cond, name='pose', nlayers=2, reuse=True)
            self.y2_latent_d = self.condition(self.ph_cond2_d, nz_cond=self.nz_cond2, name='clo_label',
                                              nlayers=self.n_layer_cond, reuse=True)

            self.is_train = tf.placeholder(dtype=bool, shape=[], name='is_training')

            # generator, i.e. vae
            g_outputs = self.generator(x=self.ph_data_g, use_res_block=self.use_res_block,
                                       use_res_block_dec=self.use_res_block_dec, cond_encoder=self.cond_encoder,
                                       is_train=self.is_train)
            # patch discriminator
            d_logits_real= self.discriminator(x=self.ph_data_d, y=self.y_latent_d,
                                                        y2=self.y2_latent_d)
            d_logits_fake = self.discriminator(x=g_outputs, y=self.y_latent_g,
                                                        y2=self.y2_latent_g, reuse=True)


            self.op_loss_g, self.op_loss_d, self.op_loss_average_g, self.op_loss_average_d  =\
                self.loss(g_outputs=g_outputs, g_gt=self.ph_gt, d_logits_real=d_logits_real,
                          d_logits_fake=d_logits_fake)

            self.op_prediction = g_outputs

            if phase == 'train':
                self.op_train_g, self.op_train_d, self.op_train_q = self.training(loss_g=self.op_loss_g,
                                                                     loss_d=self.op_loss_d,
                                                                     optimizer=self.optimizer,
                                                                     lr_g=self.lr_g, lr_d=self.lr_d,
                                                                     momentum=self.momentum,
                                                                     decay_steps=self.decay_steps,
                                                                     decay_rate=self.decay_rate)
            else: # test or demo
                # for test phase
                print('\n>>>>>>For generative experiments:')
                self.ph_z = tf.placeholder(tf.float32, (self.batch_size, int(self.nz)), 'z')
                self.ph_z_total = tf.placeholder(tf.float32,
                                                 (self.batch_size, int(self.nz) + self.nz_cond + self.nz_cond2),
                                                 'z_total')  # concat z and z_cond

                self.ph_y_latent = tf.placeholder(tf.float32, (self.batch_size, self.nz_cond), 'cond_latent')
                self.ph_y2_latent = tf.placeholder(tf.float32, (self.batch_size, self.nz_cond2), 'cond2_latent')

                # self.op_prediction = g_outputs
                self.op_cond_latent = self.condition(y=self.ph_cond_g, nz_cond=self.nz_cond, name='pose', nlayers=2, reuse=True)
                self.op_cond2_latent = self.condition(y=self.ph_cond2_g, nz_cond=self.nz_cond2, name='clo_label',
                                                      nlayers=self.n_layer_cond, reuse=True)

                with tf.variable_scope('generator', reuse=True):
                    self.op_vae_mean, self.op_vae_var = self.encoder(x=self.ph_data_g,
                                                                     y=self.op_cond_latent,
                                                                     y2=self.op_cond2_latent,
                                                                     reuse=True,
                                                                     use_res_block=self.use_res_block,
                                                                     use_cond=self.cond_encoder)

                    self.op_decoder = self.decoder_cond_vert(x=self.ph_z_total,
                                                             y=self.ph_y_latent,
                                                             y2=self.ph_y2_latent,
                                                             reuse=True, use_res_block=self.use_res_block_dec,
                                                             is_train=False)

            self.op_init = tf.global_variables_initializer()
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)


    def loss(self, g_outputs, g_gt, d_logits_real, d_logits_fake, smooth=0.1):
        with tf.name_scope('loss'):
            with tf.name_scope('reconstruction_loss'):
                if self.which_loss == "l1":
                    self.recon_loss = tf.losses.absolute_difference(predictions=g_outputs, labels=g_gt,
                                                                    weights=self.loss_mask,
                                                                    reduction=tf.losses.Reduction.MEAN)

                elif self.which_loss == "huber":
                    self.recon_loss = tf.losses.huber_loss(labels=g_gt, predictions=g_outputs,
                                                           reduction=tf.losses.Reduction.MEAN,
                                                           weights=self.loss_mask, delta=0.1)
                else:
                    self.recon_loss = tf.losses.mean_squared_error(predictions=g_outputs, labels=g_gt,
                                                                   weights=self.loss_mask,
                                                                   reduction=tf.losses.Reduction.MEAN)
            with tf.name_scope('latent_loss'):
                latent_loss = -0.5 * tf.reduce_sum(1 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar), axis=1)
                self.latent_loss = tf.reduce_mean(latent_loss)

            with tf.name_scope('edge_loss'):
                self.edge_loss = losses.edge_loss_calc(pred=g_outputs+self.verts_ref, gt=g_gt+self.verts_ref, vpe=self.vpe)

            with tf.name_scope('fc_regularization'):
                self.fc_regularization_g = self.regularization * tf.losses.get_regularization_loss(scope='generator')
                self.fc_regularization_d = self.regularization * tf.losses.get_regularization_loss(scope='discriminator')

            with tf.name_scope('gan_loss'):
                # 'soft labels' for D: 1-->0.9, 0-->0.1
                d_labels_real = tf.ones_like(d_logits_real) * (1 - smooth)
                d_labels_fake = tf.zeros_like(d_logits_fake) + smooth
                g_labels = tf.ones_like(d_logits_fake) * (1 - smooth)

                self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=g_labels))
                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=d_labels_real))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=d_labels_fake))
                self.loss_d = tf.add(d_loss_real, d_loss_fake)

            with tf.name_scope('total_loss'):
                loss_g = self.loss_g * self.lambda_gan + self.recon_loss * self.lambda_l1 +\
                         + 0. + self.edge_loss * self.lambda_edge + \
                         self.latent_loss * self.lambda_latent + self.fc_regularization_g

                loss_d = self.loss_d * self.lambda_gan + self.fc_regularization_d

            # Summaries for TensorBoard.
            tf.summary.scalar('reconstruction_loss', self.recon_loss)
            tf.summary.scalar('latent_loss', self.latent_loss)
            tf.summary.scalar('fc_regularization_g', self.fc_regularization_g)
            tf.summary.scalar('fc_regularization_d', self.fc_regularization_d)
            tf.summary.scalar('gan_loss_generator', self.loss_g)
            tf.summary.scalar('gan_loss_discriminator', self.loss_d)

            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([loss_g, loss_d])
                tf.summary.scalar('loss_g', averages.average(loss_g))
                tf.summary.scalar('loss_d', averages.average(loss_d))
                with tf.control_dependencies([op_averages]):
                    loss_average_g = tf.identity(averages.average(loss_g), name='control_g')
                    loss_average_d = tf.identity(averages.average(loss_d), name='control_d')

            return loss_g, loss_d, loss_average_g, loss_average_d


    def training(self, loss_g, loss_d, lr_g, lr_d, optimizer, decay_steps, decay_rate=0.95, momentum=0.9, warmup_duration=8):
        '''define optimizer, global step, learning rate policy etc.'''
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.global_step = global_step

            if self.lr_warmup:
                # sevearl epochs of lr warm up
                warmup_steps = int(decay_steps * warmup_duration)
                print('using lr warm-up, warmup steps {}'.format(warmup_steps))
                warmup_lr_g = (lr_g * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
                warmup_lr_d = (lr_d * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

                # then decay
                lr_g_decay = tf.train.exponential_decay(lr_g, global_step-warmup_steps, int(decay_steps), decay_rate, staircase=True)
                lr_d_decay = tf.train.exponential_decay(lr_d, global_step-warmup_steps, int(decay_steps), decay_rate, staircase=True)

                lr_g = tf.cond(global_step < warmup_steps, lambda: warmup_lr_g, lambda: lr_g_decay)
                lr_d = tf.cond(global_step < warmup_steps, lambda: warmup_lr_d, lambda: lr_d_decay)

            else:
                lr_g = tf.train.exponential_decay(lr_g, global_step, int(decay_steps), decay_rate, staircase=True)
                lr_d = tf.train.exponential_decay(lr_d, global_step, int(decay_steps), decay_rate, staircase=True)

            tf.summary.scalar('lr_g', lr_g)
            tf.summary.scalar('lr_d', lr_d)

            # Optimizer.
            if optimizer == 'adam':
                opt_g = tf.train.AdamOptimizer(learning_rate=lr_g)
                opt_d = tf.train.AdamOptimizer(learning_rate=lr_d)
            else:
                opt_g = tf.train.MomentumOptimizer(lr_g, momentum)
                opt_d = tf.train.MomentumOptimizer(lr_d, momentum)

            if self.optim_condnet:
                vars_g = [v for v in tf.trainable_variables() if v.name.startswith('generator') or 'condition' in v.name]
            else:
                vars_g = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

            grads_g, variables_g = zip(*opt_g.compute_gradients(loss_g, var_list=vars_g))
            grads_g, _ = tf.clip_by_global_norm(grads_g, 5.0) # prevent gradient explosion
            op_gradients_g = opt_g.apply_gradients(zip(grads_g, variables_g), global_step=global_step)

            vars_d = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
            grads_d, variables_d = zip(*opt_d.compute_gradients(loss_d, var_list=vars_d))
            grads_d, _ = tf.clip_by_global_norm(variables_d, 5.0)
            op_gradients_d = opt_d.apply_gradients(zip(grads_d, variables_d), global_step=global_step)

            # apply gradients first, then return learning rate
            with tf.control_dependencies([op_gradients_g, op_gradients_d]):
                op_train_g = tf.identity(lr_g, name='control_g')
                op_train_d = tf.identity(lr_d, name='control_d')

            return op_train_g, op_train_d, None

    '''
    Network components
    '''
    def condition(self, y, name, nz_cond, nlayers=1, reuse=False, trainable=True):
        '''
        Simple 1- or 2-layerd FC network to embed the condition vectors
        args:
            y: "raw" condition vector, [batch_size, dim_condition]
            name: layer name
            nz_cond: dimension of the output condition embedding
            n_layers: int, number of fc layers, if set to >=2 will build a 2 layer network, otherwise the network
                      will be just a linear layer.
            reuse: whether reuse the weights
            trainable: whether the network is trainable
        '''
        y_dim = int(y.get_shape()[-1])
        with tf.variable_scope('condition_{}'.format(name), reuse=reuse):
            if nlayers == 1:
                with tf.variable_scope('fc1', reuse=reuse):
                    print('condition_{}_fc1: ({}, {})'.format(name, int(y.shape[-1]), nz_cond))
                    y = tf.layers.dense(y, nz_cond, kernel_regularizer=self.fc_regularizer, trainable=trainable)
            else:
                if nz_cond < y_dim // 2:
                    n_out_fc1 = y_dim // 2
                elif nz_cond < y_dim * 2:
                    n_out_fc1 = y_dim
                else:
                    n_out_fc1 = nz_cond // 2
                with tf.variable_scope('fc1', reuse=reuse):
                    print('condition_{}_fc1: ({}, {})'.format(name, int(y.shape[-1]), n_out_fc1))
                    y = tf.layers.dense(y, n_out_fc1, activation=tf.nn.leaky_relu,
                                        kernel_regularizer=self.fc_regularizer, trainable=trainable)
                with tf.variable_scope('fc2', reuse=reuse):
                    print('condition_{}_fc2: ({}, {})'.format(name, int(y.shape[-1]), nz_cond))
                    y = tf.layers.dense(y, nz_cond, kernel_regularizer=self.fc_regularizer, trainable=trainable)
        return y


    def encoder(self, x, y, y2, use_res_block=False, use_cond=True, reuse=False):
        '''
        Encode the input mesh verts to the latent space
        args:
            x: input mesh verts, [batch_size, N_verts, 3]
            y: embedding of the first condition, i.e. output from the first small condition network
            y2: embedding of the second condition,  i.e. output from the second small condition network
            use_res_block: use residual block instead of plain conv
            use_cond: apply condition to the encoder
            reuse: whether to reuse weights
        returns:
            z_mean, z_var: mean and log variance of the predicted latent code
        '''
        #
        # if t is not None:
        #     x = tf.concat([x, t], -1)
        #     print('Using template mesh in encoder')
        #     print(x.shape.as_list())
        if use_cond: # concat condition vector to every vertex
            y_expanded = self.fit_cond_dim(x, y)
            y2_expanded = self.fit_cond_dim(x, y2)
            data_and_cond = tf.concat([x, y_expanded, y2_expanded], -1)
            x = data_and_cond

        print('------------Encoder------------')
        with tf.variable_scope('encoder', reuse=reuse):
            # conv layers
            for i in range(len(self.out_channels)):
                if use_res_block:
                    x = self.res_block(x, i, 'encoder_resblock{}'.format(i + 1))
                else:
                    x = self.cnp(x, i, 'encoder_conv{}'.format(i + 1))

            # one more layer of '1x1' conv to reduce #channels, so that the final fc layer of the encoder
            # is smaller, thereby preventing overfitting
            if self.reduce_dim > 0:
                with tf.variable_scope('1x1-conv'):
                    x = self.filter(x, self.Laplacian[-1], self.out_channels[-1] // self.reduce_rate, K=1)
                    print('{}: ({}, {}), K={}'.format('encoder_1x1conv', int(x.get_shape()[1]), int(x.get_shape()[2]), 1))

            x = tf.reshape(x, [self.batch_size, -1])  # N x MF
            with tf.variable_scope('fc_mean'):
                print('encoder_fc_mean: ({}, {})'.format(int(x.shape[-1]), int(self.nz)))
                z_mean = tf.layers.dense(x, int(self.nz), kernel_regularizer=self.fc_regularizer)
            with tf.variable_scope('fc_var'):
                print('encoder_fc_logvar: ({}, {})'.format(int(x.shape[-1]), int(self.nz)))
                z_var = tf.layers.dense(x, int(self.nz), kernel_regularizer=self.fc_regularizer)
        return z_mean, z_var


    def decoder_cond_vert(self, x, y, y2, reuse=False, use_res_block=False, trainable=True, is_train=False):
        '''
        Decoder that applies condition to all vertices
        args:
            x: the full latent code, sampled z concatenated with condition vector, [batch_size, nz + nz_cond + nz_cond2]
            y: embedding of the first condition, [batch_size, nz_cond]
            y2: embedding of the second condition, [batch_size, nz_cond2]
            use_res_block: use residual block instead of plain conv
            trainable: flag applied to several layers, setting wheter they are trainable
            is_train: trainiing flag for the normalization layers
        returns:
            Generated vertices
        '''
        print('------------Decoder------------')
        with tf.variable_scope('decoder', reuse=reuse):
            with tf.variable_scope('fc1'):
                out_nodes = int(self.p[-1] * self.out_channels[-1]) // self.reduce_rate
                print('decoder_fc1: ({}, {})'.format(int(x.shape[-1]), out_nodes))
                x = tf.layers.dense(x, out_nodes, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=self.fc_regularizer, trainable=trainable)  # N x MFc
            x = tf.reshape(x, [self.batch_size, int(self.p[-1]), -1])  # N x M x Fe()
            if self.reduce_dim > 0:
                # one more layer of '1x1' conv to reduce #channels, thereby dimension of flatten layer, thereby #fc params
                with tf.variable_scope('1x1-conv'):
                    x = self.filter(x, self.Laplacian[-1], self.out_channels[-1], K=1)
                    print('{}: ({}, {}), K={}'.format('decoder_1x1conv', int(x.get_shape()[1]), int(x.get_shape()[2]), 1))

            y_expanded = self.fit_cond_dim(x, y)
            y2_expanded = self.fit_cond_dim(x, y2)
            data_and_cond = tf.concat([x, y_expanded, y2_expanded], -1) # append condition features to every node feature
            x = data_and_cond

            for i in range(len(self.out_channels)):
                if use_res_block:
                    if not self.affine:
                        x = self.res_block_decoder(x, i, 'decoder_resblock_cmr{}'.format(i + 1), reuse=reuse, is_train=is_train)
                    else:
                        x = self.res_block_affine(x, i, 'decoder_resblock_affine{}'.format(i + 1))
                else:
                    x = self.udn(x, self.out_channels, i, 'decoder_conv{}'.format(i + 1))

                # concat condition vector to each vertex
                y_expanded = self.fit_cond_dim(x, y)
                y2_expanded = self.fit_cond_dim(x, y2)
                data_and_cond = tf.concat([x, y_expanded, y2_expanded], -1) # append condition features to every node feature
                x = data_and_cond

            with tf.variable_scope('outputs'):
                x = self.filter(x, self.Laplacian[0], int(self.nn_input_channel), self.poly_order[0], trainable=trainable)
                N, M, F = x.get_shape()
                print('{}: ({}, {}), K={}\n'.format('decoder_output', M, F, self.poly_order[0]))
                bias = self._bias_variable([1, M, F], trainable=trainable) # one bias per vertex per channel
                x += bias
        return x


    def generator(self, x, use_res_block=None, use_res_block_dec=None, cond_encoder=True, is_train=False):
        """
        Encoder followed by decoder
        args:
            x: input verts
            use_res_block: whether or not use residual block for encoder, 1 or 0
            use_res_block_dec: whether or not use residual block for decoder, 1 or 0
            cond_encoder: whether or not apply condition to encoder
            is_train: trainiing flag for the normalization layers
        returns:
            x_hat: the reconstructed / generated mesh vertices, [batch_size, N_verts, 3]
        """
        print('\n------------[Generator]------------')
        with tf.variable_scope('generator'):
            self.z_mean, self.z_logvar = self.encoder(x, self.y_latent_g, self.y2_latent_g,
                                                      use_res_block=use_res_block, use_cond=cond_encoder)

            z_sample = self.vae_sampling(self.z_mean, self.z_logvar)

            self.z_sample = z_sample

            z_total = tf.concat([z_sample, self.y_latent_g, self.y2_latent_g], axis=1)
            x_hat = self.decoder_cond_vert(z_total, y=self.y_latent_g, y2=self.y2_latent_g,
                                           use_res_block=use_res_block_dec, is_train=is_train)

        return x_hat


    def discriminator(self, x, y, y2, reuse=False):
        '''
        Mesh patch discriminator, inspriation taken from the Pix2pix paper, i.e. instead of returning a
        single real/fake scalar, now return the real/fake prediction for each vertex in the downsampled
        mesh. Each vertex in the downsampled mesh corresponds to a receptive field (patch) in the
        original high-res mesh.
        args:
            x: vertex sets to be discriminatoed, [batch_size, N_verts, 3]
            y: embedding of first condition, [batch_size, nz_cond]
            y2: embedding of second condition, [batch_size, nz_cond2]
            reuse: flag for reusing weights
        :return pred_map: real/false prediction for each vertex in the  [batch, N_verts_downsampled, 1]
        '''
        print('\n----------[Discriminator]----------')
        # append condition to each vertex
        y_expanded = self.fit_cond_dim(x, y)
        y2_expanded = self.fit_cond_dim(x, y2)
        data_and_cond = tf.concat([x, y_expanded, y2_expanded], -1)
        x = data_and_cond

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('shared'):
                for i in range(len(self.Downsample_mtx_d)):
                    layer_name = 'conv{}'.format(i + 1)
                    x = self.cnp_d(x, i, layer_name)

            with tf.variable_scope('prediction_map'):
                print('{}: ({}, {}), K={}'.format('pred_map', int(x.get_shape()[1]), 1, self.poly_order_d[-1]))
                pred_map = self.filter(x, self.Laplacian_d[-1], 1, self.poly_order[-1])

        return pred_map


    def gn(self, x, is_train, name, norm_type='group', G=32, eps=1e-5, reuse=False):
        '''
        Normalization layers
        '''
        with tf.variable_scope(name, reuse=reuse):
            if norm_type == 'none':
                output = x
            elif norm_type == 'batch':
                output = tf.contrib.layers.batch_norm(
                    x, center=True, scale=True, decay=0.999,
                    is_training=is_train, updates_collections=None
                )
            elif norm_type == 'group':
                # tranpose: [bs, v, c] to [bs, c, v] following the GraphCMR paper
                x = tf.transpose(x, [0, 2, 1])
                N, C, V = x.get_shape().as_list() # V: num bertices
                G = min(G, C)
                x = tf.reshape(x, [-1, G, C // G, V])
                mean, var = tf.nn.moments(x, [2, 3], keep_dims=True)
                x = (x - mean) / tf.sqrt(var + eps)
                # per channel gamma and beta
                gamma = tf.get_variable('gamma', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
                beta = tf.get_variable('beta', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                gamma = tf.reshape(gamma, [1, C, 1])
                beta = tf.reshape(beta, [1, C, 1])

                output = tf.reshape(x, [-1, C, V]) * gamma + beta
                # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
                output = tf.transpose(output, [0, 2, 1])
            else:
                raise NotImplementedError
        return output


    def res_block(self, x_in, i, name):
        '''residual block for the encoder, as an alternative to plain graph conv layers (cnp() defined in base class)'''
        with tf.variable_scope(name):
            with tf.variable_scope('filter_1'):
                x_filtered_1 = self.filter(x_in, self.Laplacian[i], self.out_channels[i], self.poly_order[i])
            with tf.variable_scope('bias_relu_1'):
                x_relu_1 = self.brelu(x_filtered_1)
            with tf.variable_scope('filter_2'):
                x_filtered_2 = self.filter(x_relu_1, self.Laplacian[i], self.out_channels[i], self.poly_order[i])

            # when out_channel not match, first do zero padding on channel dimension to match #channels
            channels_in = x_in.get_shape()[-1]
            channels_filtered = x_filtered_2.get_shape()[-1]
            if channels_in != channels_filtered:
                with tf.variable_scope('1x1-conv'):
                    x_in  = self.filter(x_in, self.Laplacian[i], channels_filtered, K=1)
            with tf.variable_scope('addition'):
                x_filtered_2 += x_in
            with tf.variable_scope('bias_relu_2'):
                x_relu_2 = self.brelu(x_filtered_2)

            with tf.name_scope('pooling'):
                x_pooled = self.pool(x_relu_2, self.Downsample_mtx[i])

            print('{}: ({}, {}), K={}'.format(name, int(x_pooled.get_shape()[1]), int(x_pooled.get_shape()[2]),
                                              self.poly_order[i]))
        return x_pooled


    def res_block_decoder(self, x_in, i, name, reuse=False, is_train=False):
        '''residual block for the decoder, as an alternative to plain graph conv layers (udn() defined in base class),
        following the design of https://arxiv.org/pdf/1905.03244.pdf that uses group normalization.
        '''
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope('unpooling'):
                x_unpooled = self.unpool(x_in, self.Upsample_mtx[-i - 1])
            x = self.gn(x_unpooled, name='group_norm', is_train=is_train, reuse=reuse)
            x = tf.nn.relu(x)
            with tf.variable_scope('graph_linear_1'):
                x = self.filter(x, self.Laplacian[-i-2], self.out_channels[-i-1]//2, K=1)
            x = self.gn(x, name='group_norm_1', is_train=is_train, reuse=reuse)
            x = tf.nn.relu(x)
            with tf.variable_scope('graph_conv'):
                x = self.filter(x, self.Laplacian[-i-2], self.out_channels[-i-1]//2, self.poly_order[-i-1])
            x = self.gn(x, name='group_norm_2', is_train=is_train, reuse=reuse)
            x = tf.nn.relu(x)
            with tf.variable_scope('graph_linear_2'):
                x = self.filter(x, self.Laplacian[-i-2], self.out_channels[-i-1], K=1)

            channels_in = x_unpooled.get_shape()[-1]
            channels_filtered = x.get_shape()[-1]
            if channels_in != channels_filtered:
                with tf.variable_scope('graph_linear_input'):
                    x_unpooled  = self.filter(x_unpooled, self.Laplacian[-i-2], channels_filtered, K=1)

            x = x + x_unpooled

            print('{}: ({}, {}), K={}'.format(name, int(x.get_shape()[1]), int(x.get_shape()[2]),
                                              self.poly_order[i]))
        return x

    def res_block_affine(self, x, i, name):
        '''residual block that adds an affine transformation to the graph conv outputs
           see https://arxiv.org/abs/2004.02658
        '''
        with tf.variable_scope(name):
            with tf.name_scope('unpooling'):
                x = self.unpool(x, self.Upsample_mtx[-i - 1])
            with tf.variable_scope('graph_conv'):
                x_gc = self.filter(x, self.Laplacian[-i-2], self.out_channels[-i-1]//2, self.poly_order[-i-1])
            x_gc = tf.nn.relu(x_gc)

            channels_filtered = x_gc.get_shape()[-1]
            with tf.variable_scope('affine'):
                x_affine = self.filter(x, self.Laplacian[-i-2], channels_filtered, K=1)
            x = x_affine + x_gc
            print('{}: ({}, {}), K={}'.format(name, int(x.get_shape()[1]), int(x.get_shape()[2]),
                                              self.poly_order[i]))
        return x


    def cnp_d(self, x, i, name):
        '''
        similar to the cnp function in base class, this one is defined for discriminator separately,
        just to avoid getting puzzled by the layer indices
        '''
        with tf.variable_scope(name):
            with tf.name_scope('filter'):
                x = self.filter(x, self.Laplacian_d[i], self.out_channels[i], self.poly_order_d[i])
            with tf.name_scope('bias_relu'):
                x = self.brelu(x)
            with tf.name_scope('pooling'):
                x = self.pool(x, self.Downsample_mtx_d[i])
            print('{}: ({}, {}), K={}'.format(name, int(x.get_shape()[1]),  int(x.get_shape()[2]),
                                              self.poly_order_d[i]))
        return x


    def fit_cond_dim(self, x, y):
        '''
        Repeats a vector y so that it has the same spatial dimension as x, i.e. repeating y
        from shape [batch, C] to [batch_size, spatial_dim, C]
        Used for stacking condition y on the channels dimension at each vertex in the newtork
        args:
            x: graph vertex with features, [batch_size, num_vertices, dim_feature_x]
            y: condition vector, [batch_size, dim_feature_y]
        returns:
            repeated y vector of shape [batch_size, spatial_dim, C]
        '''
        x_shape = list(map(int, x.get_shape()))
        y_shape = list(map(int, y.get_shape()))

        batch_dim = x_shape[0]

        y = tf.reshape(y, [batch_dim, 1, y_shape[-1]])
        y_expanded = y * tf.ones([x_shape[0], x_shape[1], y_shape[-1]])

        return y_expanded

    '''
    Training and test
    '''
    def fit(self, data_wrapper):
        '''
        training loop
        '''
        train_data, train_cond, train_cond2, train_labels = \
            data_wrapper.vertices_train, data_wrapper.cond1_train, data_wrapper.cond2_train, data_wrapper.vertices_train
        val_data, val_cond, val_cond2, val_labels = \
            data_wrapper.vertices_val, data_wrapper.cond1_val, data_wrapper.cond2_val, data_wrapper.vertices_val

        num_steps_epoch = int(train_data.shape[0] / self.batch_size)
        num_steps = self.num_epochs * num_steps_epoch

        t_start = time.time()
        sess = tf.Session(graph=self.graph)

        if self.restart is not True:
            print('\n==========Loading from checkpoint {}...'.format(self.name))
            sess = self._get_session() # will load from checkpoint
            start_step = sess.run(self.global_step)
            end_step = start_step + num_steps
        else:
            if 'rmtree_protection' in self._get_path('checkpoints') or self._get_path('checkpoints').endswith('checkpoints/'):
                raise ValueError('Please provide an expriment name by setting the --name flag.')
            print('\n==========Start training from scratch...'.format(self.name))
            shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))
            sess.run(self.op_init)
            start_step = 1
            end_step = start_step + num_steps

        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        path = os.path.join(self._get_path('checkpoints'), 'model')

        losses = []
        indices_g = collections.deque()
        indices_d = collections.deque()

        for step in range(start_step, end_step):
            # make sure to have used all the samples before using one a second time.
            if len(indices_g) < self.batch_size:
                indices_g.extend(np.random.permutation(train_data.shape[0]))
            if len(indices_d) < self.batch_size:
                indices_d.extend(np.random.permutation(train_data.shape[0]))

            idx_g = [indices_g.popleft() for i in range(self.batch_size)]
            idx_d = [indices_d.popleft() for i in range(self.batch_size)]

            batch_data_g, batch_labels_g = train_data[idx_g], train_labels[idx_g]
            batch_data_d, batch_labels_d = train_data[idx_d], train_labels[idx_d]

            if type(batch_data_g) is not np.ndarray:
                batch_data_g = batch_data_g.toarray()  # convert sparse matrices
                batch_data_d = batch_data_d.toarray()

            feed_dict = {self.ph_data_g: batch_data_g, self.ph_gt: batch_labels_g,
                         self.ph_data_d: batch_data_d, self.is_train: True}

            batch_cond_g = train_cond[idx_g]
            batch_cond2_g = train_cond2[idx_g]
            batch_cond_d = train_cond[idx_d]
            batch_cond2_d = train_cond2[idx_d]

            feed_dict[self.ph_cond_g] = batch_cond_g
            feed_dict[self.ph_cond2_g] = batch_cond2_g
            feed_dict[self.ph_cond_d] = batch_cond_d
            feed_dict[self.ph_cond2_d] = batch_cond2_d

            learning_rate_g, loss_average_g = sess.run([self.op_train_g, self.op_loss_average_g], feed_dict) # train generator
            learning_rate_d, loss_average_d = sess.run([self.op_train_d, self.op_loss_average_d], feed_dict) # train discriminator
            if step % num_steps_epoch == 0 or step == num_steps:
                epoch = int(step * self.batch_size / train_data.shape[0])
                print('step {} / {} (epoch {} / {}):'.format(step, num_steps, epoch, self.num_epochs))

                print('  learning_rate_g = {:.2e}, loss_average_g = {:.2e}'.format(learning_rate_g, loss_average_g))
                print('  learning_rate_d = {:.2e}, loss_average_d = {:.2e}'.format(learning_rate_d, loss_average_d))
                string, recon_loss, latent_loss, edge_loss = self.evaluate(val_data, val_cond, val_cond2, val_labels, sess)
                losses.append(recon_loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s'.format(time.time() - t_start))
                #
                val_summary = tf.Summary()
                val_summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                val_summary.value.add(tag='validation/loss', simple_value=recon_loss)

                writer.add_summary(val_summary, step)

                self.op_saver.save(sess, path, global_step=step)

        writer.close()
        sess.close()
        t_step = (time.time() - t_start) / num_steps
        return losses, t_step

    def encode(self, data=None, cond=None, cond2=None):
        '''encode data into latent space, encode the two conditions into
        their embeddings. Used at test time.
        '''
        size = data.shape[0]
        z_mean_pred = [0] * size
        z_logvar_pred = [0] * size
        z_cond_pred = [0] * size
        z_cond2_pred = [0] * size

        sess = self._get_session(sess=None)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data

            batch_cond = np.zeros((self.batch_size, cond.shape[1]))
            batch_cond2 = np.zeros((self.batch_size, cond2.shape[1]))
            tmp_cond = cond[begin:end, :]
            tmp_cond2 = cond2[begin:end, :]
            if type(tmp_cond) is not np.ndarray:
                tmp_cond = tmp_cond.toarray()
                tmp_cond2 = tmp_cond2.toarray()
            batch_cond[:end - begin] = tmp_cond
            batch_cond2[:end - begin] = tmp_cond2

            feed_dict = {self.ph_data_g: batch_data, self.ph_cond_g: batch_cond, self.ph_cond2_g: batch_cond2,
                         self.is_train: False}

            batch_pred_mean, batch_pred_var, batch_pred_cond, batch_pred_cond2 = sess.run([self.op_vae_mean, self.op_vae_var,
                                                                                           self.op_cond_latent, self.op_cond2_latent], feed_dict)

            z_mean_pred[begin:end], z_logvar_pred[begin:end], z_cond_pred[begin:end], z_cond2_pred[begin:end] = \
                batch_pred_mean[:end - begin], batch_pred_var[:end - begin], batch_pred_cond[:end - begin], batch_pred_cond2[:end - begin]

        z_mean_pred = np.array(z_mean_pred)
        z_logvar_pred = np.array(z_logvar_pred)
        z_cond_pred = np.array(z_cond_pred)
        z_cond2_pred = np.array(z_cond2_pred)

        return z_mean_pred, z_logvar_pred, z_cond_pred, z_cond2_pred

    def encode_only_condition(self, cond=None, cond2=None):
        """
        Encodes the two conditions to their respective embeddings,
        used at test time sampling and generation experiments.
        args:
            cond: the first condition vector, in CAPE it's body pose
            cond2: the second condition vector, in CAPE it's one-hot clothing type vector
            labels: ground truth, in autoencoding it's the data itself
            sess: tensorflow session to run evaluation
        returns:
            embeddings of the two conditions
        """
        size = cond.shape[0]
        z_cond_pred = [0] * size
        z_cond2_pred = [0] * size

        sess = self._get_session(sess=None) # load the trained model
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_cond = np.zeros((self.batch_size, cond.shape[1]))
            batch_cond2 = np.zeros((self.batch_size, cond2.shape[1]))
            tmp_cond = cond[begin:end, :]
            tmp_cond2 = cond2[begin:end, :]
            if type(tmp_cond) is not np.ndarray:
                tmp_cond = tmp_cond.toarray()
                tmp_cond2 = tmp_cond2.toarray()
            batch_cond[:end - begin] = tmp_cond
            batch_cond2[:end - begin] = tmp_cond2

            feed_dict = {self.ph_cond_g: batch_cond, self.ph_cond2_g: batch_cond2}
            batch_pred_cond, batch_pred_cond2 = sess.run([self.op_cond_latent, self.op_cond2_latent], feed_dict)
            z_cond_pred[begin:end], z_cond2_pred[begin:end] = batch_pred_cond[:end - begin], batch_pred_cond2[:end - begin]

        z_cond_pred = np.array(z_cond_pred)
        z_cond2_pred = np.array(z_cond2_pred)

        return z_cond_pred, z_cond2_pred

    def predict(self, data, cond=None, cond2=None, labels=None, sess=None, phase='train'):
        """
        Makes model prediction and evaluate the auto-encoding precision.
        args:
            data: input to the network, i.e. set of vertices
            cond: the first condition vector, in CAPE it's body pose
            cond2: the second condition vector, in CAPE it's one-hot clothing type vector
            labels: ground truth, in autoencoding it's the data itself
            sess: tensorflow session to run evaluation
            phase: 'train' or 'test'. This is only used to show a tqdm progress bar at
                    real test time (validation during training not included)
        returns:
            predicted mesh vertices, and loss values from evaluation
        """
        loss_recon = []
        loss_latent = []
        loss_edge = []

        size = data.shape[0]
        predictions = [0] * size
        # when the last batch cannot be sharply filled with datas, the rest are filled with zeros
        num_zero_phs = self.batch_size * (size/self.batch_size + 1) - size
        sess = self._get_session(sess)
        disable_tqdm = (phase!='test')
        for begin in tqdm.tqdm(range(0, size, self.batch_size), disable=disable_tqdm):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()
            batch_data[:end - begin] = tmp_data


            batch_cond = np.zeros((self.batch_size, cond.shape[1]))
            batch_cond2 = np.zeros((self.batch_size, cond2.shape[1]))
            tmp_cond = cond[begin:end, :]
            tmp_cond2 = cond2[begin:end, :]

            if type(tmp_cond) is not np.ndarray:
                tmp_cond = tmp_cond.toarray()
                tmp_cond2 = tmp_cond2.toarray()
            batch_cond[:end - begin] = tmp_cond
            batch_cond2[:end - begin] = tmp_cond2

            feed_dict = {self.ph_data_g: batch_data, self.ph_cond_g: batch_cond, self.ph_cond2_g: batch_cond2,
                         self.is_train: False}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, labels.shape[1], labels.shape[2]))
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_gt] = batch_labels
                batch_pred, batch_loss_recon, batch_loss_latent, batch_loss_edge= sess.run(
                    [self.op_prediction, self.recon_loss, self.latent_loss, self.edge_loss],
                    feed_dict)
                loss_recon.append(batch_loss_recon)
                loss_latent.append(batch_loss_latent)
                loss_edge.append(batch_loss_edge)
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            predictions[begin:end] = batch_pred[:end - begin]
        predictions = np.array(predictions)

        def calc_mean(loss_collection):
            last_batch_mean = loss_collection[-1]
            total_loss = np.sum(np.array(loss_collection)[:-1]) * self.batch_size + last_batch_mean * (self.batch_size - num_zero_phs)
            return total_loss / size

        loss_recon, loss_latent, loss_edge = list(map(calc_mean, (loss_recon, loss_latent, loss_edge)))
        if labels is not None:
            return predictions, loss_recon, loss_latent, loss_edge
        else:
            return predictions

    def evaluate(self, data, cond=None, cond2=None, labels=None, sess=None):
        """
        A wrapper on the predict() function.
        args:
            data: input to the network, i.e. set of vertices
            cond: the first condition vector, in CAPE it's body pose
            cond2: the second condition vector, in CAPE it's one-hot clothing type vector
            labels: ground truth, in autoencoding it's the data itself
            sess: tensorflow session to run evaluation
        returns:
            loss values from evaluation
        """
        t_start = time.time()
        # print(labels.shape, data.shape)
        predictions, loss_recon, loss_latent, loss_edge = self.predict(data, cond, cond2, labels, sess)

        if type(self.lambda_edge) != float:
            lambda_edge_val = sess.run(self.lambda_edge)
        else:
            lambda_edge_val = self.lambda_edge

        if type(self.lambda_l1) != float:
            lambda_l1_val = sess.run(self.lambda_l1)
        else:
            lambda_l1_val = self.lambda_l1

        string = 'recon loss: {:.2e}, latent loss: {:.2e}, edge_loss: {:.2e}' \
                 '(weighted)'.format(loss_recon*lambda_l1_val, loss_latent*self.lambda_latent,
                                     loss_edge*lambda_edge_val)
        if sess is None:
            string += '\ntime: {:.0f}s'.format(time.time() - t_start)
        return string, loss_recon, loss_latent, loss_edge


    def decode(self, data, cond=None, cond2=None):
        '''
        Decode from latent vector (for test time use).
        args:
            data: latent vector
            cond: first condition, in CAPE it is body pose
            cond2: second condition, in CAPE it is one-hot encoding of clothing type
        returns:
            x_rec: the generated mesh vertices
        '''
        size = data.shape[0]
        x_rec = [0] * size
        sess = self._get_session(sess=None)

        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data

            # if cond is not None and cond2 is not None:
            batch_cond = np.zeros((self.batch_size, cond.shape[1]))
            batch_cond2 = np.zeros((self.batch_size, cond2.shape[1]))
            if cond.shape[0] == 1: # in demos, sometimes one cond but sample multiple z
                begin_cond, end_cond = 0, self.batch_size
            else:
                begin_cond, end_cond = begin, end

            tmp_cond = cond[begin_cond:end_cond, :]
            tmp_cond2 = cond2[begin_cond:end_cond, :]
            batch_cond[:end - begin] = tmp_cond
            batch_cond2[:end - begin] = tmp_cond2

            feed_dict = {self.ph_z_total: batch_data, self.ph_y_latent: batch_cond,
                         self.ph_y2_latent: batch_cond2, self.is_train: False}


            batch_pred = sess.run(self.op_decoder, feed_dict)

            x_rec[begin:end] = batch_pred[:end - begin]

        x_rec = np.array(x_rec)
        return x_rec