import numpy as np
import yaml
import os
import copy
from demos import demo
from lib import models, mesh_sampling
from lib.load_data import BodyData, load_graph_mtx
from config_parser import parse_config
from psbody.mesh import Mesh

args, args_dict = parse_config()
np.random.seed(args_dict['seed'])
project_dir = os.path.dirname(os.path.realpath(__file__))
reference_mesh_file = os.path.join(project_dir, 'data/template_mesh.obj')
reference_mesh = Mesh(filename=reference_mesh_file)

datadir_root = os.path.join(project_dir, 'data', 'datasets')
data_dir = os.path.join(datadir_root, args.dataset)
# load data
print("Loading data from {} ..".format(data_dir))
bodydata = BodyData(nVal=100,
                    train_mesh_fn=data_dir + '/train/train_disp.npy',
                    train_cond1_fn=data_dir + '/train/train_{}.npy'.format(args.pose_type),
                    train_cond2_fn=data_dir + '/train/train_{}.npy'.format('clo_label'),
                    test_mesh_fn=data_dir + '/test/test_disp.npy',
                    test_cond1_fn=data_dir + '/test/test_{}.npy'.format(args.pose_type),
                    test_cond2_fn=data_dir + '/test/test_{}.npy'.format('clo_label'),
                    reference_mesh_file=reference_mesh_file)

if args.num_conv_layers==4:
    ds_factors = [1, args.ds_factor, 1, 1]
elif args.num_conv_layers==6:
    ds_factors = [1, args.ds_factor, 1, args.ds_factor, 1, 1]
elif args.num_conv_layers == 8:
    ds_factors = [1, args.ds_factor, 1, args.ds_factor, 1, args.ds_factor, 1, 1]

print("Pre-computing mesh pooling matrices ..")
M,A,D,U, _ = mesh_sampling.generate_transform_matrices(reference_mesh, ds_factors)
p = list(map(lambda x: x.shape[0], A))
A = list(map(lambda x: x.astype('float32'), A))
D = list(map(lambda x: x.astype('float32'), D))
U = list(map(lambda x: x.astype('float32'), U))
L = [mesh_sampling.laplacian(a, normalized=True) for a in A]

# load pre-computed graph laplacian and pooling matrices for discriminator
L_ds2, D_ds2, U_ds2 = load_graph_mtx(project_dir)

# pass params and build model
params = copy.deepcopy(args_dict)
params['restart'] = bool(args.restart)
params['use_res_block'], params['use_res_block_dec'] = bool(args.use_res_block), bool(args.use_res_block_dec)
params['nn_input_channel'] = 3

nf = args.nf
if args.num_conv_layers==4:
    params['F'] = [nf, 2*nf, 2*nf, nf]
elif args.num_conv_layers==6:
    params['F'] = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf]
elif args.num_conv_layers == 8:
    params['F'] = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf, 8*nf, 8*nf]
else:
    raise NotImplementedError

params['K'] = [2] * args.num_conv_layers
params['Kd'] = args.Kd  # Chebyshev Polynomial orders.
params['p'] = p
params['decay_steps'] = args.decay_every * len(bodydata.vertices_train) / params['batch_size']
params['cond_dim'] = bodydata.cond1_train.shape[-1]
params['cond2_dim'] = bodydata.cond2_train.shape[-1]
params['n_layer_cond'] = args.n_layer_cond
params['cond_encoder'] = bool(args.cond_encoder)
params['reduce_dim'] = args.reduce_dim
params['affine'] = bool(args.affine)
params['optimizer'] = args.optimizer
params['lr_warmup'] = bool(args.lr_warmup)
params['optim_condnet'] = bool(args.optim_condnet)

non_model_params = ['demo_n_sample', 'mode', 'dataset', 'num_conv_layers', 'ds_factor',
                    'seed', 'nf', 'config', 'pose_type', 'decay_every', 'gender',
                    'save_obj', 'vis_demo', 'smpl_model_folder']

for key in non_model_params:
    params.pop(key)

print("Building model graph...")
model = models.CAPE(L=L, D=D, U=U, L_d=L_ds2, D_d=D_ds2, **params)

# start train or test/demo
if args.mode == 'train':
    model.build_graph(model.input_num_verts, model.nn_input_channel, phase='train')
    loss, t_step = model.fit(bodydata)

    # full full test pipeline after training
    model.build_graph(model.input_num_verts, model.nn_input_channel, phase='demo')
    demos = demo(bodydata, model, args.name, args.gender, args.dataset, data_dir, datadir_root,
                 n_sample=args.demo_n_sample, save_obj=bool(args.save_obj),
                 vis=bool(args.vis_demo), smpl_model_folder=args.smpl_model_folder)
    demos.run()
else:
    model.build_graph(model.input_num_verts, model.nn_input_channel, phase='demo')
    demos = demo(bodydata, model, args.name, args.gender, args.dataset, data_dir, datadir_root,
                 n_sample=args.demo_n_sample, save_obj=bool(args.save_obj),
                 vis=bool(args.vis_demo), smpl_model_folder=args.smpl_model_folder)
    demos.run()

