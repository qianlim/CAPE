import numpy as np
import os
import copy
from demos import demo_simple
from lib import models
from lib.load_data import load_graph_mtx
from config_parser import parse_config

args, args_dict = parse_config()
np.random.seed(args_dict['seed'])
project_dir = os.path.dirname(os.path.realpath(__file__))

# load pre-computed graph laplacian and pooling matrices for discriminator
L, D, U, p, L_ds2, D_ds2, U_ds2 = load_graph_mtx(project_dir, load_for_demo=True)

# pass params and build model
params = copy.deepcopy(args_dict)
params['restart'] = bool(args.restart)
params['use_res_block'], params['use_res_block_dec'] = bool(args.use_res_block), bool(args.use_res_block_dec)
params['nn_input_channel'] = 3
params['K'] = [2] * args.num_conv_layers
params['Kd'] = args.Kd  # Chebyshev Polynomial orders.
params['p'] = p
params['n_layer_cond'] = args.n_layer_cond
params['cond_encoder'] = bool(args.cond_encoder)
params['reduce_dim'] = args.reduce_dim
params['affine'] = bool(args.affine)
params['optimizer'] = args.optimizer
params['lr_warmup'] = bool(args.lr_warmup)
params['optim_condnet'] = bool(args.optim_condnet)
params['decay_steps'] = 1
params['cond_dim'] = 126
params['cond2_dim'] = 4
nf = args.nf
params['F'] = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf, 8*nf, 8*nf]

non_model_params = ['demo_n_sample', 'mode', 'dataset', 'num_conv_layers', 'ds_factor',
                    'nf', 'config', 'pose_type', 'decay_every', 'gender',
                    'save_obj', 'vis_demo', 'smpl_model_folder']

for key in non_model_params:
    params.pop(key)

print("Building model graph...")
model = models.CAPE(L=L, D=D, U=U, L_d=L_ds2, D_d=D_ds2, **params)

model.build_graph(model.input_num_verts, model.nn_input_channel, phase='demo')
demos = demo_simple(model, args.name, args.seed)
demos.sample_vary_clotype()
