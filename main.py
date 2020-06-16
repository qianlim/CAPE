import numpy as np
import yaml
import os
import copy
from demos import demo
from lib import models, mesh_sampling
from load_data import BodyData, load_graph_mtx
from config_parser import parse_config
from psbody.mesh import Mesh

args, args_dict = parse_config()
np.random.seed(args_dict['seed'])
project_dir = os.path.dirname(os.path.realpath(__file__))
reference_mesh_file = os.path.join(project_dir, 'data/template_mesh.obj')
reference_mesh = Mesh(filename=reference_mesh_file)

'''
Data preparation code will be available soon
'''
datadir_root = os.path.join(project_dir, 'data')
# datadir_root = '/is/cluster/shared/datasets/ps/clothdata/alignments_clothed_human/packed_data/'
data_dir = os.path.join(datadir_root, args.dataset)
# # load data
# print("Loading data from {} ..".format(data_dir))
# train_condition_file1 = data_dir + '/train/train_{}.npy'.format(args.pose_type)
# train_condition_file2 = data_dir + '/train/train_{}.npy'.format('clo_label')
# test_condition_file1 = data_dir + '/test/test_{}.npy'.format(args.pose_type)
# test_condition_file2 = data_dir + '/test/test_{}.npy'.format('clo_label')
# bodydata = BodyData(nVal=200,
#                     train_file_A=data_dir + '/train/train_disp.npy',
#                     train_file_C=train_condition_file1,
#                     train_file_C2=train_condition_file2,
#                     test_file_A=data_dir + '/test/test_disp.npy',
#                     test_file_C=test_condition_file1,
#                     test_file_C2=test_condition_file2,
#                     reference_mesh_file=reference_mesh_file)

ds_factors = [1, 2, 1, 2, 1, 2, 1, 1] # mesh downsampling factor of each layer
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
params['F'] = [nf, nf, 2 * nf, 2 * nf, 4 * nf, 4 * nf, 8 * nf, 8 * nf]
params['K'] = [2] * 8
params['Kd'] = args.Kd  # Polynomial orders.
params['p'] = p
params['decay_steps'] = 800 #args.decay_every * len(bodydata.vertices_train_A) / params['batch_size']
params['cond_dim'] = 126
params['cond2_dim'] = 4
params['n_layer_cond'] = args.n_layer_cond
params['cond_encoder'] = bool(args.cond_encoder)
params['reduce_dim'] = args.reduce_dim
params['optimizer'] = args.optimizer
params['lr_warmup'] = bool(args.lr_warmup)

non_model_params = ['demo_n_sample', 'mode', 'dataset',
                    'seed', 'nf', 'config', 'pose_type', 'decay_every',
                    'save_obj', 'vis_demo', 'smpl_model_folder']

for key in non_model_params:
    params.pop(key)

print("Building model graph...")
model = models.CAPE(L=L, D=D, U=U, L_d=L_ds2, D_d=D_ds2, **params)

with open('configs/{}_config.yaml'.format(params['name']), 'w') as fn:
    yaml.dump(params, fn)

# start train or test/demo
if args.mode == 'train':
    model.build_graph(model.input_num_verts, model.nn_input_channel, phase='train')
    loss, t_step = model.fit(bodydata)
else:
    model.build_graph(model.input_num_verts, model.nn_input_channel, phase='demo')
    demos = demo(model, args.name, args.dataset, data_dir, datadir_root,
                 n_sample=args.demo_n_sample, save_obj=bool(args.save_obj),
                 vis=bool(args.vis_demo), smpl_model_folder=args.smpl_model_folder)
    demos.run()

