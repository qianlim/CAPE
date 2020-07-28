def parse_config(argv=None):
    import configargparse
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'CAPE model: mesh CVAE + discriminator'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='CAPE')
    parser.add_argument('--config', is_config_file=True, default='configs/default_config.yaml', help='config file path')
    parser.add_argument('--name', help='name of the run, will be used to save/load checkpoints')
    # architecture related
    parser.add_argument('--num_conv_layers', type=int, default=8, help='number of convolution layers')
    parser.add_argument('--ds_factor', type=int, default=2, help='downsample factor')
    parser.add_argument('--K', type=int, default=2, help='order of chebyshev polynomial for the VAE')
    parser.add_argument('--Kd', type=int, default=3, help='order of chebyshev polynomial for discriminator')
    parser.add_argument('--nf', type=int, default=64, help='number of conv filters of first encoder layer')
    parser.add_argument('--nz', type=int, default=18, help='Size of latent variable in latent space')
    parser.add_argument('--nz_cond', type=int, default=24, help='size of embedding of the first condition (pose)')
    parser.add_argument('--nz_cond2', type=int, default=8, help='size of embedding of the second condition (clotype)')
    parser.add_argument('--n_layer_cond', type=int, default=1, help='number of layers for the clothing type condition network')
    parser.add_argument('--activation', default='b1leakyrelu', help='b1relu, b1leakyrelu or b1tanh')
    parser.add_argument('--use_res_block', type=int, default=0, help='whether to use residual block in encoder')
    parser.add_argument('--use_res_block_dec', type=int, default=1, help='whether to use residual block in decoder')
    parser.add_argument('--cond_encoder', type=int, default=0, help='1 for condition the encoder too, 0 for not')
    parser.add_argument('--reduce_dim', type=int, default=64,
                        help='reduce the channels in encoder final conv layer to this number (to shrink the final fc layer size)')
    parser.add_argument('--affine', type=int, default=0, help='whether or not use affine residual block in decoder')
    parser.add_argument('--pose_type', default='rot', choices=['pose','rot'],
                        help='SMPL pose params or their rotational matrices')
    parser.add_argument('--optim_condnet', type=int, default=1)
    # training related
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--num_epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=8e-3, help='Learning Rate')
    parser.add_argument('--lr_scaler', type=float, default=1e-1, help='for training GAN model, lr is for G, multiply this scaler for D')
    parser.add_argument('--decay_every', type=int, default=1, help='decay lr after x epochs')
    parser.add_argument('--lr_warmup', type=int, default=0, help='Whether to use lr warmup')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--restart', type=int, default=1, help='restart training or resume from checkpoint')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='adam or sgd (with momentum)')
    parser.add_argument('--loss', default='l1', help='l1 or l2')
    parser.add_argument('--loss_mask', default='', type=str,
                        help='binary or None; if binary then apply a mask favoring loss on clothing-related vertices')
    parser.add_argument('--dataset', type=str, default='dataset_male_4clotypes',
                        help='name of the dataset, we will update data packing and loading code soon.')
    # loss related
    parser.add_argument('--regularization', type=float, default=2e-3, help='weight for regularization term')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='coefficient for l1 loss')
    parser.add_argument('--lambda_edge', type=float, default=1.0, help='coefficient for edge loss, while l1 loss has coeff 1')
    parser.add_argument('--lambda_latent', type=float, default=8e-4, help='coefficient for latent loss, while l1 loss has coeff 1')
    parser.add_argument('--lambda_gan', type=float, default=0.1, help='coefficient for gan loss')
    # demo related
    parser.add_argument('--mode', type=str, help='train or test or demo', choices=['train', 'test', 'demo'], default='train')
    parser.add_argument('--gender', type=str, help='used to load smplx model of desired gender at test/demo',
                        choices=['female', 'male', 'neutral'], default='neutral')
    parser.add_argument('--smpl_model_folder', type=str, help='folder where the smpl model .pkl files are stored')
    parser.add_argument('--demo_n_sample', type=int, default=5, help='generate n samples for demo')
    parser.add_argument('--save_obj', type=int, default=1, choices=[0, 1],
                        help='1 for saving meshes generated at demos, 0 for not save')
    parser.add_argument('--vis_demo', type=int, default=0, choices=[0, 1],
                        help='1 for on-screen visualization of generated mesh in test/demos, 0 for not vis')

    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    return args, args_dict