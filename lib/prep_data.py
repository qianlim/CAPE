# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
from os.path import join, exists
from glob import glob
import numpy as np
import cv2
import tqdm

import sys
sys.path.append("..")

from data.dataset_configs import dataset_config_dicts

script_dir = os.path.dirname(os.path.realpath(__file__))
packed_data_root = join(script_dir, '../data', 'datasets') # extracted sub-dataset under


def pack_unposed_datadict(collections, cape_ds_dir, subj, seq, cloth, cut_first=2, sample_rate=1):
    '''
    Add data from one sequence (subj, cloth, seq), and append them to the pack collection
    The pack collection includes the clothing disps, pose params, pose rotational matrices,
    and the clothing type (as one-hot vectors)

    args:
        collections: lists of the data (as numpy arrays) to be packed
        subj: subject ID, a string, 5-digits, 0-padded, e.g. "00032"
        seq: sequence name, a string, e.g. "move_arms"
        cloth: name of the clothing type from the CAPE dataset, a string, e.g. "longlong"
        cut_first: number of the frames to be skipped for each sequence (typically of redundant "A"-pose), int
        sample_rate: sample every X frames from the sequence. CAPE data are captured at 60FPS, use different sample rates
                     to get lower resulting frame rates.
    '''
    vdisps, poses, local_rots, clo_labels, dataset_info, broken_frames = collections

    clo_type = np.array(['shortlong', 'shortshort', 'longshort', 'longlong'])

    file_counter = 0
    fr_key = cloth + '_' + seq

    data_dir = join(cape_ds_dir, 'sequences', subj, cloth+'_'+seq)
    minimal_fn = join(cape_ds_dir, 'minimal_body_shape', subj, '{}_minimal.npy'.format(subj))
    minimal_cano = np.load(minimal_fn) # minimal clothed body shape, canonical pose

    flist = sorted(glob(join(data_dir, '*.npz')))
    if len(flist) == 0:
        print('{} no files here, skipping ..\n'.format(subj+'_'+fr_key))
        return
    flist = flist[cut_first: -cut_first : sample_rate] # take every k frame

    for fname in tqdm.tqdm(flist):
        try:
            data_dict = np.load(fname)
        except: # skip and report the broken file if any
            print('{} broken, skipping..'.format(fname))
            broken_frames.append(fname)
            continue

        pose = data_dict['pose']
        vdisp = data_dict['v_cano'] - minimal_cano
        # rotation matrices from pose params
        pose_reshaped = pose.reshape(-1, 3)
        local_rot = np.array([cv2.Rodrigues(pose_reshaped[i, :])[0] for i in range(pose_reshaped.shape[0])]).ravel()
        clo_label = (clo_type == cloth).astype(int)  # one-hot encoding of cloth type
        vdisps.append(vdisp)
        poses.append(pose)
        local_rots.append(local_rot)
        clo_labels.append(clo_label)

        file_counter += 1

    seq_info = '{} - {}: {} {} {}, {} frames\n'.format(len(vdisps)-file_counter+1, len(vdisps), subj, seq, cloth, file_counter)
    dataset_info.append(seq_info)
    return 1


def save_npy(data, savename, ds_name, phase):
    print('Saving {}'.format(savename))
    os.makedirs(join(packed_data_root, ds_name, phase), exist_ok=True)
    np.save(join(packed_data_root, ds_name, phase, '{}.npy'.format(savename)), np.array(data))


def save_all(vdisps, poses, local_rots, clo_labels, ds_name, phase='train'):
    save_npy(vdisps, '{}_disp'.format(phase), ds_name, phase)
    save_npy(poses, '{}_pose'.format(phase), ds_name, phase)
    save_npy(local_rots, '{}_rot'.format(phase), ds_name, phase)
    save_npy(clo_labels, '{}_clo_label'.format(phase), ds_name, phase)


def get_clolabel_stats(clo_labels):
    clo_labels_stats = np.array(clo_labels)
    num_shortlong = len(clo_labels_stats[(clo_labels_stats[:, 0] == 1).reshape(-1)])
    num_shortshort = len(clo_labels_stats[(clo_labels_stats[:, 1] == 1).reshape(-1)])
    num_longshort = len(clo_labels_stats[(clo_labels_stats[:, 2] == 1).reshape(-1)])
    num_longlong = len(clo_labels_stats[(clo_labels_stats[:, 3] == 1).reshape(-1)])
    return [num_shortlong, num_shortshort, num_longshort, num_longlong]


def create_dataset(phase, dataset_config_dict, cape_ds_dir, dataset_name):

    cut_first, sample_rate = dataset_config_dict['cut_first'], dataset_config_dict['sample_rate']

    print('\n===========Creating {}, {} set...'.format(dataset_name, phase.upper()))

    collections = [[], [], [], [], [], []]
    [vdisps, poses, local_rots, clo_labels, dataset_info, broken_frames] = collections
    clotype_counter = {'shortlong': 0, 'shortshort': 0, 'longshort': 0, 'longlong': 0}

    for subj in dataset_config_dict['{}_subjs'.format(phase)]:
        for seq in dataset_config_dict['{}_seqs'.format(phase)]:
            for cloth in dataset_config_dict['{}_cloth'.format(phase)]:
                if seq in dataset_config_dict['exclude_seqs']:
                    continue
                if [subj, cloth] in dataset_config_dict['exclude_cases']:
                    continue
                if not exists(join(cape_ds_dir, 'sequences', subj, cloth+'_'+seq)):
                    continue

                print('Adding {} {} {}...'.format(subj, seq, cloth))

                status = pack_unposed_datadict(collections, cape_ds_dir, subj, seq, cloth, cut_first=cut_first, sample_rate=sample_rate)

                if status is not None:
                    print(len(vdisps), vdisps[0].shape)

                    for clotype in clotype_counter.keys():
                        clotype_counter[clotype] += int(clotype in cloth)

    # count the stats
    try:
        clolabel_stats = get_clolabel_stats(clo_labels)
    except IndexError:
        pass
    if len(vdisps)>0:
        save_all(vdisps, poses, local_rots, clo_labels, dataset_name, phase=phase)

        file_mode = 'w+' if phase=='train' else 'a+'
        with open(join(packed_data_root, dataset_name, 'stats.txt'), mode=file_mode) as f:
            f.write('-----------{} SET-----------\n'.format(phase.upper()))
            for item in dataset_info:
                f.write(item)
            f.write('\nTotal sequences: {} \n'.format(clotype_counter))
            f.write('Total: {} shortlong, {} shortshort, {} longshort, {} longlong\n'.format(*clolabel_stats))
            f.write('Total: {} {} examples\n\n'.format(len(vdisps), phase))

        with open(join(packed_data_root, dataset_name, 'broken_frames.txt'), mode=file_mode) as f:
            f.write('-----------{} SET-----------\n'.format(phase.upper()))
            f.write('Broken frames that are not packed:\n')
            for item in broken_frames:
                f.write('{}\n'.format(item))
    else:
        print('Specified sequences do not exist. Please modify the dataset configuration dict.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Packing the downloaded data into dataset for model training')
    parser.add_argument('cape_ds_dir', type=str, help='path to the directory of the downloaded CAPE datasaet',
                        default='/is/cluster/shared/datasets/ps/clothdata/alignments_clothed_human/cape_release_full')
    parser.add_argument('--gender', type=str, choices=['female', 'male'], default='male')
    parser.add_argument('--ds_name', type=str, help='name of the dataset to be packed', default=None)
    parser.add_argument('--phase', type=str, help='train, test or both', choices=['train', 'test', 'both'], default='both')
    parser.add_argument('--overwrite', action='store_true',
                            help='If set, packed dataset under the same name will be overwritten')
    args = parser.parse_args()


    if args.overwrite:
        try:
            import shutil
            shutil.rmtree(os.path.join(packed_data_root, args.ds_name))
        except:
            pass

    if args.phase in ['train', 'both']:
        create_dataset('train', dataset_config_dicts[args.gender], args.cape_ds_dir, args.ds_name)

    if args.phase in ['test', 'both']:
        create_dataset('test', dataset_config_dicts[args.gender], args.cape_ds_dir, args.ds_name)