#####################################################################
# Most subjects (except 00134 and 03375) are captured with a subset
# of the following motion sequences:
#####################################################################
seqs_group1 = [
    'ballerina_spin', 'ATUsquat', 'basketball', 'bend_back_and_front', 'bend_twist', 'chicken_wings',
    'flying_eagle', 'hips', 'improvise', 'jumping_jacks', 'move_arms', 'move_shoulders', 'pose_model', 'punching',
    'ROM_lower', 'ROM_upper', 'rotate_hips', 'running_on_the_spot', 'shoulders_mill', 'simple', 'soccer',
    'squats', 'twist_tilt', 'tilt_twist_left', 'twist_tilt_left', 'walk_march'
    ]

# sequences reserved for test in the CAPE paper, completely unseen in training
test_seqs_group1 = ['ballerina_spin', 'soccer', 'pose_model', 'bend_twist']
train_seqs_group1 = list(set(seqs_group1) - set(test_seqs_group1))


####################################################################
# Some subjects (00134 and 03375) are captured with these sequences,
# typically 2 trials per sequence. The sequence name will be e.g.
# `athletics_trialX'. See their data folder for examples of such
# naming convention.
####################################################################

# complete set of seq names of 00134 and 03375
seqs_group2 = [
    'athletics', 'ATUsquat', 'babysit', 'badminton', 'basketball', 'ballet1', 'ballet2', 'ballet3', 'ballet4',
    'box', 'catchpick', 'climb', 'club', 'comfort', 'drinkeat', 'dig', 'fashion','frisbee', 'golf','handball',
    'hands_up', 'hockey', 'housework', 'lean', 'music', 'row', 'run', 'shoulders', 'simple', 'ski', 'slack',
    'soccer', 'stepover', 'stretch', 'swim', 'sword', 'tennis', 'twist', 'twist_tilt', 'umbrella', 'volleyball', 'walk'
    ]

# for "seen" seqs, their trial 1 will be seen in training and trial 2 will be used at test
test_seqs_group2_unseen = ['twist', 'soccer']
# both trials of the "unseen" seqs will be excluded from training
test_seqs_group2_seen = ['climb', 'ski']
train_seqs_group2 = ['athletics', 'ATUsquat', 'badminton', 'basketball', 'ballet1', 'box',
                     'frisbee', 'golf', 'row', 'swim', 'twist_tilt', 'volleyball']

train_seqs_group2 = sorted([x+'_trial1' for x in (train_seqs_group2+test_seqs_group2_seen)]) # tria1 of the "seen" seqs are in training
test_seqs_group2_seen = sorted([x+'_trial2' for x in test_seqs_group2_seen]) # trial2 of the "seen" seqs are for test
test_seqs_group2_unseen = sorted([x+'_trial2' for x in test_seqs_group2_unseen]) # trial2 of the "unseen" seqs are for test, their tiral1 are not used
test_seqs_group2 = test_seqs_group2_unseen + test_seqs_group2_seen


####################################################################
# modify the following dictionaries for your customized choice
# of subjects and motions.
####################################################################

dataset_female_4clotypes = {
    'cut_first': 2,
    'sample_rate': 1,

    'train_subjs': ['00159', '00134', '03223', '03331'],
    'train_seqs': train_seqs_group1 + train_seqs_group2,
    'train_cloth': ['shortlong', 'longshort', 'shortshort', 'longlong'],

    'exclude_seqs':['running_on_the_spot', 'jumping_jacks'], # excluded due to too much dynamics
    'exclude_cases': [],

    'test_subjs':  ['00159', '00134', '03223', '03331'],
    'test_seqs': test_seqs_group1 + test_seqs_group2,
    'test_cloth': ['shortlong', 'longshort', 'shortshort', 'longlong'],
}


dataset_male_4clotypes = {
    'cut_first': 2,
    'sample_rate': 1,

    'train_subjs': ['03284', '00215', '00127', '00122', '00032', '02474', '03394'],
    'train_seqs': train_seqs_group1,
    'train_cloth': ['shortlong', 'longshort', 'shortshort', 'longlong'],

    'exclude_seqs':['running_on_the_spot', 'jumping_jacks'], # excluded due to too much dynamics
    'exclude_cases': [],

    'test_subjs':  ['03284', '00215', '00127', '00122', '00032', '02474', '03394'],
    'test_seqs': test_seqs_group1,
    'test_cloth': ['shortlong', 'longshort', 'shortshort', 'longlong'],
}


dataset_config_dicts = {
    'male': dataset_male_4clotypes,
    'female': dataset_female_4clotypes
}