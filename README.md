## CAPE: Clothed Auto-Person Encoding (CVPR 2020)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/1907.13615) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)]() [![tensorflow](https://aleen42.github.io/badges/src/tensorflow.svg)]()

Tensorflow (1.13) implementation of the CAPE model, a Mesh-CVAE with a mesh patch discriminator, for **dressing SMPL bodies** with pose-dependent clothing, introduced in the CVPR 2020 paper:

**Learning to Dress 3D People in Generative Clothing**
Qianli Ma, Jinlong Yang, Anurag Ranjan, Sergi Pujades, Gerard Pons-Moll, Siyu Tang, and Michael. J. Black
[Full paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ma_Learning_to_Dress_3D_People_in_Generative_Clothing_CVPR_2020_paper.pdf) | [Paper in 1 min](https://cape.is.tue.mpg.de/uploads/ckeditor/attachments/273/382-1min.mp4) | [New dataset](https://cape.is.tue.mpg.de/dataset) | [Project Website](https://cape.is.tue.mpg.de/)


![](data/cape.gif)

## Installation

We recommend creating a new virtual environment for a clean installation of the dependencies. All following commands are assumed to be executed within this virtual environment. The code has been tested on Ubuntu 18.04, python 3.6 and CUDA 10.0.

```bash
python3 -m venv $HOME/.virtualenvs/cape
source $HOME/.virtualenvs/cape/bin/activate
pip install -U pip setuptools
```

- Install [PSBody Mesh package](https://github.com/MPI-IS/mesh). Currently we recommend installing version 0.3.
- Install [smplx python package](https://github.com/vchoutas/smplx). Follow the installation instructions there, download and setup the SMPL body model.
- Then simply run `pip install -r requirements.txt` (do this at last to ensure `numpy==1.16.1`).

## Run demo code

Download the [checkpoint](https://drive.google.com/drive/folders/1H-kbLnIv9_k2DADrldXWplGauSnoqIhF?usp=sharing) and put this checkpoint folder under the `checkpoints` folder. Then run:

```bash
python main.py --config configs/config.yaml --mode demo --vis_demo 1 --smpl_model_folder <path to SMPL model folder>
```

It will generate a few clothed body meshes in the `results/` folder and show on-screen visualization.

## Process data and training
### Prepare training data
Here we assume that the [CAPE dataset](https://cape.is.tue.mpg.de/dataset) is downloaded. The "raw" data are stored as an `.npz` file per frame. We are going to pack these data into dataset(s) that can be used to train the network. For example, the following command
```python
python lib/prep_data.py <path_to_downloaded_CAPE_dataset> --ds_name dataset_male_4clotypes --phase both
```
will create a dataset named `dataset_male_4clotypes`, both the training and test splits, under `data/datasets/dataset_male_4clotypes`. This dataset contains 28600 training and 5128 test examples. 

To customize the packed dataset, simply edit the dataset configuration dictionary `dataset_male_4clotypes` in `lib/prep_data.py` for your subject / clothing type / sequences of interest.

### Training

Once the dataset is packed, we are ready for training! Give a name to the experiment (will be used to save / load checkpoints and for Tensorboard etc.), and run:

```python
python main.py --config configs/config.yaml --name <exp_name> --smpl_model_folder <path to SMPL model folder> --mode train 
```

The training will start. You can watch the training process by running Tensorboard in `summaries/<exp_name>`. At the end of training it will automatically evaluate on the test split and run the generation demos.

To customize the architecture and training, check the arguments defined in `config_parser.py`, and set them either in `configs/config.yaml` or directly in command line by`--[some argument] <value>`.

### Evaluation

Change the `--mode` flag to `demo` to run the auto-encoding evaluation. It will also run the generation demos. 

```python
python main.py --config configs/config.yaml --name <exp_name> --smpl_model_folder <path to SMPL model folder> --mode demo 
```

### Performance

The public release of the [CAPE dataset]((https://cape.is.tue.mpg.de/dataset)) slightly differs from what we used in the paper due to the removal of faulty / corrupted frames. Therefore we retrained our model on the `dataset_male_4clotypes` dataset packed as shown above, and report the performance in terms of per-vertex auto-encoding eucledian errors (in mm).

| Method       | PCA           | CoMA-4\*        | CoMA-1\*      | CAPE          | CAPE-affine\_conv\** |
| :------------ | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| error mean   | 7.15 ± 5.27 | 7.42 ± 5.60  | 6.90 ± 5.63   | 6.35 ± 5.46   |  6.19 ± 5.28  |
| medians      | 5.83        |   5.95       | 5.37          | 4.79          |  4.70       |

\* CoMA-X stands for the model by [Ranjan et al.](https://arxiv.org/abs/1807.10267) with a spatial downsampling rate X at each downsample layer.

\*\* CAPE-affine\_conv uses an improved mesh-residual block based on the idea of [this CVPR2020 paper](https://arxiv.org/abs/2004.02658), instead of our original mesh residual block. It achieves improved results and is faster in training. To use this layer, use the flag `--affine 1` in training.


## News
**28/07/2020** Data packing and training scripts added! Also added a few new features. Check the [changelog](./CHANGELOG.md) for more details.

**26/07/2020** Updated the link to the pretrained checkpoint (previous one was faulty and generates weird shapes); minor bug fixes in the group norm param loading.

## License

Software Copyright License for non-commercial scientific research purposes. Please read carefully the [terms and conditions](./LICENSE) and any accompanying documentation before you download and/or use the CAPE data and software, (the "Dataset & Software"), including 3D meshes, pose parameters, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

The SMPL body related files  `data/{template_mesh.obj, edges_smpl.npy}` are  subject to the license of the [SMPL model](https://smpl.is.tue.mpg.de/modellicense). The [PSBody mesh package](https://github.com/MPI-IS/mesh) and [smplx python package](https://github.com/vchoutas/smplx) are subject to their own licenses.

## Citations
### Citing this work
If you find our code / paper / data useful to your research, please consider citing:

```bibtex
@inproceedings{ma2020cape,
    title = {Learning to Dress 3D People in Generative Clothing},
    author = {Ma, Qianli and Yang, Jinlong and Ranjan, Anurag and Pujades, Sergi and Pons-Moll, Gerard and Tang, Siyu and Black, Michael J.},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    month = jun,
    year = {2020},
    month_numeric = {6}
}
```

### Related projects

[CoMA (ECCV 2018)](https://coma.is.tue.mpg.de/): Our (non-conditional) convolutional mesh autoencoder for modeling extreme facial expressions. The codes of the CAPE repository are based on the [repository of CoMA](https://github.com/anuragranj/coma). If you find the code of this repository useful, please consider also citing CoMA.

[ClothCap (SIGGRAPH 2017)](http://clothcap.is.tue.mpg.de/): Our method of capturing and registering clothed humans from 4D scans. The *CAPE dataset* released with our paper incorporates the scans and registrations from ClothCap. Check out our [project website](https://cape.is.tue.mpg.de/dataset) for the data!
