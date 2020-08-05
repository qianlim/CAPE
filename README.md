## CAPE: Clothed Auto-Person Encoding (CVPR 2020)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/1907.13615) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DCNo2OyyTNi1xDG-7j32FZQ9sBA6i9Ys)

Tensorflow (1.13) implementation of the CAPE model, a Mesh-CVAE with a mesh patch discriminator, for **dressing SMPL bodies** with pose-dependent clothing, introduced in the CVPR 2020 paper:

**Learning to Dress 3D People in Generative Clothing** <br>
Qianli Ma, Jinlong Yang, Anurag Ranjan, Sergi Pujades, Gerard Pons-Moll, Siyu Tang, and Michael. J. Black <br>
[Full paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ma_Learning_to_Dress_3D_People_in_Generative_Clothing_CVPR_2020_paper.pdf) | [Paper in 1 min](https://cape.is.tue.mpg.de/uploads/ckeditor/attachments/273/382-1min.mp4) | [New dataset](https://cape.is.tue.mpg.de/dataset) | [Project website](https://cape.is.tue.mpg.de/)


![](data/cape.gif)

## Google Colab demo

In case you do not have a suitable GPU environment to run the CAPE code, we offer a demo on Google Colab. It generates and visualizes the 3D geometry of the clothed SMPL meshes:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DCNo2OyyTNi1xDG-7j32FZQ9sBA6i9Ys)

For the full demo and training, follow the next steps.

## Installation

We recommend creating a new virtual environment for a clean installation of the dependencies. All following commands are assumed to be executed within this virtual environment. The code has been tested on Ubuntu 18.04, python 3.6 and CUDA 10.0.

```bash
python3 -m venv $HOME/.virtualenvs/cape
source $HOME/.virtualenvs/cape/bin/activate
pip install -U pip setuptools
```

- Install [PSBody Mesh package](https://github.com/MPI-IS/mesh). Currently we recommend installing version 0.3.
- Install [smplx python package](https://github.com/vchoutas/smplx). 
- Download the [SMPL body model](https://smpl.is.tue.mpg.de/), and place the `.pkl` files for both genders and put them in `/body_models/smpl/`. Follow the [instructions](https://github.com/vchoutas/smplx/blob/master/tools/README.md) to remove the Chumpy objects from both model pkls.  
- Then simply run `pip install -r requirements.txt` (do this at last to ensure `numpy==1.16.2`).

## Quick demo 

- Download the SMPL body model as described above.
- `cd CAPE && mkdir checkpoints`
- Download our [pre-trained demo model](https://drive.google.com/drive/folders/11n7iuW0DBZH2ZZa67QEb-mdg29gaUyHE?usp=sharing) and put the downloaded folder under the `checkpoints` folder. Then run:

```bash
python main.py --config configs/CAPE-affineconv_nz64_pose32_clotype32_male.yaml --mode demo
```

It will generate a few clothed body meshes in the `results/` folder and show on-screen visualization.

## Process data, training and evaluation
### Prepare training data
Here we assume that the [CAPE dataset](https://cape.is.tue.mpg.de/dataset) is downloaded. The "raw" data are stored as an `.npz` file per frame. We are going to pack these data into dataset(s) that can be used to train the network. For example, the following command
```bash
python lib/prep_data.py <path_to_downloaded_CAPE_dataset> --ds_name dataset_male_4clotypes --phase both
```
will create a dataset named `dataset_male_4clotypes`, both the training and test splits, under `data/datasets/dataset_male_4clotypes`. This dataset contains 31036 training and 5128 test examples. Similarly, setting `--ds_name dataset_female_4clotypes` will create the female dataset that contains 21090 training and 5441 test examples.

To customize the packed dataset, simply edit the `dataset_config_dicts` defined in `data/dataset_configs.py` for your subject / clothing type / sequences of interest. 

### Training

Once the dataset is packed, we are ready for training! Give a name to the experiment (will be used to save / load checkpoints and for Tensorboard etc.), specify the genders (here assume we train a male model), and run:

```bash
python main.py --config configs/config.yaml --name <exp_name> --mode train
```

The training will start. You can watch the training process by running Tensorboard in `summaries/<exp_name>`. At the end of training it will automatically evaluate on the test split and run the generation demos.

To customize the architecture and training, check the arguments defined in `config_parser.py`, and set them either in `configs/config.yaml` or directly in command line by`--[some argument] <value>`.

### Evaluation

Change the `--mode` flag to `demo` to run the auto-encoding evaluation. It will also run the generation demos. 

```bash
python main.py --config configs/config.yaml --name <exp_name> --gender <gender> --mode demo
```

### Performance

The public release of the [CAPE dataset]((https://cape.is.tue.mpg.de/dataset)) slightly differs from what we used in the paper due to the removal of faulty / corrupted frames. Therefore we retrained our model on the `dataset_male_4clotypes` and `dataset_female_4clotypes` datasets packed as shown above, and report the performance in terms of per-vertex auto-encoding Eucledian errors (in mm).

**On male dataset**:

| Method       | PCA           | CoMA-4\*        | CoMA-1\*      | CAPE          | CAPE-affine\_conv\** |
| :------------ | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| error mean   | 7.13 ± 5.27 | 7.32 ± 5.57 | 6.50 ± 5.35  | 6.15 ± 5.30 |  6.03 ± 5.18  |
| medians      | 5.79       |   5.85       | 5.02        | 4.64        |  4.54      |

**On female dataset**:

| Method     |     PCA     |   CoMA-4    |   CoMA-1    |    CAPE     | CAPE-affine\_conv |
| :--------- | :---------: | :---------: | :---------: | :---------: | :---------------: |
| error mean | 3.87 ± 3.02 | 4.38 ± 3.33 | 3.86 ± 3.09 | 3.61 ± 3.01 |    3.58 ± 2.94    |
| medians    |    3.10     |    3.55     |    3.07     |    2.82     |       2.82        |

\* CoMA-X stands for the model by [Ranjan et al.](https://arxiv.org/abs/1807.10267) with a spatial downsampling rate X at each downsample layer. <br>
\*\* CAPE-affine\_conv uses an improved mesh-residual block based on the idea of [this CVPR2020 paper](https://arxiv.org/abs/2004.02658), instead of our original mesh residual block. It achieves improved results and is faster in training. To use this layer, use the flag `--affine 1` in training.

### Miscellaneous notes on training

The latent space dimension used in the paper and numbers above (set by flag `--nz`) is 18 to balance the size of the model and performance, but at a price of losing some clothing details. Increasing the latent dimension brings significant better wrinkles and edges. We also provide the model checkpoints trained with `--nz 64 --nz_cond 32 --nz_cond2 32 --affine 1`, see below.

### Pretrained models

Our pretrained models on the above two datasets can be downloaded [here](https://drive.google.com/drive/folders/12tjK-nSvDAYqezoePTJwG-ZIpcHjRuxM?usp=sharing).  Their corresponding configuration `yaml` files are already in `configs/` folder, with the same name as the name of each checkpoint folder.

To run evaluation / demo from the pretrained models, put the downloaded folder(s) under the `checkpoints` folder, and run the evaluation command, e. g.:

```bash
python main.py --config configs/CAPE-affineconv_nz18_pose24_clotype8_male.yaml --mode demo
```

## CAPE dataset

Together with the model, we introduce the new CAPE dataset, a large scale 3D mesh dataset of clothed humans in motion. It contains 150K dynamnic clothed human mesh registrations from real scan data, with consistent topology. We also provide precise body shape under clothing, SMPL pose parameters and clothing displacements for all data frames, as well as handy code to process the data. Check it out at our [project website](https://cape.is.tue.mpg.de/)!

## News

**05/08/2020** A Google Colab demo is added!

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
