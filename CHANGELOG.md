## Changelog

### 28/07/2020

- Add scripts for training, test (auto-encoding) and data preprocessing.
- Add feature: a new type of residual block for the decoder (`res_block_affine`), based on the idea of [this CVPR2020 paper](https://arxiv.org/abs/2004.02658). It achieves improved performance than our original mesh res-block. To use this layer use the flag `--affine 1`; otherwise it uses our graph res-block by default.

- Add feature: two new model configurations with 4 layers or 6 layers respectively. Specify this with the flag e.g. `--num_conv_layers 6`
- Add feature: now the spatial downsampling factor can be set by `--ds_factor X` (previously hard-set as 2).

### 27/07/2020

- Fix the download link to a faulty checkpoint that yields strange generated garment shape.
- Fix a bug that causes the group norm stats not properly loaded at inference.

