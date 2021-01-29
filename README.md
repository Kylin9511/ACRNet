## Overview
This is a PyTorch implementation of ACRNet inference. The key results in paper [Aggregated Network for Massive MIMO CSI Feedback](https://arxiv.org/abs/2101.06618) can be reproduced.

## Requirements

The following requirements need to be installed.
- Python >= 3.7
- [PyTorch == 1.6.0](https://pytorch.org/docs/1.6.0/)

## Project Preparation

#### A. Data Preparation

As mentioned in the paper, our dataset is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) channel model. The preprocessed version provided by Chao-Kai Wen in [CsiNet](https://arxiv.org/abs/1712.08919) is recommended. You can download it from [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

Note that all dataset generation related setting can be found in the ACRNet paper if you wish to generate the dataset yourself.

#### B. Checkpoints Downloading
The checkpoints of our proposed BCsiNet can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1474ub1KwMyJIlQwTlqYvVw) (passwd: he1o) or [Google Drive](https://drive.google.com/drive/folders/1YlGLPC6ukvrUi9ChAdRgIAp89qRWFuaF?usp=sharing)

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── ACRNet  # The cloned BCsiNet repository
│   ├── dataset
│   ├── model
│   ├── utils
│   ├── main.py
├── COST2100  # COST2100 dataset downloaded following section A
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder downloaded following section B
│   │     ├── table1
│   │     ├── table2
│   ├── run.sh  # The bash script
...
```

## Key Results Reproduction

The key results reported in Table I of the paper are listed as follows.

Compression Ratio | Methods | Scenario | NMSE  | Checkpoints Path
:---------------: | :-----: | :------: | ----: | ----------------
1/4 | ACRNet-1x  | indoor  | -27.16dB | table1/cr4/1x_in/model.pth
1/4 | ACRNet-1x  | outdoor | -10.71dB | table1/cr4/1x_out/model.pth
1/4 | ACRNet-10x | indoor  | -29.83dB | table1/cr4/10x_in/model.pth
1/4 | ACRNet-10x | outdoor | -13.61dB | table1/cr4/10x_out/model.pth
1/4 | ACRNet-20x | indoor  | -32.02dB | table1/cr4/20x_in/model.pth
1/4 | ACRNet-20x | outdoor | -14.25dB | table1/cr4/20x_out/model.pth
1/8 | ACRNet-1x  | indoor  | -15.34dB | table1/cr8/1x_in/model.pth
1/8 | ACRNet-1x  | outdoor | -7.85dB | table1/cr8/1x_out/model.pth
1/8 | ACRNet-10x | indoor  | -19.75dB | table1/cr8/10x_in/model.pth
1/8 | ACRNet-10x | outdoor | -9.22dB | table1/cr8/10x_out/model.pth
1/8 | ACRNet-20x | indoor  | -20.78dB | table1/cr8/20x_in/model.pth
1/8 | ACRNet-20x | outdoor | -9.68dB | table1/cr8/20x_out/model.pth
1/16 | ACRNet-1x | indoor  | -10.36dB | table1/cr16/1x_in/model.pth
1/16 | ACRNet-1x | outdoor | -5.19dB | table1/cr16/1x_out/model.pth
1/16 | ACRNet-10x | indoor  | -14.32dB | table1/cr16/10x_in/model.pth
1/16 | ACRNet-10x | outdoor | -6.30dB | table1/cr16/10x_out/model.pth
1/16 | ACRNet-20x | indoor  | -15.05dB | table1/cr16/20x_in/model.pth
1/16 | ACRNet-20x | outdoor | -6.47dB | table1/cr16/20x_out/model.pth
1/32 | ACRNet-1x | indoor  | -8.60dB | table1/cr32/1x_in/model.pth
1/32 | ACRNet-1x | outdoor | -3.31dB | table1/cr32/1x_out/model.pth
1/32 | ACRNet-10x | indoor  | -10.52dB | table1/cr32/10x_in/model.pth
1/32 | ACRNet-10x | outdoor | -3.83dB | table1/cr32/10x_out/model.pth
1/32 | ACRNet-20x | indoor  | -10.77dB | table1/cr32/20x_in/model.pth
1/32 | ACRNet-20x | outdoor | -4.05dB | table1/cr32/20x_out/model.pth
1/64 | ACRNet-1x | indoor  | -6.51dB | table1/cr64/1x_in/model.pth
1/64 | ACRNet-1x | outdoor | -2.29dB | table1/cr64/1x_out/model.pth
1/64 | ACRNet-10x | indoor  | -7.44dB | table1/cr64/10x_in/model.pth
1/64 | ACRNet-10x | outdoor | -2.61dB | table1/cr64/10x_out/model.pth
1/64 | ACRNet-20x | indoor  | -7.78dB | table1/cr64/20x_in/model.pth
1/64 | ACRNet-20x | outdoor | -2.69dB | table1/cr64/20x_out/model.pth

The key results reported in Table II of the paper are listed as follows. Note that all the compression ratio in Table II is 1/4.

Methods | Scenario | NMSE  | Checkpoints Path
:--: | :--: | :--: | :--
ACRNet-1x   | indoor  | -27.16dB | table2/1x/in_cr4/model.pth
ACRNet-1x   | outdoor | -10.71dB | table2/1x/out_cr4/model.pth
ACRNet-4x   | indoor  | -28.58dB | table2/4x/in_cr4/model.pth
ACRNet-4x   | outdoor | -13.13dB | table2/4x/out_cr4/model.pth
ACRNet-8x   | indoor  | -29.31dB | table2/8x/in_cr4/model.pth
ACRNet-8x   | outdoor | -13.45dB | table2/8x/out_cr4/model.pth
ACRNet-12x  | indoor  | -30.28dB | table2/12x/in_cr4/model.pth
ACRNet-12x  | outdoor | -13.91dB | table2/12x/out_cr4/model.pth
ACRNet-16x  | indoor  | -30.81dB | table2/16x/in_cr4/model.pth
ACRNet-16x  | outdoor | -14.16dB | table2/16x/out_cr4/model.pth

In order to reproduce the aforementioned key results, you need to download the given dataset and checkpoints. Moreover, you should arrange your project tree as instructed. An example of `Experiments/run.sh` can be found as follows.

``` bash
python /home/ACRNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained '/home/Experiments/checkpoints/table1/cr4/1x_in/model.pth' \
  --batch-size 200 \
  --workers 0 \
  --reduction 4 \
  --expansion 1 \
  --gpu 0 \
  2>&1 | tee log.out
```

> Note that the checkpoint must match exactly with the indoor/outdoor scenario and the hyper-parameter of ACRNet, including the reduction and the expansion. Otherwise the checkpoint loading would fail or the result will be incorrect.

## Acknowledgment

This repository is modified from the [BCsiNet open source code](https://github.com/Kylin9511/BCsiNet). Please refer to it if you are interested. The open source codes for [CRNet](https://github.com/Kylin9511/CRNet) and [CsiNet](https://github.com/sydney222/Python_CsiNet) can be helpful as well if you are interested in the benchmark networks.

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset.
