# PrismNet

This is a [PyTorch](https://pytorch.org/) implementation of our paper:
## Predicting dynamic cellular protein-RNA interactions using deep learning and in vivo RNA structure
Lei Sun*,  Kui Xu*, Wenze Huang*, Yucheng T. Yang, Lei Tang, Tuanlin Xiong, Qiangfeng Cliff Zhang

*: indicates equal contribution.

bioRxiv preprint: ([https://www.biorxiv.org/content/10.1101/2020.05.05.078774v1](https://www.biorxiv.org/content/10.1101/2020.05.05.078774v1))

![prismnet](https://github.com/kuixu/PrismNet/wiki/imgs/prismnet.png)

```
@article {Sun2020.05.05.078774,
	title = {Predicting dynamic cellular protein-RNA interactions using deep learning and in vivo RNA structure},
	author = {Sun, Lei and Xu, Kui and Huang, Wenze and Yang, Yucheng T. and Tang, Lei and Xiong, Tuanlin and Zhang, Qiangfeng Cliff},
	year = {2020},
	eprint = {https://www.biorxiv.org/content/early/2020/05/07/2020.05.05.078774.full.pdf},
	journal = {bioRxiv}
}
```

### Table of Contents
- [Getting started](#Getting-started)
- [Usage](#usage)
- [Contact](#contact)
- [Copyright and License Information](#copyright-and-license-information)

## Getting started


### Requirements
 
 - Python 3.6
 - PyTorch 1.0.0, with NVIDIA CUDA Support
 - pip

### Installation
Clone repository: 

```bash
git clone https://github.com/kuixu/PrismNet.git
```
Install packages:
```bash
cd PrismNet
python setup.py install
```

## Dataset

### Prepare the dataset

Scripts and pipeline are in preparing, currently, we provide a sample data in HDF5 format in `data` folder.

```
data
├── TIA1_Hela.h5
```



## Usage

### Training

to train one single protein model from scratch, run
```
exp/EXP_NAME/train.sh TIA1_Hela
```
where you replace `TIA1_Hela` with the name of the data file you want to use, you replace EXP_NAME with a specific name of this experiment. Hyper-parameters could be tuned in `exp/prismnet/train.sh`. For available training options, please take a look at `tools/train.py`.

You can monitor on http://localhost:6006 the training process using tensorboard:
```
tensorboard --logdir exp/EXP_NAME/out
```

### Evaluation
For evaluation of the models, we provide the script eval.sh. You can run it using
```
exp/EXP_NAME/eval.sh TIA1_Hela
```


### Copyright and License
This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.


