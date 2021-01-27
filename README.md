# PrismNet

This is a [PyTorch](https://pytorch.org/) implementation of our paper:
## Predicting dynamic cellular protein-RNA interactions using deep learning and in vivo RNA structure
Lei Sun*,  Kui Xu*, Wenze Huang*, Yucheng T. Yang*, Lei Tang, Tuanlin Xiong, Qiangfeng Cliff Zhang

*: indicates equal contribution.

bioRxiv preprint: ([https://www.biorxiv.org/content/10.1101/2020.05.05.078774v1](https://www.biorxiv.org/content/10.1101/2020.05.05.078774v1))

![prismnet](https://github.com/kuixu/PrismNet/wiki/imgs/prismnet.png)



### Table of Contents
- [Getting started](#Getting-started)
- [Datasets](#datasets)
- [Usage](#usage)
- [Copyright and License](#copyright-and-license)
- [Reference](#Reference)

## Getting started


### Requirements
 
 - Python 3.6
 - PyTorch 1.1.0, with NVIDIA CUDA Support
 - pip

### Installation
Clone repository: 

```bash
git clone https://github.com/kuixu/PrismNet.git
```
Install packages:
```bash
cd PrismNet
pip install requirements.txt
pip install -e .
```

## Datasets

### Prepare the datasets

Scripts and pipeline are in preparing, currently, we provide 172 samples data in *.tsv format for training and testing PrismNet.

```
# Download data
cd PrismNet/data
wget http://prismnet.zhanglab.net/data/clip_data.tgz
tar zxvf clip_data.tgz

# Generate training and validation set for binary classification
cd PrismNet
tools/gdata_bin.sh
```


## Usage

### Network Architecture

![prismnet](https://github.com/kuixu/PrismNet/wiki/imgs/prismnet-arch.png)

### Training 

To train one single protein model from scratch, run
```
exp/EXP_NAME/train.sh pu PrismNet TIA1_Hela clip_data 
```
where you replace `TIA1_Hela` with the name of the data file you want to use, you replace EXP_NAME with a specific name of this experiment. Hyper-parameters could be tuned in `exp/prismnet/train.sh`. For available training options, please take a look at `tools/train.py`.

To monitor the training process, add option `-tfboard` in `exp/prismnet/train.sh`, and view page at http://localhost:6006 using tensorboard:
```
tensorboard --logdir exp/EXP_NAME/out/tfb
```

To train all the protein models, run
```
exp/EXP_NAME/train_all.sh
```

### Evaluation
For evaluation of the models, we provide the script `eval.sh`. You can run it using
```
exp/prismnet/eval.sh TIA1_Hela clip_data 
```

### Inference
For inference data (the same format as the *.tsv file used in [Datasets](#datasets)) using the trained models, we provide the script `infer.sh`. You can run it using
```
exp/prismnet/infer.sh TIA1_Hela clip_data /path/to/inference_file.tsv
```

### Compute High Attention Regions
For computing high attention regions using the trained models, we provide the script `har.sh`. You can run it using
```
exp/prismnet/har.sh TIA1_Hela clip_data /path/to/inference_file.tsv
```

### Compute Saliency
For computing saliency using the trained models, we provide the script `saliency.sh`. You can run it using
```
exp/prismnet/saliency.sh TIA1_Hela clip_data 
```

### Plot Saliency Image
For plotting saliency image using the trained models, we provide the script `saliencyimg.sh`. You can run it using
```
exp/prismnet/saliencyimg.sh TIA1_Hela clip_data 
```

### Motif, riboSNitch and structurally variable sites analysis

Scripts of the analysis on integrative motifs, riboSNitch and structurally variable sites are orgnizing, which could be glanced at [here](https://github.com/huangwenze/PrismNet_analysis).
```
git clone https://github.com/huangwenze/PrismNet_analysis
```

### Integrative motif 

The integrative motif could be downloaded at [here](http://prismnet.zhanglab.net/data/Total_motifs-matrix-logo.xlsx).

### Dataset and Results Visualization

We also provide a website [http://prismnet.zhanglab.net/](http://prismnet.zhanglab.net/) to visualize the icSHAPE date and the results.

## Copyright and License
This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.

## Reference

```
@article {Sun2020.05.05.078774,
	title = {Predicting dynamic cellular protein-RNA interactions using deep learning and in vivo RNA structure},
	author = {Sun, Lei and Xu, Kui and Huang, Wenze and Yang, Yucheng T. and Tang, Lei and Xiong, Tuanlin and Zhang, Qiangfeng Cliff},
	year = {2020},
	eprint = {https://www.biorxiv.org/content/early/2020/05/07/2020.05.05.078774.full.pdf},
	journal = {bioRxiv}
}
@article {Sun2020.07.07.192732,
	title = {In vivo structural characterization of the whole SARS-CoV-2 RNA genome identifies host cell target proteins vulnerable to re-purposed drugs},
	author = {Sun, Lei and Li, Pan and Ju, Xiaohui and Rao, Jian and Huang, Wenze and Zhang, Shaojun and Xiong, Tuanlin and Xu, Kui and Zhou, Xiaolin and Ren, Lili and Ding, Qiang and Wang, Jianwei and Zhang, Qiangfeng Cliff},
	year = {2020},
	doi = {10.1101/2020.07.07.192732},
	eprint = {https://www.biorxiv.org/content/early/2020/07/08/2020.07.07.192732.full.pdf},
	journal = {bioRxiv}
}
```