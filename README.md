# vid-cap-ground

## Introduction

This project is aim for reduplicate the paper in pytorch

* [Finding “It”: Weakly-Supervised Reference-Aware Visual Grounding in Instructional Videos](http://ai.stanford.edu/~dahuang/papers/cvpr18-ramil.pdf)

For the visual part, a RPN is used in the repo
* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

The basenet is 

Visual Genome (Train/Test: vg_train/vg_test, scale=600, max_size=1000, ROI Align, category=2500)

model     | #GPUs | batch size |lr        | lr_decay | max_epoch     |  time/epoch | mem/GPU | mAP 
---------|--------|-----|--------|-----|-----|-------|--------|----- 
[VGG-16](http://data.lip6.fr/cadene/faster-rcnn.pytorch/faster_rcnn_1_19_48611.pth)    | 1 P100 | 4    |1e-3| 5   | 20  |  3.7 hr    |12707 MB  | 4.4


## Preparation 

First of all, clone the code
```
git clone https://github.com/jshi31/vid-cap-ground.git 
```

### prerequisites

* Python 3.6
* Pytorch 0.4.0 (<1.0.0)
* CUDA 8.0 or higher
### dependencies
* tensorboardX: Write TensorBoard events with simple function call. Install from [tensorboardX](https://github.com/lanpa/tensorboardX). If you want to visualize the tensorboard events, you may still need to install tensorflow.
* torchtext: torchtext is for obtaining glove feature. Install from [torchtext](https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb)
* opencv
* PWCNet: PWCNet get optical flow for linking tubes. Install from [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch)
* nltk


### Data Preparation

* **YouCookII**: Please download the dataset from [YouCookII](http://youcook2.eecs.umich.edu) to prepare YouCookII datasets.

`cd $ROOT/data/YouCookII`
`python genframes.py --video_dir $RAW_VIDEO_DIR`
The generated frames are stored in `sampled_frames_splnum-1`, under the same parent folder of `$RAW_VIEO_DIR`, then build a soft link to project directory as 
`ln -s $PATH_TO_sampled_frames_splnum-1 $ROOT/data/YouCookII/`
Test dataloader:
Go to $ROOT directory `python lib/datasets/youcook2.py`

### Language Processing Preprocessing Preparation
1. run `python`
2. run `nltk.download('wordnet')`
3. run `nltk.download('punkt')`
4. run `nltk.download('averaged_perceptron_tagger')`
### Pretrained Model

go to project root directory ``ROOT``, create directory ``ROOT/models/vgg16/pretrain/`` and directory ``ROOT/output/``

We used pretrained models in our experiments, VGG16 pretrained on gnome. You can download the pytorch model [VGG-16](http://data.lip6.fr/cadene/faster-rcnn.pytorch/faster_rcnn_1_19_48611.pth) 

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

| GPU model  | Architecture |
| ------------- | ------------- |
| TitanX (Maxwell/Pascal) | sm_52 |
| GTX 960M | sm_50 |
| GTX 1080 (Ti) | sm_61 |
| Grid K520 (AWS g2.2xlarge) | sm_30 |
| Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**


## Train 

To train a mode, simply run:
```
./train.sh
```
## Test
```
./eval_model.sh
```

#### Upper bound

```bash
python lib/datasets/youcook_eval.py
```

Allbox 160476

## Partial result 

Forward time for RPN is 140 fps in batch size 5

## Report 
See word [Report](https://www.dropbox.com/s/t4cnqx7jzx5jwo3/Grounding%20Report.docx?dl=0).

## Data loading 
1. Since the video are extracted with 16 fps, the parameter `sample_rate_val` decide the sample rate. Default 16, which is 1 fps. And this is how the ground truth is annotated. 
 

   
