# NAFAE

## Introduction

This project is the Pytorch implementation of [Not All Frames Are Equal: Weakly-Supervised Video Grounding
with Contextual Similarity and Visual Clustering Losses](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Not_All_Frames_Are_Equal_Weakly-Supervised_Video_Grounding_With_Contextual_CVPR_2019_paper.pdf) in [CVPR 2019](http://cvpr2019.thecvf.com/).  
**Video Grounding Definition**: Given a video segment with its language description, the aim is to localize objects query from the description to the video.

**Note**: this repository only provides the implementation for Finite Class Training mode for YouCookII Dataset.

## Prerequisites
* Python >= 3.6
* Pytorch >= 0.4.0 (<1.0.0)

## Installation
### Clone the NAFAE repository
```
git clone https://github.com/jshi31/NAFAE.git 
```
### Dependencies
* torchtext: torchtext is for obtaining glove feature. Install from [torchtext](https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb)
* opencv

### Data Preparation
1. Please download the dataset from [YouCookII](http://youcook2.eecs.umich.edu) to prepare YouCookII datasets.
We only need the folder `raw_videos` and the path to it is denoted as `$RAW_VIDEO_DIR`.  
**Note:** Please ensure that you downloaded all of the 2000 videos. If some videos are missing, please contact the authors to get them. 
2. Parse video into frames
```
cd $ROOT/data/YouCookII 
python genframes.py --video_dir $RAW_VIDEO_DIR
```
The generated frames are stored in `sampled_frames_splnum-1`, under the same parent folder of `$RAW_VIDEO_DIR`, then build a soft link to project directory as   
```
ln -s $PATH_TO_sampled_frames_splnum-1 $ROOT/data/YouCookII/
```
3. Test dataloader:  
```
python $ROOT/lib/datasets/youcook2.py
```
It is safe if no error reported.

### Model Preparation

Create directory ``$ROOT/models/vgg16/pretrain/`` 

We used [faster RCNN](https://github.com/jwyang/faster-rcnn.pytorch) with VGG16 backbone pretrained on Visual Gnome for region proposals. Download and put the [VGG16 model](http://data.lip6.fr/cadene/faster-rcnn.pytorch/faster_rcnn_1_19_48611.pth) as ``$ROOT/models/vgg16/pretrain/faster_rcnn_gnome.pth``

### Compilation FasterRCNN Layers

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

| GPU model  | Architecture |
| ------------- | ------------- |
| TitanX (Maxwell/Pascal) | sm_52 |
| GTX 960M | sm_50 |
| GTX 1080 (Ti) | sm_61 |
| Grid K520 (AWS g2.2xlarge) | sm_30 |
| Tesla K80 (AWS p2.xlarge) | sm_37 |

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```
It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. 

## Training 
```
./train.sh
```
## Evaluation
Evaluate on test set
```
./test_model.sh
```
Evaluate on validation set 
```
./eval_model.sh
```
Please change *checksession*, *checkepoch*, *checkbatch* to the same with the training setting .

## Visualization
1. Visualize groundings
Specify the *train_vis_freq* and *val_vis_freq* as $n so that the the detected result is visualized in `$ROOT/output` every $n batches
2. Visualize training curve
```
tensorboard --logdir runs
```

## Pretrained Final Model 
In order to get the result in our paper, download and put the [Final Model](https://uofr-my.sharepoint.com/:u:/g/personal/jshi31_ur_rochester_edu/EXxsrJ66cyVKsmd4ANQJVfsBrKVDvgBKvao7whSN4DUZmA?e=ZqVSp1) into ``$ROOT/output/models/vgg16/YouCookII/``. Run 
```angular2html
./test_model.sh
./eval_model.sh
``` 
|      | macro box accuracy % | macro query accuracy % |
|:----:|:--------------------:|:----------------------:|
|  val |         39.48        |          41.23         |
| test |         40.62        |          42.36         |

   
