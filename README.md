#VAE for Video Instance Segementation Tracking 

This source code is a personal customized implementation based on the proposed idea in [Video Instance Segmentation Tracking with a Modified VAE Architecture](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Video_Instance_Segmentation_Tracking_With_a_Modified_VAE_Architecture_CVPR_2020_paper.pdf)

![Prediction1](./Media/boar.png)
![Prediction2](./Media/deer.png)
![Prediction3](./Media/fox.png)
![Prediction4](./Media/hare.png)

## Installation
1. Install dependencies

```sh
pip3 install -r requirements.txt
```

## Usage
There 5 python files in this repository:
1. dataset.py : defines datasets for the training and evaluation of Mask-RCNN and VAE, reading from wild animals dataset annotation files.
2. model.py: contains model classes.
3. utils.py: contains related functions.
4. evaluation.py: post-processes prediction and saves results in .txt file for MOT, MOTS, and COCO evaluation.
5. run.py: trains or uses the trained parameters to get prediction results.
