# Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression

YOLOv5 with $\alpha$-IoU losses implemented in PyTorch

If you use this work, please consider citing:

```
@inproceedings{Jiabo_Alpha-IoU,
  author    = {He, Jiabo and Erfani, Sarah and Ma, Xingjun and Bailey, James and Chi, Ying and Hua, Xian-Sheng},
  title     = {Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression},
  booktitle = {NeurIPS},
  year      = {2021},
}
```

## Modifications in this repository

This repository is a fork of [ultralytics/yolov5](https://github.com/ultralytics/yolov5), with an implementation of $\alpha$-IoU losses while keeping the code as close to the original as possible.

### $\alpha$-IoU Losses

$\alpha$-IoU losses can be configured in Line 131 of [utils/loss.py](), functionesd as 'bbox_alpha_iou'. The $\alpha$ values and types of losses (e.g., IoU, GIoU, DIoU, CIoU) can be selected in this function, which are detailed in [utils/general.py](). Note that we should use $\epsilon$ to avoid torch.pow(0, alpha) or denominator=0.

### Configurations

Configuration files can be found in [data](). We do not change either 'voc.yaml' or 'coco.yaml' used in the original repository. However, we could do more experiments. E.g.,

```
voc25.yaml # use randomly 25% PASCAL VOC as the training set
voc50.yaml # use randomly 50% PASCAL VOC as the training set
```

Code for generating different small training sets is in [generate_small_sets.py](). Code for generating different noisy labels is in [generate_noisy_labels.py](), and we should change the 'img2label_paths' function in [utils/datasets.py]() accordingly.


## Train and evaluation commands

For detailed installation instruction and network training options, please take a look at the README file or issue of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Following is a sample command we used for training and testing Faster R-CNN with GIoU.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --use_tfboard
python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --load_ckpt {full_path_of_the_trained_weight}
```

We can also randomly generate some images for detection and visualization results in [generate_detect_images.py]().

## Pretrained weights

Here are some pretrained models using the configurations in this repository, with $\alpha=3$ in all experiments. It is a very simple yet effective method so that people is able to quickly apply our method to existing models following the 'bbox_alpha_iou' function.

 - [$\mathcal{L}_{\textrm{IoU}}$ for YOLOv5s on PASCAL VOC]()
 - [$\mathcal{L}_{\alpha \textrm{-IoU}}$ for YOLOv5s on PASCAL VOC]()
 - [$\mathcal{L}_{\textrm{DIoU}}$ for YOLOv5s on PASCAL VOC]()
 - [$\mathcal{L}_{\alpha \textrm{-DIoU}}$ for YOLOv5s on PASCAL VOC]()

