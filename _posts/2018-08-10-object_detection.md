---
layout: post
title: YOLO vs Faster RCNN
tags: [deep learning, detection]
category: post
comments: true
---
This post talks about YOLO and Faster-RCNN. These are the two popular approaches for doing object detection that are anchor based. Faster RCNN offers a regional of interest region for doing convolution while YOLO does detection and classification at the same time. I would say that YOLO appears to be a cleaner way of doing object detection since it's fully end-to-end training. The Faster RCNN offers end-to-end training as well, but the steps are much more involved. Nevertheless I will describe both approaches carefully in detail.

## Faster RCNN

### Architecture of Faster RCNN
The Faster RCNN is based of VGG16 as shown in the above image:
![image](https://ws3.sinaimg.cn/large/007BQ0gBgy1fzqqlthpt3j30c30ffmyj.jpg)
The author basically takes the original image as input and shrinks it 16x times at conv5 layer. And then applies 1x1 convolution to that feature map two times. One 1x1 convolution ouputs 2K output channels, the K stands for the number of anchors and number 2 here means either it's foreground or background. In the original paper, the author set three ratios and three scales for anchor boxes, making the total number $K=9$.

Another 1x1 convolution outputs 4K output channels. This number stands for 4 coordinate related information. They each are `x-center`,`y-center`,`width`,`height`.

Aside from outputting these 4 predictions regarding coordinates and 2 prediction regarding foreground and background. The network will also generate training labels on the fly. It takes all anchor boxes on the feature map and calculate the IOU between anchors and ground-truth. It then decides what which anchor is responsible for what ground-truth boxes by the following rules:

1. IOU > 0.7 or the biggest IOU, anchor boxes are deemed as foreground.
2. IOU <= 0.3, anchor boxes are deemed as background.
3. Any IOU in between should be labeled as "Don't care"

Then randomly sample 128 anchor boxes as foreground samples and 128 anchor boxes as background. If foreground samples are less than 128 than complement the samples with negative samples (background). This part generates labels for anchor boxes. It would also need to generate the offset locations between sampled anchor-boxes and ground-truth boxes.

```python
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
```

`gt_ctr_x` stands for ground truth coordinates. And `ex_widths` stands for anchor width.

### Training of region of interest proposal network
To get a better understanding of Faster-RCNN, I've provided the above directed graph for easier understanding.
<div class="mermaid">
graph LR
    A[Input Image]-->B[Feature Extraction]
    B-->E[Training Labels]
    B-->C[Coordinates Prediction, 4K]
    B-->D[Foreground and Background Prediction, 2K]
    C-->E[Loss function]
    D-->E[Loss function]
</div>

### Post Processing
Not much to say about this part. The predicted boxes are filtered with non-maximum suppression, which filters out boxes with IOU less than some threshold. After NMS, we preserve the top $k$ boxes.

### Classification
After ROI is done, we are left with a lot of choices for doing classification. Of course one can feed each region of interest into a pretrained CNN network for inference. However it may be slow if we fed them one by one. In the original RCNN paper, the author proposed that we resize every region of interest first, then run CNN to extract features for those region, then finally uses some kind of classification method such as SVM to decide what to label them. With Faster RCNN, also in Fast RCNN, the author proposed that we use ROI Pooling (Region of Interest Pooling) for extracting fixed sized feature maps.

Sadly I haven't gone through the code for ROI pooling myself. However I do know it takes a region of interest in the original image or the feature map and split the region into fixed sections. It then it go through maximum pooling for theses fixed sections and stack multiple feature layers onto each other. Of course for object detection there exist multiple region of interest and multiple feature maps. But let's keep it simple to say we're only left with one region of interest and three layers of feature maps for that region.

### Loss function
Faster RCNN uses cross-entropy for foreground and background loss, and l1 regression for coordinates.
![](http://om1hdizoc.bkt.clouddn.com/18-8-14/41961686.jpg)

## YOLO
YOLO stands for You Only Look Once. In practical it runs a lot faster than faster rcnn due it's simpler architecture. Unlike faster RCNN, it's trained to do classification and bounding box regression at the same time.

### Architecture of YOLO
The architecture of YOLO got it's inspiration from GoogleNet. We can view the architecture below:

![](http://om1hdizoc.bkt.clouddn.com/18-8-14/52468198.jpg)

We can see that it has some 1x1 convolutions in between to increase the non-linearity of network and also to reduce feature spaces of the network.

### Pipeline of YOLO
Before we do anything with YOLO, we have to prepare training labels. The process is shown below:
<div class="mermaid">
graph TB
    B1(Input Image)-->B2(7x7 Grid labels with 25 channels)
    B3(and ground truths)-->B2
</div>

For the 25 channels, 20 as output category, 4 as ground truth coordinates, 1 as ground truth confidence.

After preprocessing with ground truth and training data, we go through the training process:

<div class="mermaid">
graph TB
    A[Input Image]-->B[Resize 448x448]
    B-->C[Feature Extraction 7x7x30]
    C-->E[Predict 2 anchor boxes plus 2 confidence, so total output=10]
    C-->D[Predict cell category, total output=20]
    D-->G
    E-->F[Compute IOU and masks for anchor boxes and grid]
    F-->G[Compute loss function]
    B2(7x7 Grid labels with 25 channels)-->G
</div>

### Post Processing
This part is similiar to Faster RCNN, so I will not describe them here.

### Loss function
YOLO used l2 loss for bounding box regression, classification.
![](http://om1hdizoc.bkt.clouddn.com/18-8-14/97046698.jpg)

## Conclusion and comparison
We can see that YOLO and Faster RCNN both share some similarities. They both uses a anchor box based network structure, both uses bounding both regression. Things that differs YOLO from Faster RCNN is that it makes classification and bounding box regression at the same time. Judging from the year they were published, it make sense that YOLO wanted a more elegant way to do regression and classification. YOLO however does have it's drawback in object detection. YOLO has difficulty detecting objects that are small and close to each other due to only two anchor boxes in a grid predicting only one class of object. It doesn't generalize well when objects in the image show rare aspects of ratio. Faster RCNN on the other hand, do detect small objects well since it has nine anchors in a single grid, however it fails to do real-time detection with its two step architecture.

### Reference
[Faster RCNN Reference](https://zhuanlan.zhihu.com/p/24916624?refer=xiaoleimlnote)
