---
layout: post
category: blog
title: DeepMask Review
tags: [deep learning, semantic, segmentation]
---

This is a review of "DEEP MASK" which is in the category of semantic segmentation. The network consist of two branches. One branch is to predict which pixel belongs to which category. Another branch is to predict whether the object is properly centered in the picture. 

![image](https://ws1.sinaimg.cn/large/007BQ0gBgy1g0g4i1z17bj30ys0edajn.jpg)

The criteria for the second branch is whether the object is centered in the middle and the object must be fully contained in the patch in a given scale range.

Some tricks were used in this paper were:

1. Fine stride max pooling.
2. 1x1 convolution
3. Bipolar interpolation for scaling

I highly question the use of fine stride max pooling. It's basically takes max pooling many times. And each time you shift the pixels to some degree. After all max pooling are done, combine them then feed them to a FC layer or something. Haven't seen this trick applied anywhere else. Also the paper's model adopt "Fully connected layer". However it did mention about multiple scaling and feeding into the model with stride of 16 pixels. "FC" if not changed to CONV the code would spit out error. And it requires many forward passes get the final result. Hence efficiency is low. I'm guessing the author did change FC to CONV but didn't mention.

[Reference](https://towardsdatascience.com/review-deepmask-instance-segmentation-30327a072339)