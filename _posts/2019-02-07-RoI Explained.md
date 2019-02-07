---
layout: post
category: blog
title: ROI pooling, align, warping
tags: [deep learning, ROI, detection]
---


First question. How how ROI (region of interest) in the original image get mapped to a feature map?

1. Computation of receptive field
2. Computation of coordinates mapped back to feature map

The first one is easy enough. It's basically the following formula used backward.

**output field size = ( input field size - kernel size + 2*padding ) / stride + 1**

when used in backward:
$$r_i = s_i\cdot(r_{i+1}-1) + k_i -2*\text{padding}$$

when reflected in code, it becomes:
```python
def inFromOut(net, layernum):# computing receptive field from backward
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF -1)* stride) + fsize - 2*pad
    return RF
```

The second question. For reference I just copied the formula here:
$$p_i = s_i \cdot p_{i+1} + ((k_i-1)/2 - \text{padding}) $$
for conv/pooling.

Basically you are mapping the coordinate of feature map for next layer back to the coordinate for that feature point's receptive field (center) of previous feature map.

A picture can well explain this: 
![kJqliq.png](https://s2.ax1x.com/2019/02/05/kJqliq.png)  

So using this formula to get to the relationship between the coordinates of ROI in the original image and the coordinates of feature map.

Usually you want to mapping area of coordinates of the feature map a little smaller than the formula calculated. So you just do this:

For the upper left corner coordinate. We have x coordinate:
$$x^{'} = \left \lfloor{x}\right \rfloor + 1$$
and for y coordinate:
$$y^{'} = \left \lfloor{y}\right \rfloor + 1$$

For the bottom right corner, vice versa for x coordinate：
$$x^{'} = \left \lfloor{x}\right \rfloor - 1$$
and for y coordinate:
$$y^{'} = \left \lfloor{y}\right \rfloor - 1$$
where $x^{'}$and $y^{'}$ is the coordinate in the feature map.

[ROI Mapping Reference](https://zhuanlan.zhihu.com/p/24780433)

---
## ROI pooling

This concept is easy when you understand the above.
So here's a reference for your interest. I won't bother explaining this over here. I just put some bulletpoints here in case the reference fails.

[ROI Pooling Reference](https://deepsense.ai/region-of-interest-pooling-explained/)

    ## Region of interest pooling — description
    
    Region of interest pooling is a neural-net layer used for object detection tasks. It was first proposed by Ross Girshick in April 2015 (the article can be found  [here](https://cdn-sv1.deepsense.ai/wp-content/uploads/2017/02/1504.08083.pdf)) and it achieves a significant speedup of both training and testing. It also maintains a high detection accuracy. The layer takes two inputs:
    
    1.  A fixed-size feature map obtained from a deep convolutional network with several convolutions and max pooling layers.
    2.  An  N x 5 matrix of representing a list of regions of interest, where N is a number of RoIs. The first column represents the image index and the remaining four are the coordinates of the top left and bottom right corners of the region.

The procedure:

    What does the RoI pooling actually do? For every region of interest from the input list,  it takes a section of the input feature map that corresponds to it and scales it  to some pre-defined size (e.g., 7×7). The scaling is done by:
    
    1.  Dividing the region proposal into equal-sized sections (the number of which is the same as the dimension of the output)
    2.  Finding the largest value in each section
    3.  Copying these max values to the output buffer
   
**The drawbacks of ROI pooling:**
  ![kYg2Kx.png](https://s2.ax1x.com/2019/02/06/kYg2Kx.png)
  As we can see it in the picture, the ROI pooling has these roundoff errors that may occur. Due to floating point division.

The first round off error comes when you map the coordinate on image to coordinates on the feature map. Suppose the divident is 32. Then a floor function after a division will make the the coordinate on the feature map to lose 0.78*32 on the original input image.

The second round off error comes when coordinates on the feature map get quantized on the RoI pooling layer. Suppose 7x7 is what we set for the RoI pooling layer. Then floor after 20/7 will make every grid on the 7x7 map 2x2. Which means you will lose $(20-2*7) * (7+5) * 4 = 6*12*4$ pixels on the original feature map. Thus loss of resolution when feeding it through FC layers and softmax layers and so on.

---

## ROI Align

ROI Align comes from the "Mask RCNN" paper. It mainly deals the round-off errors that was introduced with ROI pooling.

![kY7gb9.png](https://s2.ax1x.com/2019/02/06/kY7gb9.png)

The difference was to introduce bilinear interpolation when calculate the pixel's value for the floating  point coordinate. For example (18.4, 240.8). This is a floating point coordinate. We however know the what pixel value for (18, 240) and (19, 241). So to estimate (18.4, 240.8) we can use a technique used in many image processing tricks that is called *bilinear interpolation*. 

The step are processed as follow:
1. The coordinates on the feature map is not quantizied as in RoI Pooling.
2. The coordinates on the RoI pooling layer is also not quantizied as in the original RoI Pooling.
3. The ROI pooling layer divide the feature map into M by N grid. For each small grid, the unit is then sampled K times. For the MaskRCNN paper they used K=4 for best result.
4. Divide each unit equally by 4 means finding the center pixel values for the these 4 regions in the unit. Of course these centers are floating point based. Therefore we use **bilinear interpolation** to predict its value.
5. After bilinearr interpolation, we perform maxpooling on thses 4 samples to output the unit's value. 

[Bilinear interpolation reference](https://en.wikipedia.org/wiki/Bilinear_interpolation)

[ROI Align Reference](https://blog.csdn.net/Bruce_0712/article/details/80287385)

RoIAlign is reported to outperform RoIPooling on both COCO and VOC 2007 dataset. The COCO dataset is more significant due to more smaller bounding boxes with smaller objects for recognition.

---

## RoI Warping Layer

Another technique that was proposed by other researcher at MS Asia. The RoI Warping Layer crop and warp a certain ROI on the feature map to a fixed dimension. 

They still use bipolar interpolation for enlarging or shrinking the image to the same dimension. 

[Java code for bipolar interpolation for image enlarging](http://tech-algorithm.com/articles/bilinear-image-scaling/)

After the warping layer, which is differentiable, they perform the standard max pooling operation for a grid say like 7x7.

The difference between ROI warping and ROI align is that warping changes the shape of the feature map. It's not clear how max pooling is done after they warps. Perhaps still uses bipolar interpolation? But if you define a fixed size warped feature map. Then there may not be any issue with floating number. Anyway, they are both differentiable thanks to bipolar interpolation which includes positions when caculation scores for classification.

---
More references:

[Good blog to browse](http://dubur.github.io/)

[Instance-aware Semantic Segmentation via Multi-task Network Cascades](https://arxiv.org/abs/1512.04412)
