# Image Colorization Using Conditional GANs



## Problem Statement

The task of colourizing black and white photographs necessitates a lot of human input and hardcoding.

The goal is to create an end-to-end deep learning pipeline that can automate the task of image colorization by taking a black 
and white image as input and producing a colourized image as output.

## Final Model Output


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Introduction to colorization problem

### RGB 

In RGB color space, there are 3 numbers for each 
pixel indicating how much red, green and blue given
pixel is.

In the following image , the leftmost image is
the "main image" and the other three are resectively
,red, blue and green channel and combining the three
channels we get "main image".

<p align="center">
<img src="https://res.cloudinary.com/practicaldev/image/fetch/s--7uHGwEG8--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://i.ibb.co/HgnybWG/rgb.png" height = "200" width = "800"/>
</p>

### L*A*B
In L*a*b color space the first channel L, encodes the Lightness of each pixel and when we visualize this channel it appears as a black and white image.
The *a and *b channels encode how much green-red and yellow-blue each pixel is, respectively.

<p align="center">
<img src="https://res.cloudinary.com/practicaldev/image/fetch/s--7uHGwEG8--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://i.ibb.co/HgnybWG/rgb.png" height = "200" width = "800"/>
</p>


## Approach

The solution to colorization problem was proposed in
[***Image-to-Image Translation with Conditional Adversarial Networks***](https://arxiv.org/abs/1611.07004) with Conditional Adversarial Networks paper
, proposed a general solution to many image-to-image tasks in deep learning which one of those was colorization.

In this approach, we define two losses:L1 loss, 
which makes it a regression task, and 
an adversarial (GAN) loss,which helps to
solve the problem in an unsupervised manner.

## GAN Model

### Generator

### Discriminator

### NET loss function we optimize 
