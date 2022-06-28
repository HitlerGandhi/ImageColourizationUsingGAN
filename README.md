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


![rgb (1)](https://user-images.githubusercontent.com/59966711/176247189-d3dae257-358b-449e-91c2-5f3f45fb747e.jpg)

### L*A*B
In L*a*b color space the first channel L, encodes the Lightness of each pixel and when we visualize this channel it appears as a black and white image.
The *a and *b channels encode how much green-red and yellow-blue each pixel is, respectively.

![lab](https://user-images.githubusercontent.com/59966711/176247114-a0e680de-d9b2-44c5-87a1-84a3d2610dbd.jpg)


## Approach

The solution to colorization problem was proposed in
[***Image-to-Image Translation with Conditional Adversarial Networks***](https://arxiv.org/abs/1611.07004) paper
, proposed a general solution to many image-to-image tasks in deep learning.

In this approach, we define two losses:L1 loss, 
which makes it a regression task, and 
an adversarial (GAN) loss,which helps to
solve the problem in an unsupervised manner.

## GAN Model
**Generative Adversarial Networks(GAN in short)** consist of two Artificial Neural Networks or Convolution Neural Networks models namely Generator and Discriminator which are trained against each other (and thus Adversarial).

In our problem , the generator model takes a grayscale image (1-channel image) and produces a 2-channel image, a channel for \*a and another for \*b.
The discriminator, takes these two produced channels and concatenates them with the input grayscale image and decides whether this new 3-channel image is fake or real. Of course the discriminator also needs to see some real images (3-channel images again in Lab color space) that are not produced by the generator and should learn that they are real. 

## Implementation

### 1. Loading Images

We have used a total of 10000 images from COCO dataset out of which 8000 images were used for training the model and 2000 images were usedfor validation and testing of model.

```python
# Importing Libraries
import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
```

```python
!pip install -U fastai
```

```python
from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
paths = glob.glob(coco_path + "/*.jpg") # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 10000 images randomly
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))
```
```python
# Shows Images
_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")
```
![image](https://user-images.githubusercontent.com/59966711/176251367-2016846c-521e-4002-8b44-e28e3b87f914.png)

### 2. Making Training and Validation datasets and dataloaders

```python
```


