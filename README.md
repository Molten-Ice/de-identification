# de-identification

The goal of this code is to take an input image, containing one or several individuals, and anonymize all faces present.

## Table of Contents
Version 2: [ Pre-trained GAN](#pretrained-gan)
Version 1: [ DCGAN (end-to-end)](#dcgan)

## Pretrained GAN

Here a pretrained GAN will be used and aligned with the image

## DCGAN

*Last visited 23/04/2023*

This was my first attempt at the problem and I sought to develop- an end-to-end solution to the problem.

The overall goal is to replace the obscured white area with a synthesized face, which looks realistic however is not identifiable as the original individual.


1. [Dataset Creation](#example)
2. [DCGAN Image generation](#example2)
3. [Facial Inpainting](#third-example)
4. [Synthesized faces within image](#fourth-examplehttpwwwfourthexamplecom)


## Example
## Example2
## Third Example
## [Fourth Example](http://www.fourthexample.com) 
A quick overview:
- Dataset generation

First:
- Us


![gif1](dev-notebooks/media/10-epochs-gan-fast.gif)

![example-masking](dev-notebooks/media/example-masking.png)

![dataset-creation](dev-notebooks/media/dataset-creation.png)

![faces_before](dev-notebooks/media/faces_before.png)

![faces_after](dev-notebooks/media/faces_after.png)

![before_batch](dev-notebooks/media/before_batch.png)

![after_batch](dev-notebooks/media/after_batch.png)


