# DIP Project

> Poor Quality Images are unacceptable. So, Let's *improve* them!!

A Python implementation of paper, titled **Histogram equalization and optimal profile compression based approach for colour image enhancement**, as a part of DIP Project 2021.
[Link](https://www.sciencedirect.com/science/article/abs/pii/S1047320316300529) to the Paper.

## Table of Contents

1. [Abstract](#abstact)
2. [Instructions](#instructions)
3. [Demo](#demo)
4. [Presentation](#presentation)

## Abstract

Images contain much information about a captured scene. However, they can be lost due to environmental conditions or the capturing process resulting in a poor quality image. The quality of an image is related to the colorfulness and contrast of the picture. Histogram Equalization and Contrast Stretching are among the most widely used techniques for improving the information content as well as contrast and colorfulness of an image. However, histogram equalization introduces unwanted artefacts in the image and thus decreases the image quality. The *simple* histogram equalization needs modifications to mitigate its drawbacks. The methods suggested in the paper and implemented in this repo attempt to improve the quality of Images while evading or, at least, minimizing the side effects of those methods.

The method follows a four-step process, i.e. Color Contrast Stretching, Histogram Equalization, Hyperbolic tangent-based profile compression of Histogram magnitude and finally, Saturation Maximization. All the mentioned steps either handle different aspects of image quality, viz. Colorfulness, vividness, etc., or control the side-effects of an operation.

## Instructions

<!-- Add link-->
<!-- add output of pip freeze for dependency builds-->
First You will need to install following python libraries:

- `sudo apt-get install python3-opencv`
- `pip3 install matplotlib`
- `pip3 install streamlit`
- `pip3 install pillow`

Now in order to run GUI, use following command in the main directory:-

- `streamlit run app.py`

Now in the GUI you can simply upload the input image and see the enhanced image along with the different score metrics to judge the quality of the image. In the left sidebar there is a selectbox where you can can switch between the output produced by proposed pipeline and general histogram equalization.

You can also run test.py script which will produce score of 300 images stored in flower directory. Steps that are need to be followed are:
 - Unzip flower.zip
 - python3 test.py
 - A csv file will be created called data.csv which will contain scores of output enhanced images.

## Demo

<!-- Maybe a GIF of Platform, if time permits-->
<img src="https://github.com/Digital-Image-Processing-IIITH/dip-project-team_d-7/blob/main/demo.gif" width="800" height="600" />

It is evident in above video that general histogram equalization, even the adaptive one have some artefacts in the image near low and high intensities. But these same artefacts are not present in the proposed pipeline output.

## Presentation

The presentation slides can be accessed [here](https://github.com/Digital-Image-Processing-IIITH/dip-project-team_d-7/blob/main/final.pptx)

## TODO

- [x] Add Instructions to run the code and replicate results.
- [x] Links to input and output images.
- [ ] Link to Hosted app using Streamlit.
- [x] Link to Presentation
