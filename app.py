import streamlit as st
from PIL import Image
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import cv2
from utility import *
from metrics import *

st.title("Image enhancement using histogram equalization")
up_img = st.file_uploader("Upload an image", type=("png", "jpg", "jpeg"))

typeHE = st.sidebar.selectbox('Select method for contrast enhancement', ['Proposed Pipeline' , 'General Histogram Equalization'])

if up_img != None:
    image = Image.open(up_img)
    img = np.array(image)
    img = cv2.resize(img, (300,300) , cv2.INTER_CUBIC)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    fig , ax = plt.subplots()
    ax.imshow(img)
    st.pyplot(fig)
    f, arr = plt.subplots()
    stretch = colorLinContrastStretching(img,0,255)
    out, bestK = colorHistEqualization(stretch,typeHE)
    arr.imshow(out)
    st.write('')
    st.write('')
    st.header('Enhanced image is: ')
    st.pyplot(f)

    col1, col2 = st.columns(2)
    data = compute_Sharpness(out)
    data = round(data,2)
    col1.metric(label = 'Sharpness' , value=data)
    data = compute_Colorfullness(out)
    data = round(data,2)
    col2.metric(label = 'Colorfullness' , value = data)
    col21, col22 = st.columns(2)
    data = compute_ColorRichness(out)
    data = round(data,2)
    col21.metric(label = 'Color Richness' , value=data)
    data = compute_Entropy(out)
    data = round(data,2)
    col22.metric(label = 'Entropy' , value = data)

    bestK = round(bestK,2)
    col31, col32 = st.columns(2)
    data = compute_PSNR(img , out)
    data = round(data,2)
    col31.metric(label = 'Noise' , value=data)
    col32.metric(label = 'Best Gain Factor' , value = bestK)