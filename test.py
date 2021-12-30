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
import pandas as pd

colorness = []
sharpness = []
entro = []
colorrich = []
noise  = []

bestK = []

colorness1 = []
sharpness1 = []
entro1 = []
colorrich1 = []
noise1 = []

for i in range(100,400):
	path = './flower/flower_0'+str(i)+'.jpg'
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	img = cv2.resize(img, (300,300) , cv2.INTER_CUBIC)
	img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
	stretch = colorLinContrastStretching(img,0,255)
	out, bestk = colorHistEqualization(stretch,'Proposed Pipeline')

	colorness.append(compute_Colorfullness(out))
	colorrich.append(compute_ColorRichness(out))
	entro.append(compute_Entropy(out))
	sharpness.append(compute_Sharpness(out))
	noise.append(compute_PSNR(img , out))
	bestK.append(bestk)
	out , bestk = colorHistEqualization(stretch,'General Histogram Equalization')
	sharpness1.append(compute_Sharpness(out))
	colorness1.append(compute_Colorfullness(out))
	colorrich1.append(compute_ColorRichness(out))
	entro1.append(compute_Entropy(out))
	noise1.append(compute_PSNR(img , out))
	print(str(i)+'  Done')


data = {'colorness': colorness,'sharpness': sharpness,'entro':entro ,'colorrich': colorrich,'noise': noise,'bestK': bestK,'colorness1':colorness1 ,'sharpness1' : sharpness1,'entro1':entro1 ,'colorrich1': colorrich1, 'noise1': noise1}
df = pd.DataFrame(data)
df.to_csv('data.csv')