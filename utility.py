import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math
import cmath
from numpy.core.fromnumeric import clip
import random as rand

def rbgtohsi(img):
    n = len(img)
    m = len(img[0])
    r = 1.0*img[:,:,0]/255.0
    g = 1.0*img[:,:,1]/255.0
    b = 1.0*img[:,:,2]/255.0
    hsi = np.zeros((3,n,m),dtype='float')
    mini = np.minimum(np.minimum(r,g),b)
    hsi[1] = 1.0 - (3.0/(r+g+b+0.001)*mini)
    hsi[2] = np.divide(r+g+b,3.0)
    for i in range(n):
        for j in range(m):
            hsi[0][i][j] = 0.5*((r[i][j]-g[i][j])+(r[i][j]-b[i][j]))/(math.sqrt(0.000001+(r[i][j]-g[i][j])**2 + ((r[i][j] - b[i][j])*(g[i][j]-b[i][j]))))
            hsi[0][i][j] = math.acos(hsi[0][i][j])
            if b[i][j]>g[i][j]:
                hsi[0][i][j] = 2.0*math.pi - hsi[0][i][j]

    return hsi

def hsitorgb(H,S,I):
    n = len(H)
    m = len(H[0])
    out = np.zeros((n,m,3),dtype = 'uint8')
    for i in range(n):
        for j in range(m):
            h = H[i][j]
            s = S[i][j]
            v = I[i][j]
            he = h
            if he>=2*math.pi/3.0:
                he -= 2*math.pi/3.0
            if he>=2*math.pi/3.0:
                he -= 2*math.pi/3.0
            x = v*(1.0-s)
            y = v*(1.0 + (s*math.cos(he))/(math.cos(math.pi/3.0 - he)+0.001))
            z = 3.0*v-(x+y)
            r = y
            g = z
            b = x
            if h>=2*math.pi/3.0 and h<4*math.pi/3.0:
                r = x
                g = y
                b = z
            elif h>=4*math.pi/3.0:
                g = x
                b = y
                r = z
            out[i][j][0] = int(255.0*r)
            out[i][j][0] = max(0,out[i][j][0])
            out[i][j][0] = min(255,out[i][j][0])
            out[i][j][1] = int(255.0*g)
            out[i][j][1] = max(0,out[i][j][1])
            out[i][j][1] = min(255,out[i][j][1])
            out[i][j][2] = int(255.0*b)
            out[i][j][2] = max(0,out[i][j][2])
            out[i][j][2] = min(255,out[i][j][2])

    return out


def histogram(img, normalized=False):
    '''
    Compute Histogram of gray image
    '''
    img_flat = img.ravel()
    hist = np.zeros((256), dtype=int)
    # add values at intensity location
    for i in range(img_flat.size):
        hist[int(img_flat[i])] += 1
    if normalized:
        hist = np.divide(hist, img.size)

    return hist


def linContrastStretching(img, a, b, color=False):
    kimg = np.zeros_like(img)
    r = 255
    l = 0
    s = 0
    n = len(img)
    m = len(img[0])
    f = np.array([0]*256)
    for i in img:
        for j in i:
            if color == True:
                avg = j[0]/3.0 + j[1]/3.0 + j[2]/3.0
            else:
                avg = j
            f[int(avg)] += 1
    while f[l]+s < (n*m)/20.0:
        s += f[l]
        l += 1
    s = 0
    while f[r]+s < (n*m)/20.0:
        s += f[r]
        r -= 1
    if r == l:
        return kimg
    fac = (b-a)/(r-l)
    for i in range(n):
        for j in range(m):
            if color == False:
                if img[i][j] < l:
                    kimg[i][j] = a
                elif img[i][j] > r:
                    kimg[i][j] = b
                else:
                    kimg[i][j] = a + int((img[i][j]-l)*fac)
                continue
            for k in range(3):
                if img[i][j][k] < l:
                    kimg[i][j][k] = a
                elif img[i][j][k] > r:
                    kimg[i][j][k] = b
                else:
                    kimg[i][j][k] = a + int((img[i][j][k]-l)*fac)
    return kimg


def colorLinContrastStretching(img, a, b):
    inp = np.transpose(img, (2, 0, 1))
    out = np.zeros_like(inp)
    for i in range(3):
        out[i] = linContrastStretching(inp[i], a, b)
    out = np.transpose(out, (1, 2, 0))
    return out


def entropy(img):
    f = np.zeros(256, dtype='float')
    for i in img:
        for j in i:
            f[j] += 1.0
    n = len(img)
    m = len(img[0])
    f = f/(n*m)
    h = f*np.log2(f+9.53674316e-7)
    return -np.sum(h)


def compress(img, out, k):
    delta = out.astype('float')-img.astype('float')
    h = 0.5*(1.0 + np.tanh(k*(1-np.abs(delta)/128.0) - 3.0))
    ret = img.astype('float') + h*delta
    ret = ret.astype('uint8')
    return ret


def findK(img, out, l, r):
    gold = 0.618
    # x1 = l+0.618*(r-l)
    # x2 = r-0.618*(r-l)
    # i1 = compress(img, out, x1)
    # i2 = compress(img, out, x2)
    # e1 = -entropy(i1)
    # e2 = -entropy(i2)
    # if e1<e2:
    #     return findK(img, out, x2, r)
    # else:
    #     return findK(img , out, l, x1)
    ret = l
    emax = entropy(compress(img,out,l))
    x = l+0.1
    while x<=r:
        e = entropy(compress(img,out,x))
        if e>=emax:
            emax = e
            ret = x
        x+=0.1
    return ret

def compressEqualization(img, out, k = -1):
    if k==-1:
        k = findK(img, out , 1.5, 4.5)

    ret = compress(img,out,k)
    return ret, k




def histEqualization(img):
    kimg = np.zeros_like(img)
    n = len(img)
    m = len(img[0])
    f = np.array([0]*256)
    for i in range(n):
        for j in range(m):
            f[int(img[i][j])] += 1
    mp = np.array([0]*256)
    cur = 0.0
    x = np.mean(img)
    delta = min(255-x, x)
    gmin = x-delta
    gmax = x+delta
    for i in range(256):
        cur += f[i]*1.0
        ip = gmin + ((gmax-gmin)*cur)/(n*m)
        mp[i] = int(ip)

    for i in range(n):
        for j in range(m):
            kimg[i][j] = mp[int(img[i][j])]
    return kimg

def colorHistEqualization(img, typeHE):

    hsi = cv2.cvtColor(img , cv2.COLOR_RGB2HSV)
    hsi = np.transpose(hsi , (2,0,1))
    inter = histEqualization(hsi[2])
    if typeHE != 'Proposed Pipeline':
        hsi[2] = inter
        hsi = np.transpose(hsi , (1,2,0))
        out = cv2.cvtColor(hsi , cv2.COLOR_HSV2RGB)
        return out, 0.0
    hsi[2], bestK = compressEqualization(hsi[2],inter)
    hsi = np.transpose(hsi , (1,2,0))
    out1 = cv2.cvtColor(hsi , cv2.COLOR_HSV2RGB)
    hsi1 = cv2.cvtColor(out1, cv2.COLOR_RGB2HSV)
    hsi1 = np.transpose(hsi1, (2,0,1))
    hsi = np.transpose(hsi, (2,0,1))
    hsi[1] = np.maximum(hsi[1],hsi1[1])
    hsi = np.transpose(hsi , (1,2,0))
    out = cv2.cvtColor(hsi , cv2.COLOR_HSV2RGB)
    return out, bestK


def gradient(img, direction):
    '''
    Compute gradient in any dirction dirn

    Algorithm: Remove 1st and last row of Image and subtract them will give gradient in X 
    '''
    if direction == 'x' or 'X':
        gradient = img.astype('float')[:-1, :] - img.astype('float')[1:, :]
    elif direction == 'y' or 'Y':
        gradient = img.astype('float')[:, :-1] - img.astype('float')[:, 1:]

    return gradient


def restore_Saturation(img_eq: np.ndarray, sat_origin: np.ndarray):
    '''
    Restore Saturation of Image after HE. This is done by maximization of S channel in HSI of Eqaulized 
    Image.

    Params
    ------
    img_eq : np.ndarray
        Histogram Equalized Image in HSI format

    sat_origin : np.ndarray
        Saturation Channel of original Image

    Rteurns
    -------
    img_sat_corrected : np.ndarray
        RGB image with Corrected Saturation 
    '''
    rgb_eq = cv2.cvtColor(img_eq, cv2.COLOR_HSV2RGB)
    hsi_eq = cv2.cvtColor(rgb_eq, cv2.COLOR_RGB2HSV)
    saturation_eq = hsi_eq[:, :, 1]
    saturation_corrected = np.maximum(saturation_eq, sat_origin)

    # TODO: check dimnetion and correct this step if error arises
    img_sat_corrected = np.dstack(
        (hsi_eq[:, :, 0], saturation_corrected, hsi_eq[:, :, 2]))
    img_sat_corrected = cv2.cvtColor(img_sat_corrected, cv2.COLOR_HSV2RGB)

    return img_sat_corrected
