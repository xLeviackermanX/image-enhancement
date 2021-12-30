import numpy as np
import cv2
from utility import histogram, gradient , entropy
from math import log10


def compute_Entropy(img: np.ndarray):
    '''
    Measure information content in the image

    Params
    ------
    img : np.ndarray
        Numpy Image whose entropy to compute  

    Returns
    -------
    entropy : float
        entropy of image in Bits

    '''
    if img.shape[-1] == 3:
        hsi = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        intensity = hsi[:, :, -1]
    else:
        intensity = img
    # hist = histogram(intensity, True)
    # plogp = hist*np.log2(hist)

    # entropy = plogp.sum()

    return entropy(intensity)


def compute_Sharpness(img: np.ndarray):
    '''
    Compute Sharpness measure of Gray scale Image

    Params
    ------
    img : np.ndarray
        Numpy Image whose sharpess to compute  

    Returns
    -------
    grad_magnitude : np.float32
        scaler value as magnitude of sharpness for whole image
    '''
    if img.shape[-1] == 3:
        hsi = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        intensity = hsi[:, :, -1]
    else:
        intensity = img

    gx = gradient(intensity, direction='x')    # gradient of img in x dirn
    gy = gradient(intensity, direction='y')    # gradient of img in y dirn

    grad_magnitude = np.sqrt(gx**2 + gy**2)

    return np.mean(grad_magnitude)


def compute_Colorfullness(im: np.ndarray):
    '''
    Compute Colorfullness measure. It related to range of color and its value decreases when image
    is taken in bad weather condition

    Params
    ------
    img : np.ndarray
        Numpy Image whose colourfullness to compute  

    Return
    ------ 
    C : np.float32
        Colourfullness value of Image
    '''
    img = im.astype('float')
    r, g, b = np.dsplit(img, 3)
    r, g, b = np.squeeze(r), np.squeeze(g), np.squeeze(b)

    del_rg, del_yb = r-g, 0.5*(r+g)-b
    sigma_rg, sigma_yb = del_rg.std(), del_yb.std()
    mu_rg, mu_yb = del_rg.mean(), del_yb.mean()

    simga_rgyb, mu_rgyb = np.sqrt(
        sigma_rg**2+sigma_yb**2), np.sqrt(mu_rg**2+mu_yb**2)

    C = simga_rgyb + 0.3 * mu_rgyb
    return C


def compute_ColorRichness(img: np.ndarray):
    '''
    Compute richness of object colours.

    Params
    ------
    img : np.ndarray
        Numpy Image whose ColorRichness to compute  

    Return
    ------
    mu_saturation : np.float32 
    '''
    saturation = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    mu_saturation = saturation.mean()
    return mu_saturation

#########################################################################
################  EXPERIMENTAL METRICS -- NOT IN PAPER   ################


def compute_DifferentialEntropy(img_input, img_output):
    '''
    Differential Entropy is the increase in Entropy of the Image after an operation

    It can also be called Information Gain(I) in the final Image wrt. to Original

    Params
    ------
    img_input : np.ndarray
        Original Numpy Image  

    img_output : np.ndarray
        Original Numpy Image  

    Returns
    -------
    info_gain : np.float16
        Information Gain of Two images. Also called Differential Entropy

    '''
    entropy_in = compute_Entropy(img_input)
    entropy_out = compute_Entropy(img_output)
    info_gain = entropy_in - entropy_out

    return -info_gain


def compute_KLDivergence(img_input, img_output):
    '''
    Computes KL Divergence between histograms of given two Images
    KL Divergence is the measure of Information Gain when Histogram P(x) is used instead of Q(x)

    It is a measure of relative Entropy.

    Params
    ------
    img_input : np.ndarray
        Original Numpy Image  

    img_output : np.ndarray
        Original Numpy Image  

    Returns
    -------
    KL_div : np.float16
        KL Divergence between histograms of given two Images

    '''
    hist_in = histogram(img_input)
    hist_out = histogram(img_output)

    log_differnce = np.log2(hist_out+9.53674316e-7) - np.log2(hist_in+9.53674316e-7)

    KL_div = -1.0*(hist_out * log_differnce).sum()

    return KL_div


def compute_AMBE(img_input: np.ndarray, img_output: np.ndarray):
    '''
    Compute Absolute Mean Brightness Error

    This metric is associated with global change in brightness

    Params
    ------
    img_input : np.ndarray
        Original Numpy Image  

    img_output : np.ndarray
        Original Numpy Image  

    Returns
    -------
    ambe : np.float16
        Absolute Mean Brightness Error between Original and Final Image
    '''
    intensity_in = cv2.cvtColor(img_input, cv2.COLOR_RGB2HSV)[:, :, -1]
    intensity_out = cv2.cvtColor(img_output, cv2.COLOR_RGB2HSV)[:, :, -1]

    ambe = intensity_out.mean() - intensity_in.mean()

    return abs(ambe)


def compute_PSNR(im_input: np.ndarray, im_output: np.ndarray):
    '''
    Compute Peak Means Sqaured Ratio

    This metric is associated with noise content in the image and thus a measure of quality of image

    Params
    ------
    img_input : np.ndarray
        Original Numpy Image  

    img_output : np.ndarray
        Original Numpy Image  

    Returns
    -------
    psnr : np.float16
        PSNR value in final image Final Image
    '''
    img_input = im_input.astype('float')
    img_output = im_output.astype('float')
    mse = ((img_input - img_output)**2).mean()
    psnr = 10 * log10(255**2/mse)

    return psnr

#########################################################################
###########################  Test Functions   ###########################


def test():
    '''
    Testing utility for all functions
    '''
    in_img_path = ''
    out_img_path = ''

    img_in = cv2.imread(in_img_path)
    img_out = cv2.imread(out_img_path)

    print('{} of Image is : {}'.format('Entropy', compute_Entropy(img_in)))
    print('{} of Image is : {}'.format(
        'Colourfullness', compute_Colorfullness(img_in)))
    print('{} of Image is : {}'.format(
        'Colour Richness of Object', compute_ColorRichness(img_in)))
    print('{} of Image is : {}'.format('Sharpness', compute_Sharpness(img_in)))
    print('{} of Image is : {}'.format(
        'Absolute Mean Brightness Error', compute_AMBE(img_in, img_out)))


if __name__ == '__main__':
    test()
