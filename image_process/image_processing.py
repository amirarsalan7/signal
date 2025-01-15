import numpy as np
from scipy.fftpack import fft2, fftshift, dct
import pywt
from PIL import Image

def load_image(filepath):
    img = Image.open(filepath).convert('L')
    return np.array(img)

def compute_fft(img):
    fft_result = fft2(img)
    fft_shifted = fftshift(fft_result)
    return np.log(np.abs(fft_shifted) + 1)

def compute_dct(img):
    dct_result = dct(dct(img.T, norm='ortho').T, norm='ortho')
    return np.log(np.abs(dct_result) + 1)

def compute_dwt(img):
    coeffs = pywt.dwt2(img, 'haar')
    return coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]
