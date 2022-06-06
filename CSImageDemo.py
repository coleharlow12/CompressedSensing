from PIL import Image
from PIL import ImageOps
import numpy as np
import os

# Purpose

# Implementation

# Loads the original image
cwd = os.getcwd()										#Gets current working directory
imgPath = os.path.join(cwd,"images","ManwithCam.jpg")	#Creates path to image
im = ImageOps.grayscale(Image.open(imgPath)) 			#Converts to grayscale
OrigIm = np.asarray(im)									#Stores data in numpy array

print(OrigIm.shape)

im.show()

# Take FFT of the image to show its frequency composition
OrgImFFT = np.fft.fft2(OrigIm)


# Take wavelet decomposition to show its composition
