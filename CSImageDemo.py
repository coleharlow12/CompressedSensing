from PIL import Image
from PIL import ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt

# Purpose

# Implementation

# Loads the original image
cwd = os.getcwd()										#Gets current working directory
imgPath = os.path.join(cwd,"images","ManwithCam.jpg")	#Creates path to image
im = ImageOps.grayscale(Image.open(imgPath)) 			#Converts to grayscale
OrigIm = np.asarray(im)									#Stores data in numpy array

im.show()

# Take FFT of the image to show its frequency composition

NX = OrigIm.shape[0]+1 #size of x after fft
NY = OrigIm.shape[1]+1 #size of y after fft

OrigImFFT = np.fft.fftshift(np.fft.fft2(OrigIm,s=[NX,NY])) 	#Takes fft and shift along all axes
ImFFTNorm = 10*np.log10(np.absolute(OrigImFFT)/np.max(np.absolute(OrigImFFT)))

xf = np.arange(start=-NX/2+1,stop=NX/2+1,step = 1)
yf = np.arange(start=-NY/2+1,stop=NY/2+1,step = 1)

xfg,yfg = np.meshgrid(xf,yf)
print(xf)
print(NX/2)
print(yfg.shape)
print(ImFFTNorm.shape)


plt.pcolor(xfg,yfg,ImFFTNorm,cmap='jet');
plt.colorbar();plt.clim(-20,0)
plt.xlabel('cycles per aperture',fontsize=16,c='black')
plt.ylabel('cycles per aperture',fontsize=16,c='black')

plt.show()

# Preparation for Sparse Recovery
