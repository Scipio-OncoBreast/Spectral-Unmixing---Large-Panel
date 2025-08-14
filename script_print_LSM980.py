# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:22:48 2024

@author: lorenzo.scipioni
"""
#%% import and function definition
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import LS_Functions as lsf
import random
import skimage
import pandas as pd
import czifile
import re

from tkinter.filedialog import askopenfilenames
from gentable import wavelen2rgb
from skimage.filters import median, gaussian
from skimage.morphology import disk
from cellpose import models
from scipy import ndimage as nd
from aicsimageio import AICSImage

def Arbitrary_CCF(img1,img2,Logic_Mask):
    CCF_mask = CCF_mask_compute(Logic_Mask)
    G = (np.sum(img1)/np.sum(Logic_Mask))*(np.sum(img2)/np.sum(Logic_Mask))
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    CCF12 = F1*np.conjugate(F2)
    CCF12 = np.real(np.fft.fftshift(np.fft.ifft2(CCF12),axes=(0,1)))
    CCF_masked = CCF12/CCF_mask/G
    return CCF_masked
def CCF_mask_compute(mask):
    F_mask = np.fft.fft2(mask)
    CCF_mask = F_mask*np.conjugate(F_mask)
    CCF_mask = np.real(np.fft.fftshift(np.fft.ifft2(CCF_mask),axes=(0,1)))
    return CCF_mask
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile
def MakeRGB_fromIMG(IMG_tiled,Color_matrix=[[1,0,1],[0,1,1],[0,1,0],[1,1,0],[1,1,1]]):
    S = IMG_tiled.shape
    RGB = np.zeros((S[1],S[2],3))
    for i_col in range(S[0]):
        for i_rgb in range(3):
            RGB[:,:,i_rgb] = RGB[:,:,i_rgb] + IMG_tiled[i_col,:,:]*Color_matrix[i_col][i_rgb]
    RGB = RGB/np.percentile(RGB,99)
    return RGB

def binArray(data, axis = (0,1), binstep = 2, func=np.nanmean):
#modified from https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    for ax in axis:
        argdims[0], argdims[ax]= argdims[ax], argdims[0]
        data = data.transpose(argdims)
        data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+1)),0),0) for i in np.arange(dims[ax]//binstep)]
        data = np.array(data).transpose(argdims)   
    return data
def Get_GS_Pure(Spectra,i_dyes):
    g, s, ph, m = lsf.PhasorTransform_Slow_2D(Spectra,n_harmonic=1)
    GS_Pure = np.zeros((len(i_dyes),len(i_dyes)))
    GS_Pure[0,:] = g
    GS_Pure[1,:] = s
    if len(i_dyes)>2:
        g1, s1, ph1, m1 = lsf.PhasorTransform_Slow_2D(Spectra,n_harmonic=2)
        GS_Pure[2,:] = g1
    if len(i_dyes)>3:
        GS_Pure[3,:] = s1
    if len(i_dyes)>4:
        g1, s1, ph1, m1 = lsf.PhasorTransform_Slow_2D(Spectra,n_harmonic=3)
        GS_Pure[4,:] = g1
    if len(i_dyes)>5:
        GS_Pure[5,:] = s1
    if len(i_dyes)>6:
        g1, s1, ph1, m1 = lsf.PhasorTransform_Slow_2D(Spectra,n_harmonic=4)
        GS_Pure[6,:] = s1
    if len(i_dyes)>7:
        GS_Pure[7,:] = s1
    return GS_Pure
def GS2IMG(GS,GS_Pure,img,MedFilt_gs):
    f = np.matmul(np.linalg.inv(GS_Pure),GS)
    IMG = np.zeros((GS_Pure.shape[0],np.shape(img)[1],np.shape(img)[2]))
    INT = median(np.sum(img,0),MedFilt_gs)
    for i in range(GS_Pure.shape[0]):
        IMG[i,:,:]          = np.reshape(f[i,:], np.shape(img)[1:3])*INT
    IMG[np.isnan(IMG)] = 0
    IMG[np.isinf(IMG)] = 0
    return IMG
def Get_GS(img,GS_Pure,MedFilt_gs):
    GS          = np.ones((GS_Pure.shape[0],np.shape(img)[1]*np.shape(img)[2]))
    g, s, ph, m = lsf.PhasorTransform_Slow_3D(img,n_harmonic=1,axis = 0)
    g[np.isnan(g)] = 0; g[np.isinf(g)] = 0; s[np.isnan(s)] = 0; s[np.isinf(s)] = 0
    g = median(g,MedFilt_gs)
    s = median(s,MedFilt_gs)
    GS[0,:]          = np.ravel(g)
    if np.shape(GS_Pure)[0]>1:
        GS[1,:]          = np.ravel(s)
    if np.shape(GS_Pure)[0]>2:
        g, s, ph, m = lsf.PhasorTransform_Slow_3D(img,n_harmonic=2,axis = 0)
        g[np.isnan(g)] = 0; g[np.isinf(g)] = 0; s[np.isnan(s)] = 0; s[np.isinf(s)] = 0
        GS[2,:]          = np.ravel(g)
        if np.shape(GS_Pure)[0]>3:
            GS[3,:]          = np.ravel(s)
    if np.shape(GS_Pure)[0]>4:
        g, s, ph, m = lsf.PhasorTransform_Slow_3D(img,n_harmonic=3, axis = 0)
        g[np.isnan(g)] = 0; g[np.isinf(g)] = 0; s[np.isnan(s)] = 0; s[np.isinf(s)] = 0
        GS[4,:] = np.ravel(g)
        if np.shape(GS_Pure)[0]>5:
            GS[5,:] = np.ravel(s)
    if np.shape(GS_Pure)[0]>6:
        g, s, ph, m = lsf.PhasorTransform_Slow_3D(img,n_harmonic=4, axis = 0)
        g[np.isnan(g)] = 0; g[np.isinf(g)] = 0; s[np.isnan(s)] = 0; s[np.isinf(s)] = 0
        GS[6,:] = np.ravel(g)
        if np.shape(GS_Pure)[0]>7:
            GS[7,:] = np.ravel(s)
    return GS
def Unmix(img,GS_Pure,MedFilt_gs):
    GS = Get_GS(img,GS_Pure,MedFilt_gs)
    IMG = GS2IMG(GS,GS_Pure,img,MedFilt_gs)
    return IMG, GS
def scramble(mask):
    idx = np.unique(mask[mask>0])
    idx_new = idx.copy()
    random.shuffle(idx)
    mask_new = np.zeros_like(mask)
    for n,i in enumerate(idx):
        mask_new[mask==idx[n]] = idx_new[n]
    return mask_new

#%% Parameters
# Flags

Experiment_Name = 'Test'
FLAG_SAVEFIG                = True     # Save figure, if False displays but doesn't save (faster)
FLAG_BidirectionalTiling    = True     # Bidirectional mode for tiling
#%% Select files
filenames = askopenfilenames(title = "Select spectral files (.czi)",
                             filetypes = (("czi", ".czi"),))

#%% Analysis
for n_img in range(len(filenames)):
    
    print('Loading file #'+str(n_img+1)+'/'+str(len(filenames))+'...',end='')
    czi = czifile.CziFile(filenames[n_img])
    img_spectral = np.squeeze(AICSImage(filenames[n_img]).data)

    N_ch = img_spectral.shape[0]
    metadata = czi.metadata()
    channels_vect = metadata.split('<Channel Id=')[1:N_ch+1]  
    ch_names = [re.search('Name="(.*)">\n', ch).group(1) for ch in channels_vect]
    pos_PMT = np.where([ch=='T-Pmt' for ch in ch_names])[0][0]
    if pos_PMT.is_integer():
        img_PMT = img_spectral[pos_PMT]
        img_spectral = np.delete(img_spectral, pos_PMT, 0)
        channels_vect = np.delete(channels_vect, pos_PMT, 0)
        ch_names = np.delete(ch_names, pos_PMT, 0)
    Detector_wavelengths = [np.int64(ch[:3]) for ch in ch_names]
    ch_colors = [re.search('<Color>(.*)</Color>', ch).group(1) for ch in channels_vect]
    pixel_size_nm = np.float64(re.search('<Distance Id="X">\n          <Value>(.*)</Value>', metadata).group(1))*1e9                 # Computes pixel size
    RGB_ch = [wavelen2rgb(nm) for nm in Detector_wavelengths[:32]]      # Converts nm to RGB
    
    filename_NoExt = '.'.join(filenames[n_img].split('.')[:-1])     # get path, no extension
    filename_tmp = filename_NoExt.split('/')[-1]                  # get file name, no path
    RGB = np.zeros(img_spectral.shape[1:3]+(3,))                    # initialize RGB images
    for n,rgb in enumerate(RGB_ch):                                 # converts spectral image to RGB
        for i_c in range(3):
            RGB[:,:,i_c] = RGB[:,:,i_c] + (img_spectral[n]/2**16*2**8)*rgb[i_c]
    RGB = RGB-np.percentile(RGB,1)
    RGB = RGB/np.percentile(RGB,99.5)
    RGB[RGB<0] = 0
    RGB[RGB>1] = 1
            
    print('Plotting...')                                            # plot output
    fig,ax = plt.subplots(1,2,figsize = (15,7.5),frameon = False, dpi = 100)
    ax[0].imshow(img_PMT, cmap = 'gray')
    ax[1].imshow(RGB)
    for ax1 in np.ravel(ax):
        ax1.set_axis_off()
    plt.tight_layout()
    # ax[0].set_title(filename_tmp, y=1.0, pad=-14,fontsize=16)
    if FLAG_SAVEFIG:
        plt.savefig(filename_NoExt+'.png')
    plt.show()
#%%


