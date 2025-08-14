import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    import cupy
except:
    print('LS_Functions: Couldn''t import cupy')
try:
    import cupyx
except:
    print('LS_Functions: Couldn''t import cupyx')
import array
import scipy
import math
import scipy.io as spio
import cv2
import scipy.optimize as opt
from skimage.morphology import disk
import skimage
from lfdfiles import sflim_decode
from gentable import wavelen2rgb

#%% Phasor S-FLIM Functions
def SFLIM_Acquire(dev,L_Acquisition,filename):
        rxBytes = array.array('B', [0]) * L_Acquisition
        print(L_Acquisition)
        dev.read(0x81, rxBytes)
        # Writes file and closes
        f = open(filename, 'w+b')
        binary_format = bytearray(rxBytes)
        f.write(binary_format)
        f.close()       
def SFLIM_PhasorTransform(sflim,n_harmonics,g_CuPy_sp,s_CuPy_sp,g_CuPy_lt,s_CuPy_lt):

    Shape           = sflim.shape;                    # Shape of the S-FLIM Stack
    dim_sp          = 0
    dim_lt          = 1
    
    N_stacks_sp     = 1;                                        # Number of stacks in the spectral dimension tranferred to GPU (max 2 for 256,256,256)
    N_stacks_lt     = 8;                                        # Number of stacks in the spectral dimension tranferred to GPU (max 2 for 256,256,256)
    n_stack_sp      = np.intp(Shape[dim_sp]/N_stacks_sp)                 # Number of iterations needed to tranfer entire matrix on GPU
    n_stack_lt      = np.intp(Shape[dim_lt]/N_stacks_lt)                 # Number of iterations needed to tranfer entire matrix on GPU
    
    # Divide S-FLIM in chunks, save lists
    chunks_sp = [];
    chunks_lt = [];
    for i in range(n_stack_sp):
        chunks_sp.append(sflim[i*N_stacks_sp:i*N_stacks_sp+N_stacks_sp,:,:,:])    
    for i in range(n_stack_lt):
        chunks_lt.append(sflim[:,i*N_stacks_lt:i*N_stacks_lt+N_stacks_lt,:,:])    
        
    # Process Lifetime and Spectral Phasors    
    
    # loop for lifetime phasor - spectral chunks
    for i in range(0,n_stack_sp):
        x_Num2Cu   = cupy.asarray(chunks_sp[i],sflim.dtype)
        y = cupyx.scipy.fftpack.fft(x_Num2Cu,x_Num2Cu.shape[dim_lt],dim_lt,overwrite_x=True);
        c = y[:,0,:,:].real
        c = cupy.tile(c[:,None,:,:],(1,n_harmonics,1,1))
        s_CuPy_lt[i*N_stacks_sp:i*N_stacks_sp+N_stacks_sp,:,:,:] = y[:,1:n_harmonics+1,:,:].real/c
        g_CuPy_lt[i*N_stacks_sp:i*N_stacks_sp+N_stacks_sp,:,:,:] = y[:,1:n_harmonics+1,:,:].imag/c
    g_lt   = cupy.asnumpy(g_CuPy_lt)
    s_lt   = cupy.asnumpy(s_CuPy_lt)
    
    # loop for Spectral phasor - lifetime chunks
    for i in range(0,n_stack_lt):
        x_Num2Cu   = cupy.asarray(chunks_lt[i],sflim.dtype)
        y = cupyx.scipy.fftpack.fft(x_Num2Cu,x_Num2Cu.shape[dim_sp],dim_sp,overwrite_x=True);
        c = y[0,:,:,:].real
        c = cupy.tile(c[None,:,:,:],(n_harmonics,1,1,1))
        s_CuPy_sp[:,i*N_stacks_lt:i*N_stacks_lt+N_stacks_lt,:,:] = y[1:n_harmonics+1,:,:,:].real/c
        g_CuPy_sp[:,i*N_stacks_lt:i*N_stacks_lt+N_stacks_lt,:,:] = y[1:n_harmonics+1,:,:,:].imag/c
    g_sp   = cupy.asnumpy(g_CuPy_sp)
    s_sp   = cupy.asnumpy(s_CuPy_sp)
    
    # Reorder arrays
    g_lt = np.einsum('ijkl->klij',g_lt)
    s_lt = np.einsum('ijkl->klij',s_lt)
    g_sp = np.einsum('ijkl->klji',g_sp)
    s_sp = np.einsum('ijkl->klji',s_sp)
    return g_lt, s_lt, g_sp, s_sp 
def SFLIM_Decode_FromFile(filename,Acquisition,N_Threads = 6):
    PixelTime = math.ceil(Acquisition['DwellTime'] * 256 / 255 * Acquisition['Freq_factor'] * Acquisition['Freq'])
    sflim = np.zeros((Acquisition['N_SpectralChannels'], Acquisition['N_LifetimeBins'], Acquisition['Image_size'], Acquisition['Line_Length']), dtype=np.uint8)
    data = np.fromfile(filename, dtype=np.uint32)
    sflim_decode(data, sflim, pixeltime=PixelTime, maxframes=Acquisition['N_Frames'], numthreads=N_Threads)
    sflim = SFLIM_Shift(sflim,Acquisition)
    return sflim
def SFLIM_Shift(sflim,Acquisition):
    sflim = np.roll(sflim,Acquisition['Shift_Lambda'],axis=0)
    for i in range(Acquisition['N_SpectralChannels']):
        sflim[i,:,:,:] = np.roll(sflim[i,:,:,:],-int(Acquisition['Shift_Time'][i]),axis=0)
    sflim = np.roll(sflim,Acquisition['Shift_X'],axis=2)
    sflim = np.roll(sflim,Acquisition['Shift_Y'],axis=3)
    return sflim
def SFLIM_ComputeCalibration(Path_Dye,Acquisition,Tau,Ch_vect,n_harmonic=np.asarray([1])):
    Acquisition['Shift_Time'] = np.zeros(Acquisition['N_SpectralChannels'])
    sflim = SFLIM_Decode_FromFile(Path_Dye,Acquisition,N_Threads = 6)
    sflim = np.sum(sflim[Ch_vect,:,:,:],(2,3))
    g, s, ph, M = PhasorTransform_Slow_2D(sflim,axis=1,n_harmonic=n_harmonic)
    g_exp, s_exp, ph_exp, M_exp = Phasor_ExpectedPhasorPosition(Tau,freq=Acquisition['Freq_factor'] * Acquisition['Freq'],n_harm=n_harmonic)
    dP = ph-ph_exp
    xM = M/M_exp
    Ch_shift = Acquisition['N_LifetimeBins']/(2*math.pi)*dP
    return dP, xM, Ch_shift

#%% Phasor Functions
def PhasorTransform_Slow_3D(Matrix,axis=0,n_harmonic=1):
    FFT = np.fft.fft(Matrix,axis=axis)
    if axis==0:
        g = np.real(np.conj(FFT[n_harmonic,:,:]))/np.real(FFT[0,:,:])
        s = np.imag(np.conj(FFT[n_harmonic,:,:]))/np.real(FFT[0,:,:])
    if axis==1:
        g = np.real(np.conj(FFT[:,n_harmonic,:]))/np.real(FFT[:,0,:])
        s = np.imag(np.conj(FFT[:,n_harmonic,:]))/np.real(FFT[:,0,:])
    if axis==2:
        g = np.real(np.conj(FFT[:,:,n_harmonic]))/np.real(FFT[:,:,0])
        s = np.imag(np.conj(FFT[:,:,n_harmonic]))/np.real(FFT[:,:,0])
    if axis==3:
        g = np.real(np.conj(FFT[:,:,:,n_harmonic]))/np.real(FFT[:,:,:,0])
        s = np.imag(np.conj(FFT[:,:,:,n_harmonic]))/np.real(FFT[:,:,:,0])
    ph = np.arctan2(s,g)%(2*math.pi)
    M = np.sqrt(g**2+s**2)
    return g, s, ph, M
def PhasorTransform_Slow_2D(Matrix,axis=0,n_harmonic=np.asarray([1])):
    FFT = np.fft.fft(Matrix,axis=axis)
    if axis==0:
        g = np.real(np.conj(FFT[n_harmonic,:]))/np.real(FFT[0,:])
        s = np.imag(np.conj(FFT[n_harmonic,:]))/np.real(FFT[0,:])
    if axis==1:
        g = np.real(np.conj(FFT[:,n_harmonic]))/np.real(FFT[:,0])
        s = np.imag(np.conj(FFT[:,n_harmonic]))/np.real(FFT[:,0])
    ph = np.arctan2(s,g)%(2*math.pi)
    M = np.sqrt(g**2+s**2)
    return g, s, ph, M
def Phasor_ExpectedPhasorPosition(Tau,freq,n_harm=np.asarray([1])):
    omegatau = freq*2*math.pi*Tau*1e-9
    g = 1/(1+omegatau**2)
    s = omegatau/(1+omegatau**2)
    ph = np.arctan2(s,g)%(2*math.pi)
    M = np.sqrt(g**2+s**2)
    return g, s, ph, M
def PhasorTransform_CorrectCoordinates(g,s,dP,xM):
    ph_corr = np.arctan2(s,g)%(2*math.pi)-dP
    M_corr = np.sqrt(g**2+s**2)/xM
    g_corr = M_corr*np.cos(ph_corr)
    s_corr = M_corr*np.sin(ph_corr)
    return g_corr, s_corr, ph_corr, M_corr

def PhasorPlot_Spectral(g,s, Mask='all', MedFilt = 1,FigSize = (25,25),Bins = 256,Range = [[-1,1],[-1,1]],CMap='nipy_spectral'):
    if isinstance(Mask, str):
        Mask = np.full_like(g, True,dtype=bool)
    print('Filtering using skimage.filters.median: disk('+str(MedFilt)+')')
    g = skimage.filters.median(g,disk(MedFilt))*Mask
    s = skimage.filters.median(s,disk(MedFilt))*Mask
    logic = ((np.isnan(g))|(np.isnan(g))|(g==0)|(s==0))==False
    g_hist = g[logic]
    s_hist = s[logic]
    fig,ax = plt.subplots(figsize=FigSize)
    ax.hist2d(g_hist,s_hist,bins= Bins,range = Range,cmap = CMap)
    PhasorPlot = {'logic':logic,
                  'g_hist':g_hist,
                  's_hist':s_hist,
                  'fig':fig,
                  'ax':ax,
                  }
    return PhasorPlot
    
def PhasorUnmixing(Image3D,GS_Pure,Medfilt=1,Norm_option=True):
    N_comp = np.shape(GS_Pure)[0]
    GS = np.zeros((N_comp,np.shape(Image3D)[1]*np.shape(Image3D)[2]))
    g, s, ph, m = PhasorTransform_Slow_3D(Image3D,n_harmonic=1)
    GS[0,:] = np.ravel(scipy.signal.medfilt2d(g,Medfilt))
    GS[1,:] = np.ravel(scipy.signal.medfilt2d(s,Medfilt))
    if np.shape(GS_Pure)[0]>2:
        g, s, ph, m = PhasorTransform_Slow_3D(Image3D,n_harmonic=2)
        GS[2,:] = np.ravel(scipy.signal.medfilt2d(g,Medfilt))
        if np.shape(GS_Pure)[0]>3:
            GS[3,:] = np.ravel(scipy.signal.medfilt2d(s,Medfilt))
    if np.shape(GS_Pure)[0]>4:
        g, s, ph, m = PhasorTransform_Slow_3D(Image3D,n_harmonic=3)
        GS[4,:] = np.ravel(scipy.signal.medfilt2d(g,Medfilt))
        if np.shape(GS_Pure)[0]>5:
            GS[5,:] = np.ravel(scipy.signal.medfilt2d(s,Medfilt))
        
    if Norm_option:
        GS[-1,:] = 1
    f = np.matmul(np.linalg.inv(GS_Pure),GS)
    IMG = np.zeros((N_comp,np.shape(Image3D)[1],np.shape(Image3D)[2]))
    INT = scipy.signal.medfilt2d(np.sum(Image3D,0),Medfilt)
    for i in range(N_comp):
        IMG[i,:,:] = np.reshape(f[i,:], np.shape(Image3D)[1:3])*INT
    return IMG
def PhasorUnmixing_N2V(Image3D,N2V_model,GS_Pure,Medfilt=1,Norm_option=True):
    N_comp = np.shape(GS_Pure)[0]
    GS = np.zeros((N_comp,np.shape(Image3D)[1]*np.shape(Image3D)[2]))
    g, s, ph, m = PhasorTransform_Slow_3D(Image3D,n_harmonic=1)
    g = N2V_model.predict(g,'YX')
    s = N2V_model.predict(s,'YX')
    GS[0,:] = np.ravel(scipy.signal.medfilt2d(g,Medfilt))
    GS[1,:] = np.ravel(scipy.signal.medfilt2d(s,Medfilt))
    if np.shape(GS_Pure)[0]>2:
        g, s, ph, m = PhasorTransform_Slow_3D(Image3D,n_harmonic=2)
        g = N2V_model.predict(g,'YX')
        s = N2V_model.predict(s,'YX')
        GS[2,:] = np.ravel(scipy.signal.medfilt2d(g,Medfilt))
        if np.shape(GS_Pure)[0]>3:
            GS[3,:] = np.ravel(scipy.signal.medfilt2d(s,Medfilt))
    if np.shape(GS_Pure)[0]>4:
        g, s, ph, m = PhasorTransform_Slow_3D(Image3D,n_harmonic=3)
        g = N2V_model.predict(g,'YX')
        s = N2V_model.predict(s,'YX')
        GS[4,:] = np.ravel(scipy.signal.medfilt2d(g,Medfilt))
        if np.shape(GS_Pure)[0]>5:
            GS[5,:] = np.ravel(scipy.signal.medfilt2d(s,Medfilt))
        
    if Norm_option:
        GS[-1,:] = 1
    f = np.matmul(np.linalg.inv(GS_Pure),GS)
    IMG = np.zeros((N_comp,np.shape(Image3D)[1],np.shape(Image3D)[2]))
    INT = scipy.signal.medfilt2d(np.sum(Image3D,0),Medfilt)
    for i in range(N_comp):
        IMG[i,:,:] = np.reshape(f[i,:], np.shape(Image3D)[1:3])*INT
    return IMG
#%% Image Processing Functions
def ColormapPlusInt(img_Int,img_Property,Range,CMap,Norm = 'max'):
    cmap = matplotlib.cm.get_cmap(CMap)    
    img_Property = img_Property-Range[0]
    img_Property = img_Property/(Range[1]-Range[0])
    img_Property[img_Property<0] = 0
    img_Property[img_Property>1] = 1
    if Norm == 'max':
        img_Int = img_Int/np.max(img_Int)
    else:
        img_Int = img_Int/Norm
    Shape = img_Int.shape
    tmp_col = cmap(img_Property)
    output  = tmp_col
    
    for i_x in range(Shape[0]):
        for i_y in range(Shape[1]):
            output[i_x,i_y,:] = tmp_col[i_x,i_y,:]*img_Int[i_x,i_y]
    output[:,:,3] = 1
    return output
def wavelength_to_rgb(wavelength, gamma=1):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))    
def wavelength_to_rgb_vect(vect, gamma=1):
    if isinstance(vect, int)|isinstance(vect, float):
        RGB_vect = wavelength_to_rgb(vect, gamma=gamma)
    else:
        RGB_vect = np.zeros((len(vect),3))
        for i in range(len(vect)):
            RGB = wavelength_to_rgb(vect[i], gamma=gamma)
            RGB_vect[i,0] = RGB[0]
            RGB_vect[i,1] = RGB[1]
            RGB_vect[i,2] = RGB[2]
    return RGB_vect


def SpectralStack2RGB(Stack, Ch_wavelength):
    RGB_vect = wavelength_to_rgb_vect(Ch_wavelength)
    Size = np.shape(Stack)
    RGB_img = np.zeros((Size[1], Size[2], 3))
    for ch in range(Size[0]):
        for col in range(3):
            RGB_img[:, :, col] = RGB_img[:, :, col] + Stack[ch, :, :] * RGB_vect[ch, col]
    return RGB_img
def Binning_Spectral(img, Binning_step=[2,2]):
    Size = img.shape
    if len(Size)>2:
        # img1 = np.zeros((Size[0],int(Size[1]/Binning_step[0]),int(Size[2]/Binning_step[1])))
        img1 = np.zeros((int(Size[1]/Binning_step[0]),int(Size[2]/Binning_step[1]),Size[0]))
        for i_img in range(Size[0]):
            tmp_img = scipy.ndimage.uniform_filter(img[i_img,:,:], Binning_step)
            img1[:,:,i_img] = tmp_img[0:Size[1]:Binning_step[0],0:Size[2]:Binning_step[1]]
    else:            
        img1 = scipy.ndimage.uniform_filter(img, Binning_step)
        img1 = img1[0:Size[0]:Binning_step[0],0:Size[1]:Binning_step[1]]
    return img1
def Binning(img, Binning_step=[2,2]):
    Size = img.shape
    if len(Size)>2:
        # img1 = np.zeros((Size[0],int(Size[1]/Binning_step[0]),int(Size[2]/Binning_step[1])))
        img1 = np.zeros((int(Size[0]/Binning_step[0]),int(Size[1]/Binning_step[1]),Size[2]))
        for i_img in range(Size[2]):
            tmp_img = scipy.ndimage.uniform_filter(img[:,:,i_img], Binning_step)
            img1[:,:,i_img] = tmp_img[0:Size[0]:Binning_step[0],0:Size[1]:Binning_step[1]]
    else:            
        img1 = scipy.ndimage.uniform_filter(img, Binning_step)
        img1 = img1[0:Size[0]:Binning_step[0],0:Size[1]:Binning_step[1]]
    return img1
def cellpose_binning(img,model,channels,Binning_vect, diameter=30, FLAG_spectral = True):
    if FLAG_spectral:
        masks = model.eval(Binning_Spectral(img,Binning_vect), channels=channels, diameter=diameter)
    else:
        masks = model.eval(Binning(img,Binning_vect), channels=channels, diameter=diameter)
            
    masks = masks[0]
    L = np.max(Binning_vect)
    masks1 = scipy.ndimage.zoom(masks, L)
    L_max = np.max(masks1.shape)
    masks_new = np.zeros_like(masks1,dtype=np.uint16)
    kernel = np.ones((L,L))
    for i in range(1,np.max(masks1)+1):
        tmp_erode = cv2.erode(np.uint16(masks1==i), kernel)
        idx = np.where(tmp_erode)
        roi = np.asarray([np.min(idx[0])-L,np.max(idx[0])+L,np.min(idx[1])-L,np.max(idx[1])+L],dtype = int)
        roi[roi<0] = 0
        roi[roi>L_max] = L_max
        masks_new[roi[0]:roi[1],roi[2]:roi[3]] = masks_new[roi[0]:roi[1],roi[2]:roi[3]] + i*cv2.dilate(tmp_erode[roi[0]:roi[1],roi[2]:roi[3]], kernel)
    return masks_new

#%% Plotting Functions
def plot_marker_map(ax,x,y,CMap,Marker='o',MarkerSize=10,MarkerEdgeColor=[0,0,0]):
    MarkerList = list()
    for i in range(len(x)):
        plot_tmp, = ax.plot(x[i],y[i])
        MarkerList.append(plot_tmp)
        MarkerList[i].set_markersize(MarkerSize)
        MarkerList[i].set_marker(Marker)
        MarkerList[i].set_markerfacecolor((CMap[i,:]))
        MarkerList[i].set_markeredgecolor(MarkerEdgeColor)
    return MarkerList
def plot_marker_map_update(MarkerList,x,y,CMap,Marker='o',MarkerSize=10,MarkerEdgeColor=[0,0,0]):
    for i in range(len(x)):
        MarkerList[i].set_xdata(x[i])
        MarkerList[i].set_ydata(y[i])
        MarkerList[i].set_markersize(MarkerSize)
        MarkerList[i].set_marker(Marker)
        if MarkerEdgeColor==None:
            MarkerList[i].set_markeredgecolor((CMap[i,:]))
        else:
            MarkerList[i].set_markeredgecolor(MarkerEdgeColor)
        MarkerList[i].set_markerfacecolor((CMap[i,:]))
    return MarkerList       
def plot_line_map(ax,x,Y,CMap,Line='-',LineWidth=1):
    LinesList = list()
    if len(np.shape(Y))==1:
        plot_tmp, = ax.plot(x,Y)
        LinesList.append(plot_tmp)
        LinesList[0].set_linewidth(LineWidth)
        LinesList[0].set_linestyle(Line)
        LinesList[0].set_color((CMap))
    else:
        for i in range(np.shape(Y)[1]):
             plot_tmp, = ax.plot(x,Y[:,i])
             LinesList.append(plot_tmp)
             LinesList[i].set_linewidth(LineWidth)
             LinesList[i].set_linestyle(Line)
             LinesList[i].set_color((CMap[i,:]))
    return LinesList
        
#%% Correlation
def ACF2D(img,L_ACF):
    Size = np.shape(img)
    F = np.fft.fft2(img)
    ACF = F*np.conjugate(F)
    G = np.sum(img)**2/Size[0]/Size[1]
    ACF = np.real(np.fft.fftshift(np.fft.ifft2(ACF),axes=(0,1)))/G-1
    ACF = ACF[Size[0]//2-L_ACF:Size[0]//2+L_ACF,Size[1]//2-L_ACF:Size[1]//2+L_ACF]
    # ACF [L_ACF,L_ACF] = ACF [L_ACF,L_ACF+1] # Central value is set to the neighboring value
    return ACF

def CCF2D(img1,img2,L_CCF):
    Size = np.shape(img1)
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    CCF = F1*np.conjugate(F2)
    G = np.sum(img1)*np.sum(img2)/Size[0]/Size[1]
    CCF = np.real(np.fft.fftshift(np.fft.ifft2(CCF),axes=(0,1)))/G-1
    CCF = CCF[Size[0]//2-L_CCF:Size[0]//2+L_CCF,Size[1]//2-L_CCF:Size[1]//2+L_CCF]
    return CCF

#%% Fitting
def twoD_Gaussian(amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Returns a gaussian function with the given parameters"""
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    return lambda x,y: offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
def moments(data):
    """Returns (height, x, y, width_x, width_y,0,0)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0, 0
def fit2Dgaussian(data,Display=0):
    """Returns (height, x, y, width_x, width_y, theta, offset)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(twoD_Gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = opt.leastsq(errorfunction, params)
    if Display==1:
        plt.matshow(data, cmap=plt.cm.jet)
        params = fit2Dgaussian(data)
        fit = twoD_Gaussian(*params)
        plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
        (height, x, y, width_x, width_y, theta, offset) = params
    return p

#%% Miscellanea
def SpectralDetector_A10766(vect=np.arange(32,dtype=int)):
    Spectral_Wavelengths = np.array([404.3, 413.4, 422.4, 431.4, 440.4, 449.3, 458.3, 467.2, 476.2, 485.0, 493.9, 502.7, 511.5, 520.3, 529.0, 537.7, 546.3, 554.9, 563.5, 572.0, 580.4, 588.9, 597.2, 605.5, 613.8, 622.0, 630.1, 638.2, 646.2, 654.1, 662.0, 669.8]);
    Vect = Spectral_Wavelengths[vect]
    return Vect
def tile_stitching_spectral(img, m, n, bidirectional=False, percentage_overlap = 0):
    #image should be (Tile,S,Y,X)
    X = img.shape[-2]
    Y = img.shape[-1]
    dX = int(percentage_overlap/100*X/2)
    dY = int(percentage_overlap/100*Y/2)

    img_rec = np.zeros(np.array([img.shape[1], X-2*dX, Y-2*dY]) * np.array([1, m, n]))

    cnt_slice = 0
    for i in range(m):
        if bidirectional & ((i % 2) != 0):
            j_range = np.flip(np.arange(n))
        else:
            j_range = np.arange(n)
        for j in j_range:
            img_rec[:,i*(X-2*dX):(i+1)*(X-2*dX),j*(Y-2*dY):(j+1)*(Y-2*dY)] = img[cnt_slice,:,dX:X-dX,dY:Y-dY]
            cnt_slice += 1
    return img_rec

def tile_stitching_spectral_4D(img, m, n, bidirectional=False, percentage_overlap = 0):
    X = img.shape[-2]
    Y = img.shape[-1]
    dX = int(percentage_overlap/100*X/2)
    dY = int(percentage_overlap/100*Y/2)

    img_rec = np.zeros(np.array([img.shape[1], img.shape[2], X-2*dX, Y-2*dY]) * np.array([1, 1, m, n]))

    cnt_slice = 0
    for i in range(m):
        if bidirectional & ((i % 2) != 0):
            j_range = np.flip(np.arange(n))
        else:
            j_range = np.arange(n)
        for j in j_range:
            img_rec[:,:,i*(X-2*dX):(i+1)*(X-2*dX),j*(Y-2*dY):(j+1)*(Y-2*dY)] = img[cnt_slice,:,:,dX:X-dX,dY:Y-dY]
            cnt_slice += 1
    return img_rec

#%% From Matlab
def _todict(matobj):
# Load structure variable from Matlab .mat workspace and converts it to dict
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
def loadmat(filename,variablename,PrintKeys = False):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = data[variablename]
    Dict = dict()
    for i in range(len(data)):
        tmp_dict = _todict(data[i])
        Dict[tmp_dict['Name']] = tmp_dict
    if PrintKeys:
        for key, value in Dict.items():
            print(key)
    return Dict


        
        
        
        
        