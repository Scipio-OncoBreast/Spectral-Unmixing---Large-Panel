#%% Imports and function definitions
'''
This script is used to analyze spectral data from uptake experiments.
The data format is CZI, multiple Scenes (dyes) are expected as a function of time, single spectral images without tiling
'''
import numpy.typing as npt
import LS_Functions as lsf
import aicspylibczi as czi
import numpy as np
import re
import shutil
import os

from skimage.filters import threshold_otsu, gaussian
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt
# %%
ch_vect = np.arange(2,35)
sigma_smooth = 1
Time_step_minutes = 10
# %%
filename = askopenfilename(
    title="Select CZI file",
    filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
)
filename_dark = askopenfilename(
    title="Select DARK CZI file",
    filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
)
Czi = czi.CziFile(filename_dark)
Dark_spectrum = np.squeeze(Czi.read_image()[0])[ch_vect].mean(axis=(1,2)).astype(float)
# %%
# Create a temporary directory
print("Copying file locally to temporary directory")
temp_dir = 'temp'
if not(os.path.exists(temp_dir)):
   os.mkdir(temp_dir)
# Copy the selected file to the temp directory
local_filename = os.path.join(temp_dir, os.path.basename(filename))
shutil.copy2(filename, local_filename)
# %%
print("Loading local file and selecting channels...")
import czifile
Czi = czifile.CziFile(local_filename)
metadata = Czi.metadata()
channels_vect = metadata.split('<Channel Id=')[1:36]  
ch_names = [re.search('Name="(.*)">\n', ch).group(1) for ch in channels_vect]
ch_wavelengths = np.asarray([int(ch_names[c].split(' ')[0]) for c in ch_vect])

Czi = czi.CziFile(local_filename)
N_Dyes = Czi.get_dims_shape()[0]['S'][1]
# %% Loop through scenes (dyes)
analysis_params = {}

for i_dye in range(N_Dyes):
    print(f"Processing dye {i_dye + 1} of {N_Dyes}")
    dye_name = input("Enter dye name (or press ENTER to skip): ")
    pathsave = os.path.join(os.path.dirname(filename), f"Dye_{i_dye + 1}_") + dye_name
    img = np.squeeze(Czi.read_image(S=i_dye)[0])[:, ch_vect].astype(float)
    T, C, Y, X = img.shape
    t_vect_hours = np.arange(T) * Time_step_minutes / 60
    print("Smoothing...")
    gaussian(img, sigma=(0, 0, sigma_smooth, sigma_smooth),
             preserve_range=True, out=img)

    img_int = img.sum(1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(img_int.max(0), cmap='gray')
    ax.set_title(f"Dye {i_dye + 1} - Select areas without bright spots")
    plt.show()
    x_range = input("Select x-range for intensity profile (e.g.: 100,200), ENTER if ok: ")
    y_range = input("Select y-range for intensity profile (e.g.: 100,200), ENTER if ok: ")
    if x_range == '':
        x_range = '0,' + str(X)
    if y_range == '':
        y_range = '0,' + str(Y)
    x_min, x_max = [int(i) for i in x_range.split(',')]
    y_min, y_max = [int(i) for i in y_range.split(',')]

    # Save analysis parameters for this dye
    analysis_params[f"Dye_{i_dye + 1}_{dye_name}"] = {
        "dye_index": i_dye,
        "dye_name": dye_name,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "sigma_smooth": sigma_smooth,
        "channels": ch_vect.tolist(),
        "time_step_minutes": Time_step_minutes,
        "file": filename,
        "dark_file": filename_dark
    }

    # Draw a red rectangle around the selected analysis area
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(img_int.max(0), cmap='gray')
    ax.set_title(f"Dye {i_dye + 1} - Selected area for analysis")
    linewidth = 2
    rect = plt.Rectangle((x_min + linewidth, y_min + linewidth), x_max - 2 * linewidth - x_min, y_max - y_min - 2 * linewidth,
                         linewidth=linewidth, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.draw()
    plt.savefig(pathsave + "_Selected_Area.png")

    img_int = img_int[:, y_min:y_max, x_min:x_max]
    img = img[:, :, y_min:y_max, x_min:x_max]
    mask_int = img_int > 1.5 * threshold_otsu(img_int[-1])
    Spectra = np.zeros((C, T))
    Int = np.zeros((T))
    for i_t in range(T):
        Spectra[:, i_t] = np.nanmean(img[i_t, :, mask_int[i_t]], axis=0) - Dark_spectrum
        Int[i_t] = np.nanmean(Spectra[:, i_t])
        Spectra[:, i_t] = Spectra[:, i_t] / Spectra[:, i_t].sum()
    tmp = Spectra.copy()
    tmp[tmp < 0] = 0
    g, s, ph, m = lsf.PhasorTransform_Slow_2D(tmp, n_harmonic=1, axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    ax[1, 0].plot(g, s, '-', color='black')
    ax[0, 1].plot(t_vect_hours, Int, '-', color='black')
    ax[1, 1].plot(t_vect_hours, ph, '-', color='black')
    for c in range(Spectra.shape[1]):
        color = plt.get_cmap('nipy_spectral')(c / Spectra.shape[1])
        ax[0, 0].plot(ch_wavelengths, Spectra[:, c], color=color, label=f'T {c}')
        ax[1, 0].plot(g[c], s[c], 'o', markerfacecolor=color, markeredgecolor='black')
        ax[0, 1].plot(t_vect_hours[c], Int[c], 'o', markerfacecolor=color, markeredgecolor='black')
        ax[1, 1].plot(t_vect_hours[c], ph[c], 'o', markerfacecolor=color, markeredgecolor='black')
    ax[0, 0].set_xlabel('Lambda (nm)')
    ax[0, 0].set_ylabel('Intensity')
    ax[0, 0].set_title('Normalized Spectra over time')
    ax[1, 0].set_xlabel('g')
    ax[1, 0].set_ylabel('s')
    ax[1, 0].set_xlim((-1, 1))
    ax[1, 0].set_ylim((-1, 1))
    ax[1, 0].set_title('Spectral phasor over time')
    ax[0, 1].set_xlabel('Time [h]')
    ax[0, 1].set_ylabel('Intensity')
    ax[0, 1].set_title('Intensity vs T')
    ax[1, 1].set_xlabel('Time [h]')
    ax[1, 1].set_ylabel('Phase (rad)')
    ax[1, 1].set_title('Phasor Phase vs T')
    plt.suptitle(f"Dye {i_dye + 1} - " + dye_name + " - Spectra and Intensity")
    plt.tight_layout()
    plt.savefig(pathsave + "_Spectra_and_Intensity.png")
    plt.show()

np.save(os.path.join(os.path.dirname(filename),"analysis_params.npy"), analysis_params)
# %%
# At the end of the script, remove the temp directory
del Czi
print("Deleting temporary directory")
try:
    os.remove(temp_dir)
    print("Successfully deleted!")
except:
    print("Delete foder manually at:", temp_dir)

# %%
