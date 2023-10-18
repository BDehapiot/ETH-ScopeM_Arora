#%% Imports -------------------------------------------------------------------

import cv2
import nd2
import time
import numpy as np
import pandas as pd
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from joblib import Parallel, delayed 
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    remove_small_holes, remove_small_objects, 
    binary_dilation, label, white_tophat, disk
    )

#%% Parameters ----------------------------------------------------------------

# Paths
data_path = Path('D:/local_Arora/data')
task1_path = Path(data_path, 'task1')

# Segmentation
sigma = 2
thresh_coeff = 1

# Colocalization
norm = "chn" # "none", "chn", "obj"

#%% Initialize ----------------------------------------------------------------

# Open data
alldata = []
smn1_count, smn2_count, smn1m_count = 0, 0, 0
for path in data_path.iterdir():
    
    if path.suffix == ".nd2":
    
        # Read nd2 file 
        with nd2.ND2File(path) as ndfile:
            img = ndfile.asarray()
            voxel = list(ndfile.voxel_size())
    
        # Format data task1 (see specific rules)
        name = path.stem
        if "3-smn1" in name:        
            cond = "smn1"; smn1_count += 1
            img = img[11,...] 
            alldata.append({
                "name": name,
                "cond": cond,
                "count": smn1_count,
                "voxel": voxel,
                "img": img
                })
 
        if "13-smn2" in name:        
            cond = "smn2"; smn2_count += 1
            img = img[0:2,...]
            alldata.append({
                "name": name,
                "cond": cond,
                "count": smn2_count,
                "voxel": voxel,
                "img": img
                })
            
        if "14-smn1mut" in name:        
            cond = "smn1m"; smn1m_count += 1
            if img.shape[0] == 3:
                img = img[0:2,...]
            elif img.shape[0] == 4:
                img = img[np.r_[0,2],...]
            alldata.append({
                "name": name,
                "cond": cond,
                "count": smn1m_count,
                "voxel": voxel,
                "img": img
                })
                       
#%% Process -------------------------------------------------------------------

start = time.time()
print('Process')

for i, data in enumerate(alldata):

    img = data["img"]
    name = data["name"]
    cond = data["cond"] 
    count = data["count"] 
    voxel = data["voxel"] 

    # Get mask
    c1, c2 = img[0,...], img[1,...]
    gblur = gaussian(np.mean(img, axis=0), sigma=sigma)
    mask = gblur > threshold_otsu(gblur) * thresh_coeff
    
    # Filter mask
    mask = remove_small_holes(mask, area_threshold=256)
    mask = remove_small_objects(mask, min_size=256)
    outlines = binary_dilation(mask) ^ mask
    labels = label(mask)
    
    # Normalization
    if norm == "none":
        c1norm = c1.copy().astype("float32"); c1norm[mask==False] = np.nan
        c2norm = c2.copy().astype("float32"); c2norm[mask==False] = np.nan
    
    if norm == "chn":
        # c1norm = rescale_intensity(c1, out_range=(0,1))
        # c2norm = rescale_intensity(c2, out_range=(0,1))
        c1norm = c1 / np.mean(c1)
        c2norm = c2 / np.mean(c2)
        # c1norm = rescale_intensity(c1 / np.mean(c1), out_range=(0,1))
        # c2norm = rescale_intensity(c2 / np.mean(c2), out_range=(0,1))
        c1norm[mask==False] = np.nan
        c2norm[mask==False] = np.nan
    
    if norm == "obj":
        c1norm = np.full_like(c1, np.nan, dtype="float32")
        c2norm = np.full_like(c2, np.nan, dtype="float32")   
        for lab in np.unique(labels):
            if lab > 0:
                idx = labels == lab
                c1val, c2val= c1[idx], c2[idx]
                # c1norm[idx] = rescale_intensity(c1val, out_range=(0,1))
                # c2norm[idx] = rescale_intensity(c2val, out_range=(0,1))
                c1norm[idx] = c1val / np.mean(c1val)
                c2norm[idx] = c2val / np.mean(c2val)
                
    # Measure objects & display

    objdisplay = np.zeros_like(c1, dtype="float32") 
    objdata = {
        "name": [], "cond": [], "count": [],
        "area": [], "corr": [], 
        "c1norm_crop": [], "c2norm_crop": [],
        }
    
    for prop in regionprops(labels):
        
        lab = prop["label"]
        area = prop["area"]
        y, x = prop["centroid"]
        bbox = prop["bbox"]
        idx = labels == lab
        c1norm_crop = c1norm[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        c2norm_crop = c2norm[bbox[0]:bbox[2], bbox[1]:bbox[3]]      
        corr, p = pearsonr(c1norm[idx], c2norm[idx])
        objdisplay = cv2.putText(
            objdisplay, str(format(corr, '.3f')), (round(x), round(y)),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, (1,1,1), 1, cv2.LINE_AA
            )       

        # Append objdata
        objdata["name"].append(name)
        objdata["cond"].append(cond)
        objdata["count"].append(count)
        objdata["area"].append(area)
        objdata["corr"].append(corr)
        objdata["c1norm_crop"].append(c1norm_crop)
        objdata["c2norm_crop"].append(c2norm_crop)
                
    # Append alldata
    alldata[i]["mask"] = mask
    alldata[i]["outlines"] = outlines
    alldata[i]["labels"] = labels
    alldata[i]["img_norm"] = np.stack((c1norm, c2norm), axis=0)
    alldata[i]["objdata"] = objdata
    alldata[i]["objdisplay"] = objdisplay

# Extract objdata as dataframe
objdata = pd.concat(
    [pd.DataFrame(data["objdata"]) for data in alldata],
    ignore_index=True,
    )

end = time.time()
print(f'  {(end-start):5.3f} s')  

#%% Plots ---------------------------------------------------------------------
 
# Area v.s. Pearson correlation -----------------------------------------------

# Extract data
smn1_area = objdata['area'][objdata['cond'] == "smn1"]
smn1_corr = objdata['corr'][objdata['cond'] == "smn1"]
smn2_area = objdata['area'][objdata['cond'] == "smn2"]
smn2_corr = objdata['corr'][objdata['cond'] == "smn2"]
smn1m_area = objdata['area'][objdata['cond'] == "smn1m"]
smn1m_corr = objdata['corr'][objdata['cond'] == "smn1m"]

# Plot
fig = plt.figure(figsize=(6, 12))
ax = plt.subplot(3, 1, 1)
plt.scatter(smn1_area, smn1_corr)
plt.axhline(0, color='black', linestyle='--', linewidth=0.75)
ax.axis([0, 10000, -1, 1])
plt.title("smn1")
ax = plt.subplot(3, 1, 2)
plt.scatter(smn2_area, smn2_corr)
plt.axhline(0, color='black', linestyle='--', linewidth=0.75)
ax.axis([0, 10000, -1, 1])
plt.title("smn2")
ax = plt.subplot(3, 1, 3)
plt.scatter(smn1m_area, smn1m_corr)
plt.axhline(0, color='black', linestyle='--', linewidth=0.75)
ax.axis([0, 10000, -1, 1])
plt.title("smn1m")

# Pixel to pixel correlation --------------------------------------------------

# Extract data
smn1_c1val, smn1_c2val = [], []
smn2_c1val, smn2_c2val = [], []  
smn1m_c1val, smn1m_c2val = [], []  
for data in alldata:
    
    cond = data["cond"]
    mask = data["mask"]
    c1_norm = data["img_norm"][0,...]
    c2_norm = data["img_norm"][1,...]
    
    if cond == "smn1":
        smn1_c1val.append(c1_norm[mask])
        smn1_c2val.append(c2_norm[mask])
    if cond == "smn2":
        smn2_c1val.append(c1_norm[mask])
        smn2_c2val.append(c2_norm[mask])
    if cond == "smn1m":
        smn1m_c1val.append(c1_norm[mask])
        smn1m_c2val.append(c2_norm[mask])
  
# Format data
smn1_c1val = np.concatenate(smn1_c1val)
smn1_c2val = np.concatenate(smn1_c2val)
smn2_c1val = np.concatenate(smn2_c1val)
smn2_c2val = np.concatenate(smn2_c2val)
smn1m_c1val = np.concatenate(smn1m_c1val)
smn1m_c2val = np.concatenate(smn1m_c2val)
   
# Plot 
vmin, vmax = 0, 500
fig = plt.figure(figsize=(6, 12))

ax = plt.subplot(3, 1, 1)
plt.hist2d(
    smn1_c1val, smn1_c2val, bins=(100, 100), 
    cmap='plasma', vmin=vmin, vmax=vmax
    )
plt.colorbar(label='Density', ax=ax)
plt.title("smn1")

ax = plt.subplot(3, 1, 2)
plt.hist2d(
    smn2_c1val, smn2_c2val, bins=(100, 100), 
    cmap='plasma', vmin=vmin, vmax=vmax
    )
plt.colorbar(label='Density', ax=ax)
plt.title("smn2")

ax = plt.subplot(3, 1, 3)
plt.hist2d(
    smn1m_c1val, smn1m_c2val, bins=(100, 100), 
    cmap='plasma', vmin=vmin, vmax=vmax
    )
plt.colorbar(label='Density', ax=ax)
plt.title("smn1m")

plt.show()
    
#%% Save ----------------------------------------------------------------------

# Setup LUTs
val_range = np.arange(256, dtype='uint8')
lut_gray = np.stack([val_range, val_range, val_range])
lut_green = np.zeros((3, 256), dtype='uint8')
lut_green[1, :] = val_range
lut_magenta= np.zeros((3, 256), dtype='uint8')
lut_magenta[[0,2],:] = np.arange(256, dtype='uint8')

# Save images
for data in alldata:
    
    img_norm = data["img_norm"]
    outlines = data["outlines"]
    objdisplay = data["objdisplay"]
    cond = data["cond"] 
    count = data["count"] 
    voxel = data["voxel"] 
    
    # Add outlines to multichannel images
    outlines = outlines[np.newaxis, :, :].astype('float32') * 65535
    objdisplay = objdisplay[np.newaxis, :, :].astype('float32') * 65535
    img_norm = np.concatenate([img_norm, outlines, objdisplay], axis=0).astype('float32')
    
    # img_norm
    io.imsave(
        Path(data_path, 'task1', f"{cond}_{count:02}_norm.tif"),
        img_norm, check_contrast=False, imagej=True,
        resolution=(1 / voxel[0], 1 / voxel[1]),
        metadata={
            'axes': 'CYX', 
            'mode': 'composite',
            'LUTs': [lut_magenta, lut_green, lut_gray, lut_gray],
            'unit': 'um',
            },
        photometric='minisblack',
        planarconfig='contig'
        )
                                        
#%% Display -------------------------------------------------------------------       

# import napari
# viewer = napari.Viewer()
# viewer.add_image(img)