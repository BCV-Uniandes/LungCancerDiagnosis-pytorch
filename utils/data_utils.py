
import time
import numpy as np
import random
import os
import argparse
from joblib import Parallel, delayed
import scipy.misc
import misc_utils as gp
from scipy import ndimage

def create_test_dataset(data_dir, AUG):
    NSL = 5 #Number of slices (lanes) for multi-line network

    files = os.listdir(data_dir)
    files.sort()

    f = np.load(data_dir + files[0])
    """
    f['arr_0'] = data
    f['arr_1'] = scores
    f['arr_2'] = patient name
    f['arr_3'] = mosaic
    """

    data = np.swapaxes(f['arr_0'],0,3)
    data = np.swapaxes(data,1,2)
    data_train = data[:,:,:,0:NSL]

    label_train = np.load('ISBI_train_label.npy')
    scores_train = f['arr_1'][0:NSL]
    data_train = np.expand_dims(data_train, axis=4)
    scores_train = np.expand_dims(scores_train, axis=0)

    idxp = range(0, 59, 2)
    idxi = range(1, 60, 2)
    label_train = label_train[idxp]

    for ff in range(len(files)-1):
        start_time = time.time()
        filename = data_dir + files[ff+1]
        f = np.load(filename)
        data = np.swapaxes(f['arr_0'],0,3)
        data = np.swapaxes(data,0,2)
        scores = f['arr_1']
        data = np.expand_dims(data, axis=4)
        scores = np.expand_dims(scores, axis=0)
        data_train = np.append(data_train, data[:,:,:,0:NSL], axis=4)
        scores_train = np.append(scores_train, scores[:,0:NSL], axis=0)
    return data_train, label_train    

def create_candidate_slices(vol_dir, cand_dir, slid_dir):
    files = os.listdir(cand_dir)
    sz = 12 #size of slice (sz*2,sz*2,sz*2: 24 x 24 x 24)
    for f in range(len(files)):
        start_time = time.time()
        vol = np.load(str(vol_dir + '/' + files[f]))
        centroids = np.load(str(cand_dir + '/' + files[f]))
        szcen = len(centroids)
        if len(np.shape(centroids)) > 1:
            tz, tx, ty = np.shape(vol)
            I = []
            candsx = centroids[:,1]
            candsy = centroids[:,2]
            candsz = centroids[:,0]
            good = np.where(np.logical_and(np.logical_and(candsx > sz , (tx - candsx) > sz) ,
                 np.logical_and(np.logical_and(candsy > sz , (ty - candsy) > sz) ,
                 np.logical_and(candsz > sz , (tz - candsz) > sz))))
            centroids = centroids[good,:]
            centroids = centroids.reshape(np.shape(centroids)[1],np.shape(centroids)[2])
            for k in range(len(centroids)):
                 im = []
                 for l in range(-sz,sz):
                     im1 = vol[int(centroids[k,0]+l),
                              int(centroids[k,1]-sz) : int(sz+centroids[k,1]),
                              int(centroids[k,2]-sz) : int(sz+centroids[k,2])]
                     im.extend([im1])
                 im = np.asarray(im)
                 im = np.swapaxes(im,0,2)
                 I.extend([im])
            slides = np.asarray(I)
            out_name = str(slid_dir + '/' + str(files[f][:-4]))
            np.save(out_name, slides)
            print('slices - Patient: ' + str(f+1) + '/' + str(len(files)) +
                 ' (' + str(round(time.time() - start_time,2)) + 's)')

def candidate_extraction(vol_dir, cand_dir):
    files = os.listdir(vol_dir)
    for f in range(len(files)):
        start_time = time.time()
        pix_resampled = np.load(str(vol_dir + '/' + files[f]))
        # lung extraction
        segmented_lungs_fill = gp.segment_lung_mask(pix_resampled, True)
        extracted_lungs = pix_resampled * segmented_lungs_fill
        minv = np.min(extracted_lungs)
        extracted_lungs = extracted_lungs - minv
        extracted_lungs[extracted_lungs == -minv] = 0
        # filtering
        filtered_vol = ndimage.median_filter(extracted_lungs, 3)
        # opening by reconstruction
        marker = gp.erode(filtered_vol, [3,3,3]) # 3D grey erosion
        op_reconstructed = gp.reconstruct(marker, filtered_vol) # 3D grey reconstruction
        regional_max = gp.regional_maxima(op_reconstructed) # Regional maxima
        # Computed centroids and centroids from annotations
        centroids, nconncomp = gp.centroids_calc(regional_max) # Computed centroids
        np.save(cand_dir + '/' + str(files[f][:-4]), centroids)
        print('cands - Patient: ' + str(f+1) + '/' + str(len(files)) +
          ' (' + str(round(time.time() - start_time,2)) + 's)')

def create_patients_from_dicom(dicom_dir, vols_dir):
    patients = os.listdir(dicom_dir)
    patients.sort()
    for i in range(len(patients)):
        start_time = time.time()
        subdir = os.listdir(str(dicom_dir + patients[i]))
        subdir.sort()
        i_patient = gp.load_scan(dicom_dir + patients[i] + '/' + subdir[1])
        i_patient_pixels = gp.get_pixels_hu(i_patient)
        pix_resampled, spacing = gp.resample(i_patient_pixels, i_patient, [1.26,.6929,.6929]) 
        filename = (vols_dir + patients[i] + '_2000')
        np.save(filename, pix_resampled)
        print('discom2npy - Patient: ' + str(i+1) + '/' + str(len(patients)) +
              ' (' + str(round(time.time() - start_time,2)) + 's)')

