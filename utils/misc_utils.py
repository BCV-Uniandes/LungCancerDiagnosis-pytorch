import numpy as np 
import dicom
import os
import scipy.ndimage
from skimage import measure
from skimage.morphology import erosion, ball, reconstruction

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]): 
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def segment_lung_mask(image, fill_lung_structures=True):
    umb = np.average(image) + np.std(image)*0.1
    binary_image = np.array(image > umb, dtype=np.int8)
    binary_image = 1 - binary_image
    labels = measure.label(binary_image)
    background_label = labels[0,0,0]
    background_label2 = labels[np.shape(image)[0] - 1,np.shape(image)[1] - 1,np.shape(image)[2] - 1]
    #Fill the air around the person
    binary_image[labels == background_label] = 0
    binary_image[labels == background_label2] = 0
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image

def erode(data, radius):
    if len(radius) == 1:
        # define structured element as a ball
        selem = ball(radius[0])
    else:
        # define structured element as a box with input dimensions
        selem = np.ones((radius[0], radius[1], radius[2]), dtype=np.dtype)
    return erosion(data, selem=selem, out=None)

def reconstruct(seed, mask):
    return reconstruction(seed, mask)
    
def regional_maxima(op_reconstructed):
    regional_max = scipy.ndimage.filters.maximum_filter(op_reconstructed, 3) #regional maxima
    regional_max = (op_reconstructed == regional_max) #// convert local max values to binary mask
    marker1 = np.zeros(np.shape(regional_max))
    marker1[0,0,0] = True
    regional_max2 = reconstruct(marker1, regional_max)
    regional_max2 = regional_max - regional_max2
    return regional_max2

def centroids_calc(regional_max):
    labeled_comp = measure.label(regional_max, 8) # Connected components
    nconncomp = np.max(labeled_comp) # Number of connected components
    if nconncomp == 0:
        centroids = []
    else:
        regions_prop = measure.regionprops(labeled_comp)
        centroids = np.zeros((nconncomp - 1, 3))
        for i in range(nconncomp - 1):
                coordss = regions_prop[i+1].coords
                cen_x = np.sum(coordss[:,0])/np.size(coordss,0)
                cen_y = np.sum(coordss[:,1])/np.size(coordss,0)
                cen_z = np.sum(coordss[:,2])/np.size(coordss,0)
                centroids[i,:] = [cen_x, cen_y, cen_z] # Centroids calculated
    return centroids, nconncomp

