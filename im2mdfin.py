#this method recieves either an RGB or LAB Image and returns an 227(W)x227(H)x((3(Color Space)x3())xNSP) array with the following structure:
#NSP triplets of RGB arrays arranged as follows: 
#1st RGB - a tight bounding box around a segment where the area around the segment has been set to mean of dataset
#2nd RGB - a bounding box containing the segment and it's neighboring segments
#3rd RGB - the whole image where the segment pixels have been set to mean pixel value over the dataset

#####option - might change SLIC to a different superpixel method#####
import numpy as np
import scipy as sp
import os
from skimage import io
from skimage.segmentation import slic as slic_wrap
#####TODO add the mean pixel value over the data set as a parameter and subtract it
def im2mdfin(img,NSP,mean):
    class MDFInData(object):
        pass
    result = MDFInData()
    mean_image = sp.misc.imresize(mean,img.shape)

    #Superpixel segmentation - to be replaced by other segmentation if necessary
    SLIC_seg = slic_wrap(img, NSP, 10, sigma=1, enforce_connectivity=True)
    segments = np.unique(SLIC_seg)
    numSP = segments.size
    result_mid =[]
    for SPi in segments:
        curr_sp = segments[SPi]
        indices = np.where((SLIC_seg == curr_sp)!=0)
        bb = np.array([[np.min(indices[0]),np.max(indices[0])],[np.min(indices[1]),np.max(indices[1])]])
        #extracting only the superpixel
        seg_img = np.copy(img[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        mean_seg = np.copy(mean_image[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        local_seg = SLIC_seg[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]]
        #zeroing area around superpixel
        seg_img[local_seg != curr_sp,:]=0
        mean_seg[local_seg != curr_sp,:]=0
        #finding the neighbor segments
        neighbors = np.unique(local_seg)
        #extracting locations of neighbor segments in image
        ix = np.where(np.in1d(SLIC_seg.ravel(),neighbors).reshape(SLIC_seg.shape))
        #calculating a bounding box over neghbor superpixels
        bb_mid= np.array([[np.min(ix[0]),np.max(ix[0])],[np.min(ix[1]),np.max(ix[1])]])
        #cropping the bounding box - this is the input to the 2nd mini CNN
        bounding_box_second = np.copy(img[bb_mid[0,0]:bb_mid[0,1],bb_mid[1,0]:bb_mid[1,1]])
        #mean subtraction on region B
        bounding_box_second = bounding_box_second - mean_image[bb_mid[0,0]:bb_mid[0,1],bb_mid[1,0]:bb_mid[1,1]]
        #resizing superpixel to net input size
        superpixel= sp.misc.imresize(seg_img,[227,227,3])
        #resizing neighborhood to net input size
        neighborhood = sp.misc.imresize(bounding_box_second,[227,227,3])
        #picture with segment masked
        picture = np.copy(img)-mean_image
        picture[SLIC_seg == curr_sp,:]=0
        #the final Described triplet of RGB for the segments
        pair = [superpixel,neighborhood,picture]
        result_mid.append(pair)
    result = result_mid
    return result

