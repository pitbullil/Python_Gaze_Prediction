from skimage import io
import dill
import scipy as sp
import os
import numpy as np
from skimage.segmentation import slic as slic_wrap

class MDFInRecord():
    SP_Region = np.zeros([227,227,3],dtype = np.uint8)
    SP_Neighbour = np.zeros([227,227,3],dtype = np.uint8)
    Pic = np.zeros([227,227,3],dtype = np.uint8)
    SP_mask= np.zeros([227,227],dtype = np.uint8)
    saliency = 0

class MDFInData(object):
    segments = []
    width    = 227
    height   = 227
    NSP      = 0
    depth    = 9

#returning a list of MDF records for each segment containing sets of training examples and input of the following form:
#SP_Region - [227x227x3] bounding box of the segment with the area around it set to mean image values
#SP_Neighbour - [227x227x3] bounding box of the resized segment and it's immediate neighbouring segments
#Pic - [227x227x3] bounding box of the resized image with the segment blackened
#SP_mask - 227x227x3 mask for the segement location in the original image
#saliency - a saliency score for the segment if one can be decided upon.
def im2mdfin(img,nsp,mean,gt):

    result = MDFInData()
    mean_image = sp.misc.imresize(mean,img.shape)
    
    #Superpixel segmentation - to be replaced by other segmentation if necessary
    SLIC_seg = slic_wrap(img, nsp, 10, sigma=1, enforce_connectivity=True)
    segments = np.unique(SLIC_seg)
    numSP = 0
    
    for SPi in segments:
        pair = MDFInRecord()
        curr_sp = segments[SPi]
        sp_mask = np.zeros(img.shape,dtype = np.uint8)
        sp_mask[SLIC_seg == curr_sp] = 1
        indices = np.where((SLIC_seg == curr_sp)!=0)
        bb = np.array([[np.min(indices[0]),np.max(indices[0])],[np.min(indices[1]),np.max(indices[1])]])
        #extracting only the superpixel
        seg_img = np.copy(img[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        mean_seg = np.copy(mean_image[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        local_seg = SLIC_seg[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]]
        #zeroing area around superpixel
        seg_img[local_seg != curr_sp,:]=0
        mean_seg[local_seg != curr_sp,:]=0
        num_pixels = np.sum(local_seg == curr_sp)
        seg_img = seg_img-mean_seg
        GT_label = np.copy(gt[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        GT_label[local_seg != curr_sp]=0
        saliency_score = np.sum(GT_label/255)/num_pixels
        #Saliency score is deemed reliant so we can add it here
        if saliency_score > 0.7 or saliency_score < 0.3:
            numSP = numSP+1
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
            pair.SP_Region= sp.misc.imresize(seg_img,[227,227,3])
            #resizing neighborhood to net input size
            pair.SP_Neighbour = sp.misc.imresize(bounding_box_second,[227,227,3])
            #picture with segment masked
            picture = np.copy(img)-mean_image
            picture[SLIC_seg == curr_sp,:]=0
            pair.Pic = sp.misc.imresize(picture,[227,227,3])
            pair.saliency = round(saliency_score) 
            pair.SP_mask = sp.misc.imresize(sp_mask,[227,227,3])
            result.segments.append(pair)
    return result


def msradirtomdfin(dir_path,NSP):
    #extracting names of images in dataset
    [gt_maps,images] =  dirtomdfbatchmsra(dir_path)
    #creating mean image of dataset
 #   mean_image = sp.zeros([227,227,3],dtype = sp.uint64)
 #   for i in range(0,images.__len__()):#(images.__len__()-1)):
 #       mean_image = mean_image + sp.misc.imresize(io.imread(dir_path+images[i]),[227,227,3])
 #   mean_image = mean_image/images.__len__()
 #   sp.misc.imsave(os.getcwd()+'/mean_image.jpg',mean_image)
    mean_image = io.imread('mean_image.jpg')
    out_dir = './mdfinputs/'
    for i in range(0, (images.__len__())):
        img = io.imread(dir_path+images[i])
        gt = io.imread(dir_path+gt_maps[i])
        record = im2mdfin(img,NSP,mean_image,gt)
        outfile = out_dir+images[i][0:-4]+'mdf.out'
        filehandle = open(outfile,'wb')
        dill.dump(record,filehandle)
        filehandle.close()
# -*- coding: utf-8 -*-

def dirtomdfbatchmsra(dirpath):
    image_ext = 'jpg'
    images = [fn for fn in os.listdir(dirpath) if fn.endswith(image_ext)]
    images.sort()
    gt_ext = 'png'
    gt_maps = [fn for fn in os.listdir(dirpath) if fn.endswith(gt_ext)]
    gt_maps.sort()
    return gt_maps,images
