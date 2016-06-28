from skimage import io
import dill
import scipy as sp
import os
import numpy as np
from skimage.segmentation import slic as slic_wrap

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

BOUNDING_BOX_WIDTH = 227
BOUNDING_BOX_HEIGHT = 227
DEPTH = 9


class MDFInRecord():
    SP_Region = np.zeros([BOUNDING_BOX_WIDTH,BOUNDING_BOX_HEIGHT,3],dtype = np.uint8)
    SP_Neighbor = np.zeros([BOUNDING_BOX_WIDTH,BOUNDING_BOX_HEIGHT,3],dtype = np.uint8)
    Pic = np.zeros([BOUNDING_BOX_WIDTH,BOUNDING_BOX_HEIGHT,3],dtype = np.uint8)
    SP_mask= np.zeros([BOUNDING_BOX_WIDTH,BOUNDING_BOX_HEIGHT],dtype = np.uint0)

class MDFInData(object):
    segments = []
    width    = 227
    height   = 227
    NSP      = 0
    depth    = 9

def dirtomdfbatchmsra(dirpath):
    image_ext = 'jpg'
    images = [fn for fn in os.listdir(dirpath) if fn.endswith(image_ext)]
    images.sort()
    gt_ext = 'png'
    gt_maps = [fn for fn in os.listdir(dirpath) if fn.endswith(gt_ext)]
    gt_maps.sort()
    return gt_maps,images

def save_SLIC_segmentations_MSRA(images,in_dir,out_dir,NSP):
    if 'SLIC_Segs' not in os.listdir(out_dir):
        os.mkdir(out_dir+'/SLIC_Segs')
    if str(NSP) not in os.listdir(out_dir+'SLIC_Segs'):
        os.mkdir(out_dir+'/SLIC_Segs/'+str(NSP))
    for fimg in images :
        img = io.imread(in_dir+fimg)
        gt = io.imread(in_dir+fimg[0:-3]+'png')/255
        SLIC_seg = np.uint16(slic_wrap(img,NSP, 10, sigma=1, enforce_connectivity=True))
        saliency = []
        segments = []
        
        segments_temp = np.unique(SLIC_seg)
        for segment in segments_temp:
            sal_temp = calc_saliency_score(segment,SLIC_seg,gt)                
            if sal_temp >= 0 :
                segments.append(segment)
                saliency.append(np.uint0(sal_temp))
        fslic = open(out_dir+'/SLIC_Segs/'+str(NSP)+'/'+fimg[0:-4]+'.slic','wb')
        dill.dump(SLIC_seg,fslic)
        dill.dump(segments,fslic)
        dill.dump(saliency,fslic)
        fslic.close()        

    
#calculates a segment's saliency score - binary label 0 or 1
    #if saliency is undecided(not enough pixels of 1 class )
def calc_saliency_score(segment,slic,gt):
    mask = np.uint0(slic == segment)
    pixels = np.sum(mask)
    sal = -1
    sal_temp = np.sum(mask*gt)/pixels
    sal = -1
    if sal_temp > 0.7 :
        sal = 1
    elif sal_temp < 0.3:
        sal = 0
    return sal

#returning a list of MDF records for each segment containing sets of training examples and input of the following form:
#SP_Region - [227x227x3] bounding box of the segment with the area around it set to mean image values
#SP_Neighbour - [227x227x3] bounding box of the resized segment and it's immediate neighbouring segments
#Pic - [227x227x3] bounding box of the resized image with the segment blackened
#SP_mask - 227x227x3 mask for the segement location in the original image
#saliency - a saliency score for the segment if one can be decided upon.
def im2mdfin(img,mean,segmap,segments):

    result = MDFInData()
    mean_image = sp.misc.imresize(mean,img.shape)
    
    #Superpixel segmentation - to be replaced by other segmentation if necessary
    #SLIC_seg = slic_wrap(img, nsp, 10, sigma=1, enforce_connectivity=True)
    #segments = np.unique(SLIC_seg)
    #numSP = 0
    
    for SPi in range(0,segments.__len__()):
        pair = MDFInRecord()
        curr_sp = segments[SPi]
        sp_mask= np.uint0(segmap == curr_sp)
        indices = np.where((segmap == curr_sp)!=0)
        bb = np.array([[np.min(indices[0]),np.max(indices[0])],[np.min(indices[1]),np.max(indices[1])]])
        #extracting only the superpixel
        seg_img = np.copy(img[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        mean_seg = np.copy(mean_image[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        local_seg = segmap[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]]
        #zeroing area around superpixel
        seg_img[local_seg != curr_sp,:]=0
        mean_seg[local_seg != curr_sp,:]=0
        #num_pixels = np.sum(local_seg == curr_sp)
        seg_img = seg_img-mean_seg
        #GT_label = np.copy(gt[bb[0,0]:bb[0,1],bb[1,0]:bb[1,1]])
        #GT_label[local_seg != curr_sp]=0
        #saliency_score = np.sum(GT_label/255)/num_pixels
        #Saliency score is deemed reliant so we can add it here
        #if saliency_score > 0.7 or saliency_score < 0.3:
        #numSP = numSP+1
        #finding the neighbor segments
        neighbors = np.unique(local_seg)
        #extracting locations of neighbor segments in image
        ix = np.where(np.in1d(segmap.ravel(),neighbors).reshape(segmap.shape))
        #calculating a bounding box over neghbor superpixels
        bb_mid= np.array([[np.min(ix[0]),np.max(ix[0])],[np.min(ix[1]),np.max(ix[1])]])
        #cropping the bounding box - this is the input to the 2nd mini CNN
        bounding_box_second = np.copy(img[bb_mid[0,0]:bb_mid[0,1],bb_mid[1,0]:bb_mid[1,1]])
        #mean subtraction on region B
        bounding_box_second = bounding_box_second - mean_image[bb_mid[0,0]:bb_mid[0,1],bb_mid[1,0]:bb_mid[1,1]]
        #resizing superpixel to net input size
        pair.SP_Region= sp.misc.imresize(seg_img,[227,227,3])
        #resizing neighborhood to net input size
        pair.SP_Neighbor = sp.misc.imresize(bounding_box_second,[227,227,3])
        #picture with segment masked
        picture = np.copy(img)-mean_image
        picture[segmap == curr_sp,:]=0
        pair.Pic = sp.misc.imresize(picture,[227,227,3])
        #pair.saliency = round(saliency_score) 
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



def _generate_image_segments_and_label_batch(images,img_dir,seg_dir,mean_img):
    #choosing 5 images at random
    test_batch = []
    train_batch = []
    for i in range(0,images.__len__()):
        img = io.imread(img_dir+images[i])
        segf = open(seg_dir+images[i][0:-3]+'slic','rb')
        segmap = dill.load(segf)
        segments_l = dill.load(segf)
        sal_l = dill.load(segf)
        segf.close()
        data = im2mdfin(img,mean_img,segmap,segments_l)

        for j in range(0,segments_l.__len__()):
            x = data.segments[j]
            if i <1000:
                test_batch.append([[x.SP_Region,x.SP_Neighbor,x.Pic],sal_l[j]])
            else:
                train_batch.append([[x.SP_Region,x.SP_Neighbor,x.Pic],sal_l[j]])

    return test_batch,train_batch

def dill_file_to_shuffle_batch(file_path):
    f = open(file_path,'rb')
    batch = dill.load(f)
    f.close()
    
#def _generate_image_segments_and_label_batch(image, label, min_queue_examples,
#batch_size, shuffle):


