import sys
import numpy as np
import tensorflow as tf
from mdf_preprocessing import im2mdfin2, mult_seg
import mdfgraph as mdf

import time

#INPUT:
# image - an RGB image
# segdata - a dictionary containig different segmentations of the image
# primary key should be a segmantation number and secondary should be
#   'segmap' - containing segmantation map
#   'segments' - a list of all segments
#    fuseweights - weights of each segmentation saliency map for fusion
#OUTPUT: saliency map
def image_to_saliency_map_mdf(image,mean,seg_param_path,fuseweights,sess,s3cnn,sp_in,nn_in,pic_in):
    t_preprocess =0
    eps = sys.float_info.epsilon
    t_seg = time.time()
    segdata = mult_seg(image,seg_param_path)
    t_seg = time.time() - t_seg
    print('segmentation duration: %f' % t_seg)

    salmap_temp = np.zeros(image.shape[0:2])
    t_net = 0
    for i in range(0,segdata.__len__()):
            t_preprocess_temp = time.time()
            temp = np.zeros(image.shape[0:2])
            seg = segdata[str(i)]
            sp,nn,pic = im2mdfin2(image,mean,seg['segmap'],seg['seglist'],seg['neighbour_mat'])
            t_preprocess = t_preprocess+time.time()-t_preprocess_temp
            #sp = np.reshape(np.ravel(mdfin[0:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
            #nn = np.reshape(np.ravel(mdfin[1:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
            #pic = np.reshape(np.ravel(mdfin[2:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
            t_net_temp = time.time()
            labels = np.uint0([])

            prob = np.float32([])

            for j in range(0,np.uint16(1+seg['seglist'].__len__()/mdf.MAX_BATCH_SIZE)):
                if j*mdf.MAX_BATCH_SIZE == seg['seglist'].__len__() :
                    continue
                tensors = [s3cnn.nnout]
                xdim = (227,227,3)
                feed_dict = {sp_in :sp[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE], nn_in : nn[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE], pic_in : pic[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE]}

                with tf.device('/gpu:0'):

                    up = sess.run(tensors, feed_dict=feed_dict)
                    labels_temp = np.uint0(np.argmax(up[0],1))
                    labels = np.concatenate((labels,labels_temp))
                    prob_temp = np.float32(np.max(up[0],1))
                    prob = np.concatenate((prob,prob_temp))
            t_net = t_net+time.time()-t_net_temp
            for j in range(0,seg['seglist'].__len__()):
                if labels[j] == 1:
                    prob[j]=1-prob[j]
                temp = temp+ (prob[j])*(seg['segmap'] == seg['seglist'][j])

            salmap_temp = fuseweights[i]*temp+salmap_temp
    salmap = np.uint8((salmap_temp-np.min(salmap_temp))/(np.max(salmap_temp)-np.min(salmap_temp)+eps)*255)
    return  t_preprocess, t_net,salmap
