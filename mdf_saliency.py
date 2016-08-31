import sys
import mdfgraph as mdf
import numpy as np
import tensorflow as tf
from mdf_preprocessing import im2mdfin2, mult_seg
from skimage import io as sio
import time

#INPUT:
# image - an RGB image
# segdata - a dictionary containig different segmentations of the image
# primary key should be a segmantation number and secondary should be
#   'segmap' - containing segmantation map
#   'segments' - a list of all segments
#    fuseweights - weights of each segmentation saliency map for fusion
#OUTPUT: saliency map
def image_to_saliency_map_mdf(image,mean,seg_param_path,fuseweights):

    segdata = mult_seg(image,seg_param_path)
    salmap = np.zeros(image.shape[0:2])
    s3cnn = mdf.S3CNN()
    xdim = (227,227,3)
    eps = sys.float_info.epsilon
    with tf.Session() as sess:
        sp_in = tf.placeholder(tf.float32, (None,) + xdim)
        nn_in = tf.placeholder(tf.float32, (None,) + xdim)
        pic_in = tf.placeholder(tf.float32, (None,) + xdim)

        with tf.name_scope("content_s3cnn"):
            s3cnn.mdf_full(sp_in,nn_in,pic_in, debug=True)
        print('Finished building Network.')
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(0,segdata.__len__()):
            temp = np.zeros(image.shape[0:2])
            seg = segdata[str(i)]
            mdfin = im2mdfin2(image,mean,seg['segmap'],seg['seglist'])
            sp = np.reshape(np.ravel(mdfin[0:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
            nn = np.reshape(np.ravel(mdfin[1:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
            pic = np.reshape(np.ravel(mdfin[2:mdfin.__len__():3]),[np.uint16(mdfin.__len__()/3),227,227,3])
            labels = np.uint0([])

            prob = np.float32([])

            for j in range(0,np.uint16(1+seg['seglist'].__len__()/mdf.MAX_BATCH_SIZE)):
                tensors = [s3cnn.nnout]
                xdim = (227,227,3)
                feed_dict = {sp_in :sp[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE], nn_in : nn[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE], pic_in : pic[j*mdf.MAX_BATCH_SIZE:(1+j)*mdf.MAX_BATCH_SIZE]}

                with tf.device('/gpu:0'):

                    up = sess.run(tensors, feed_dict=feed_dict)
                    labels_temp = np.uint0(np.argmax(up[0],1))
                    labels = np.concatenate((labels,labels_temp))
                    prob_temp = np.float32(np.max(up[0],1))
                    prob = np.concatenate((prob,prob_temp))

            for j in range(0,seg['seglist'].__len__()):
                if labels[j] == 1:
                    prob[j]=1-prob[j]
                temp = temp+ (prob[j])*(seg['segmap'] == seg['seglist'][j])

            salmap = fuseweights[i]*temp+salmap
    return np.uint8((salmap-np.min(salmap))/(np.max(salmap)-np.min(salmap)+eps)*255)
