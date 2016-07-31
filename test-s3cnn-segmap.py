import sys
sys.path.insert(0,'./tensorflow-fcn')
import mdfgraph as mdf
import numpy as np
import tensorflow as tf
from mdf_preprocessing import im2mdfin2
from skimage import io as sio


#INPUT:
# image - an RGB image
# segdata - a dictionary containig different segmentations of the image
# primary key should be a segmantation number and secondary should be
#   'segmap' - containing segmantation map
#   'segments' - a list of all segments
#    fuseweights - weights of each segmentation saliency map for fusion
#OUTPUT: saliency map
def image_to_saliency_map_mdf(image,mean,segdata,fuseweights):
    salmap = np.zeros(image.shape[0:2])
    s3cnn = mdf.S3CNN()
    xdim = (227,227,3)

    with tf.Session() as sess:
        sp_in = tf.placeholder(tf.float32, (None,) + xdim)
        nn_in = tf.placeholder(tf.float32, (None,) + xdim)
        pic_in = tf.placeholder(tf.float32, (None,) + xdim)

        with tf.name_scope("content_s3cnn"):
            s3cnn.build(sp_in,nn_in,pic_in, debug=True)
        print('Finished building Network.')
        init = tf.initialize_all_variables()
        sess.run(tf.initialize_all_variables())

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
                temp = temp+ labels[j]*(seg['segmap'] == j)

            salmap = fuseweights[i]*temp+salmap
    return np.uint8(salmap/segdata.__len__()*255)


mean = np.load("/home/nyarbel/Python_Gaze_Prediction/mean.npy")
image = sio.imread("/home/nyarbel/Python_Gaze_Prediction/0010.jpg")
segdata = np.load("/home/nyarbel/Python_Gaze_Prediction/0010.npy").item()
fuseweights = [0.9432, 0.9042, 0.9337, 0.9392, 0.9278, 0.930, 0.9148, 0.945, 0.8742, 0.9177, 0.8755, 0.8616,0.9298,0.8742,0.9089]

sal_map = image_to_saliency_map_mdf(image,mean,segdata,fuseweights)
sio.imsave('x.jpg',sal_map)

