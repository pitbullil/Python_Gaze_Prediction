import sys
sys.path.insert(0,'./tensorflow-fcn')
import utils
import mdfgraph as mdf
import numpy as np
import tensorflow as tf
import scipy as scp

with tf.Session() as sess:
    batch = np.load('mdf_input/batch0.bin')
    labels = np.uint8(batch[0:batch.__len__():4])
    sp = np.reshape(np.ravel(batch[1:batch.__len__():4]),[np.uint16(batch.__len__()/4),227,227,3])
    nn = np.reshape(np.ravel(batch[2:batch.__len__():4]),[np.uint16(batch.__len__()/4),227,227,3])
    pic = np.reshape(np.ravel(batch[3:batch.__len__():4]),[np.uint16(batch.__len__()/4),227,227,3])
    sp_in = tf.placeholder("float")
    nn_in = tf.placeholder("float")
    pic_in = tf.placeholder("float")
    feed_dict = {sp_in :sp[0:1], nn_in : nn[0:1], pic_in : pic[0:1]}
    in_1 = [sp[0],nn[0],pic[0]]
    s3cnn = mdf.S3CNN()
    with tf.name_scope("content_s3cnn"):
        s3cnn.build(sp_in,nn_in,pic_in, debug=True)
    print('Finished building Network.')
    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())
    print('Running the Network')
    tensors = [s3cnn.feat]
    with tf.device('/cpu:0'):
        down, up = sess.run(tensors, feed_dict=feed_dict)
    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])
    scp.misc.imsave('fcn16_downsampled.png', down_color)
    scp.misc.imsave('fcn16_upsampled.png', up_color)


