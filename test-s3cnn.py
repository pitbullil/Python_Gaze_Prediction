import sys
import mdfgraph as mdf
import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    batch = np.load('mdf_input/batch0.bin')
    labels = np.uint8(batch[0:batch.__len__():4])
    sp = np.reshape(np.ravel(batch[1:batch.__len__():4]),[np.uint16(batch.__len__()/4),227,227,3])
    nn = np.reshape(np.ravel(batch[2:batch.__len__():4]),[np.uint16(batch.__len__()/4),227,227,3])
    pic = np.reshape(np.ravel(batch[3:batch.__len__():4]),[np.uint16(batch.__len__()/4),227,227,3])
    xdim = sp.shape[1:]
    sp_in = tf.placeholder(tf.float32, (None,) + xdim)
    nn_in = tf.placeholder(tf.float32, (None,) + xdim)
    pic_in = tf.placeholder(tf.float32, (None,) + xdim)
    s3cnn = mdf.S3CNN()
    with tf.name_scope("content_s3cnn"):
        s3cnn.inference(sp_in,nn_in,pic_in, debug=True)
    print('Finished building Network.')
    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())
    print('Running the Network')
    tensors = [s3cnn.feat]
    x = []
    for i in range(0,10):
        feed_dict = {sp_in :sp[0:100], nn_in : nn[0:100], pic_in : pic[0:100]}

        with tf.device('/gpu:0'):
            up = sess.run(tensors, feed_dict=feed_dict)
            x.append(up)
        print('Finished Running')

    #down_color = utils.color_image(down[0])
    #up_color = utils.color_image(up[0])
    #scp.misc.imsave('fcn16_downsampled.png', down_color)
    #scp.misc.imsave('fcn16_upsampled.png', up_color)


