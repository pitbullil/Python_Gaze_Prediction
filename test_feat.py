import sys
sys.path.insert(0,'./tensorflow-fcn')
import mdfgraph as mdf
import numpy as np
import tensorflow as tf
from mdf_preprocessing import im2mdfin2
from skimage import io as sio
from mdf_preprocessing import trainable_segmentations_from_batch
import time
import matplotlib.pyplot as plt
with tf.Session() as sess:
    s3cnn = mdf.S3CNN()
    feat =tf.placeholder(tf.float32, (None,)+(12288,))
    s3cnn.inference(feat)
    x= np.random.rand(10,12288)
    feed_dict = {feat : x}
    tensor = s3cnn.nnout
    init = tf.initialize_all_variables()
    sess.run(init)

    up = sess.run(tensor, feed_dict=feed_dict)
