# -*- coding: utf-8 -*-
import mdf_preprocessing as mdp
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")#every image will turn out a batch of ~100
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
"""Path to the CIFAR-10 data directory.""")
