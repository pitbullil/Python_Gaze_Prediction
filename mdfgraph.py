# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
from math import ceil
import sys
import numpy as np
import tensorflow as tf
import mdf_preprocessing as mdp
sys.path.insert(0,'./tensorflow-fcn')


class S3CNN:
    
    #provide a path to the weights file for each layer
    
    #loading initial weights
    def __init__(self,s3cnn_npy_path=None):
        if s3cnn_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "s3cnn_vgg.npy")
            print(path)
            s3cnn_npy_path = path
            
        self.data_dict = np.load(s3cnn_npy_path, encoding='latin1').item()
        self.wd = 5e-4
        print("npy file loaded")
        
    def build(self, sp,nn,pic, train=False,random_init_fc8=False,debug=False):
        
        self.sp_out = self.__stream(sp,"sp_")
        self.nn_out = self.__stream(nn,"nn_")
        self.pic_out = self.__stream(pic,"pic_")
        
        self.feat = tf.concat(1,[sp_out,nn_out,pic_out])
        self.nn1 = self._fc_layer(self.feat, "nn1")
        self.nn2 = self._fc_layer(self.nn1, "nn2")
        self.scor_fr = self._fc_layer(self.nn2, "score_fr")
        self.pred = tf.argmax(self.score_fr, dimension=3)


    def __stream(self,rgb,name , train=False, num_classes=2,debug=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR

        with tf.name_scope('Processing'):

            red, green, blue = tf.split(3, 3, rgb)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(3, [
                blue,
                green,
                red,
            ])

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)
        
        if name == 'sp_':
            self.sp_conv1_1 = self._conv_layer(bgr, name+"conv1_1")        
            self.sp_conv1_2 = self._conv_layer(self.sp_conv1_1, name+"conv1_2")
            self.sp_pool1 = self._max_pool(self.sp_conv1_2, name+'pool1', debug)
            
            self.sp_conv2_1 = self._conv_layer(self.sp_pool1, name+"conv2_1")
            self.sp_conv2_2 = self._conv_layer(self.sp_conv2_1, name+"conv2_2")
            self.sp_pool2 = self._max_pool(self.sp_conv2_2, name+'pool2', debug)
            self.sp_conv3_1 = self._conv_layer(self.sp_pool2, name+"conv3_1")
            self.sp_conv3_2 = self._conv_layer(self.sp_conv3_1,name+ "conv3_2")
            self.sp_conv3_2 = self._conv_layer(self.sp_conv3_2, name+"conv3_3")
            self.sp_pool3 = self._max_pool(self.sp_conv3_2, name+'pool3', debug)
            
            self.sp_conv4_1 = self._conv_layer(self.sp_pool3, name+"conv4_1")
            self.sp_conv4_2 = self._conv_layer(self.sp_conv4_1, name+"conv4_2")
            self.sp_conv4_3 = self._conv_layer(self.sp_conv4_2, name+"conv4_3")
            self.sp_pool4 = self._max_pool(self.sp_conv4_3, name+'pool4', debug)
            
            self.sp_conv5_1 = self._conv_layer(self.sp_pool4, name+"conv5_1")
            self.sp_conv5_2 = self._conv_layer(self.sp_conv5_1, name+"conv5_2")
            self.sp_conv5_3 = self._conv_layer(self.sp_conv5_2, name+"conv5_3")
            self.sp_pool5 = self._max_pool(self.sp_conv5_3, name+'pool5', debug)
            self.sp_fc6 = self._fc_layer(self.sp_pool5, name+"fc6")
            
            if train:
                self.sp_fc6 = tf.nn.dropout(self.sp_fc6, 0.5)
                
            self.sp_fc7 = self._fc_layer(self.sp_fc6, name+"fc7")
            if train:
                self.sp_fc7 = tf.nn.dropout(self.sp_fc7, 0.5)
                #returning input to S3-CNN
            return self.sp_fc7
            
        if name == 'nn_':
            self.nn_conv1_1 = self._conv_layer(bgr, name+"conv1_1")        
            self.nn_conv1_2 = self._conv_layer(self.nn_conv1_1, name+"conv1_2")
            self.nn_pool1 = self._max_pool(self.nn_conv1_2, name+'pool1', debug)
            
            self.nn_conv2_1 = self._conv_layer(self.nn_pool1, name+"conv2_1")
            self.nn_conv2_2 = self._conv_layer(self.nn_conv2_1, name+"conv2_2")
            self.nn_pool2 = self._max_pool(self.nn_conv2_2,name+ 'pool2', debug)
            self.nn_conv3_1 = self._conv_layer(self.nn_pool2, name+"conv3_1")
            self.nn_conv3_2 = self._conv_layer(self.nn_conv3_1,name+ "conv3_2")
            self.nn_conv3_2 = self._conv_layer(self.nn_conv3_2, name+"conv3_3")
            self.nn_pool3 = self._max_pool(self.nn_conv3_2, name+'pool3', debug)
            
            self.nn_conv4_1 = self._conv_layer(self.nn_pool3, name+"conv4_1")
            self.nn_conv4_2 = self._conv_layer(self.nn_conv4_1, name+"conv4_2")
            self.nn_conv4_3 = self._conv_layer(self.nn_conv4_2, name+"conv4_3")
            self.nn_pool4 = self._max_pool(self.nn_conv4_3, name+'pool4', debug)
            
            self.nn_conv5_1 = self._conv_layer(self.nn_pool4, name+"conv5_1")
            self.nn_conv5_2 = self._conv_layer(self.nn_conv5_1, name+"conv5_2")
            self.nn_conv5_3 = self._conv_layer(self.nn_conv5_2, name+"conv5_3")
            self.nn_pool5 = self._max_pool(self.nn_conv5_3, name+'pool5', debug)
            self.nn_fc6 = self._fc_layer(self.nn_pool5, name+"fc6")
            
            if train:
                self.nn_fc6 = tf.nn.dropout(self.nn_fc6, 0.5)
                
            self.nn_fc7 = self._fc_layer(self.nn_fc6, name+"fc7")
            if train:
                self.nn_fc7 = tf.nn.dropout(self.nn_fc7, 0.5)
                #returning input to S3-CNN
            return self.nn_fc7
        
        if name == 'pic_':
            self.pic_conv1_1 = self._conv_layer(bgr, name+"conv1_1")        
            self.pic_conv1_2 = self._conv_layer(self.pic_conv1_1, name+"conv1_2")
            self.pic_pool1 = self._max_pool(self.pic_conv1_2, name+'pool1', debug)
            
            self.pic_conv2_1 = self._conv_layer(self.pic_pool1, name+"conv2_1")
            self.pic_conv2_2 = self._conv_layer(self.pic_conv2_1, name+"conv2_2")
            self.pic_pool2 = self._max_pool(self.pic_conv2_2, name+'pool2', debug)
            self.pic_conv3_1 = self._conv_layer(self.pic_pool2, name+"conv3_1")
            self.pic_conv3_2 = self._conv_layer(self.pic_conv3_1,name+ "conv3_2")
            self.pic_conv3_2 = self._conv_layer(self.pic_conv3_2, name+"conv3_3")
            self.pic_pool3 = self._max_pool(self.pic_conv3_2, name+'pool3', debug)
            
            self.pic_conv4_1 = self._conv_layer(self.pic_pool3, name+"conv4_1")
            self.pic_conv4_2 = self._conv_layer(self.pic_conv4_1, name+"conv4_2")
            self.pic_conv4_3 = self._conv_layer(self.pic_conv4_2, name+"conv4_3")
            self.pic_pool4 = self._max_pool(self.pic_conv4_3, name+'pool4', debug)
            
            self.pic_conv5_1 = self._conv_layer(self.pic_pool4, name+"conv5_1")
            self.pic_conv5_2 = self._conv_layer(self.pic_conv5_1, name+"conv5_2")
            self.pic_conv5_3 = self._conv_layer(self.pic_conv5_2, name+"conv5_3")
            self.pic_pool5 = self._max_pool(self.pic_conv5_3, name+'pool5', debug)
            self.pic_fc6 = self._fc_layer(self.pic_pool5, name+"fc6")
            
            if train:
                self.pic_fc6 = tf.nn.dropout(self.pic_fc6, 0.5)
                
            self.pic_fc7 = self._fc_layer(self.pic_fc6, name+"fc7")
            if train:
                self.pic_fc7 = tf.nn.dropout(self.pic_fc7, 0.5)
                #returning input to S3-CNN
            return self.pic_fc7
    
    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'nn1':
                filt = self.get_fc_weight_reshape(name, [1, 1, 12289, 300])
            elif name == 'nn2':
                filt = self.get_fc_weight_reshape(name, [1, 1, 300, 301])

            elif name == 'score_fr':
                name = 'nnout'  # Name of score_fr layer in MDF Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 301, 2],
                                                  num_classes=num_classes)
            elif name == 'fc7':
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
                
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.pack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            deconv.set_shape([None, None, None, 2])
            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        return tf.get_variable(name="biases", initializer=init, shape=shape)

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init, shape=shape)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
