import numpy as np
from skimage import io as sio
import time
import tensorflow as tf
from mdf_saliency import image_to_saliency_map_mdf
import mdfgraph as mdf
import mdf_preprocessing as mpl


mean = np.load("/home/nyarbel/Python_Gaze_Prediction/mean.npy")
image = sio.imread('/home/nyarbel/Python_Gaze_Prediction/gerbi.jpg')

#image = sio.imread('/home/nyarbel/Python_Gaze_Prediction/MSRA10K_Imgs_GT/Imgs/75.jpg')

seg_param_path = '/home/nyarbel/Python_Gaze_Prediction/seg_para.npy'
fuseweights = [0.9432, 0.9042, 0.9337, 0.9392, 0.9278, 0.930, 0.9148, 0.945, 0.8742, 0.9177, 0.8755, 0.8616,0.9298,0.8742,0.9089]

with tf.Session() as sess:
    start_time = time.time()
    s3cnn , sp_in,nn_in,pic_in= mdf.build_graph(sess)
    t_preprocess, t_net, sal_map = image_to_saliency_map_mdf(image,mean,seg_param_path,fuseweights,sess,s3cnn,sp_in,nn_in,pic_in)
    duration = time.time() - start_time
    print('preprocess duration: %f' % t_preprocess)
    print('net run duration: %f' % t_net)

    print(duration)
    sess.close()
sio.imshow(sal_map)
sio.show()
sio.imsave('gerbi-seg.jpg',sal_map)

