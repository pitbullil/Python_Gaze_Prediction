import numpy as np
from skimage import io as sio
import time
import tensorflow as tf
from mdf_saliency import image_to_saliency_map_mdf
import mdfgraph as mdf
import mdf_preprocessing as mpl

img_dir = '/home/nyarbel/Python_Gaze_Prediction/MSRA10K_Imgs_GT/Imgs/'
gt , images = mpl.dirtomdfbatchmsra(img_dir)
mean = np.load("/home/nyarbel/Python_Gaze_Prediction/mean.npy")

seg_param_path = '/home/nyarbel/Python_Gaze_Prediction/seg_para.npy'
fuseweights = [0.9432, 0.9042, 0.9337, 0.9392, 0.9278, 0.930, 0.9148, 0.945, 0.8742, 0.9177, 0.8755, 0.8616,0.9298,0.8742,0.9089]
out_path = '/home/nyarbel/Python_Gaze_Prediction/MSRA10K_MDF_MAPS/'
out = {}
im_list = [81,84,186,190,195,228,229,234,250,261,282,310,319,329,330,337,352,385,420,442,445,470,472,523,537,542,571,
        589,598,621,622,676,738,750,753,754,757,771,806,815,852,900,919,944,946,1049,1065,1079,1080,1082,1109,1128,1148,1150,1159,1181,1193,1237,1240,1263,1280
        ,1269,1276,1286,1389,1399,1442,1446,1458,1469,1513,1578,1595,1626,1636,1643,1678,1684,1736,1746,1793,1794,1867,1869,1881,1888,1903,1908,1912,1941,1979
        ,1928,1982,1991,2018,2032,2064,2084]
with tf.Session() as sess:
    s3cnn , sp_in,nn_in,pic_in= mdf.build_graph(sess)
    j = 2084

    for ind in im_list:#images[j:]:
        #81-101770.jpg#84-101817.jpg#186-103953.jpg#190#195#228#229#234#250#261#282#310#319#329#330#337#352#385#420#442#445#470#472#523#537#542#571
        #589#598#621#622#676#738#750#753#754#757#771#806#815#852#900#919#944#946#1049#1065#1079#1080#1082#1109#1128#1148#1150#1159#1181#1193#1237#1240#1263#1280
        #1269#1276#1286#1389#1399#1442#1446#1458#1469#1513#1578#1595#1626#1636#1643#1678#1684#1736#1746#1793#1794#1867#1869#1881#1888#1903#1908#1912#1941#1979
        #1928#1982#1991#2018#2032#2064#2084
        print(j)
        j=j+1
        #print(img)
        start_time = time.time()
        image = sio.imread(img_dir+images[ind])
        t_preprocess, t_net, sal_map = image_to_saliency_map_mdf(image,mean,seg_param_path,fuseweights,sess,s3cnn,sp_in,nn_in,pic_in)
        duration = time.time() - start_time
        print('preprocess duration: %f' % t_preprocess)
        print('net run duration: %f' % t_net)
        print('mdf total duration %f' % duration)
        out['t_preprocess']=t_preprocess
        out['t_net']=t_net
        out['t_total']=duration
        out['sal_map']=sal_map
        np.save(out_path+images[ind][0:-4],out)
sio.imshow(sal_map)
sio.show()
sio.imsave('x.jpg',sal_map)
