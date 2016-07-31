# -*- coding: utf-8 -*-
from skimage import io as sio
import dill
import mdf_preprocessing as mpl
import numpy as np
import resource as rsc



img_dir = '/home/nyarbel/Python_Gaze_Prediction/MSRA10K_Imgs_GT/Imgs/'
gt , images = mpl.dirtomdfbatchmsra(img_dir)
seg_dir = '/home/user/Python_Gaze_Prediction/SLIC_Segs/300/'
mean_img = io.imread('mean_image.jpg')
batch_size = 20
batches = 30;#np.uint8(images.__len__()/batch_size)
images_np = np.array(images)
soft, hard = rsc.getrlimit(rsc.RLIMIT_AS)
rsc.setrlimit(rsc.RLIMIT_AS, (1024*1024*1024*14, hard))

for i in range(0,1):
    path = './mdf_input/batch'+str(i)+'.bin'
    mpl.write_batch_to_file(path,images,i,batch_size,img_dir,seg_dir,mean_img)
    f = 0
#test_batch = mpl._generate_image_segments_and_label_batch(images_n
# p[shuffle_i[2000*(i+1):2000*(i+2)]],img_dir,seg_dir,mean_img)
 
