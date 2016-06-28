# -*- coding: utf-8 -*-
from skimage import io
import dill
import mdf_preprocessing as mpl

img_dir = '/home/user/Python_Gaze_Prediction/MSRA10K_Imgs_GT/Imgs/'

gt , images = mpl.dirtomdfbatchmsra(img_dir)
seg_dir = '/home/user/Python_Gaze_Prediction/SLIC_Segs/300/'
mean_img = io.imread('mean_image.jpg')
test_batch,train_batch = mpl._generate_image_segments_and_label_batch(images,img_dir,seg_dir,mean_img)
f = open('trainy','wb')
dill.dump(train_batch,f)
f.close()
f = open('testy','wb')
dill.dump(test_batch,f)
f.close()