import mdf_preprocessing as mpl
import numpy as np
import resource as rsc



in_dir = '/home/nyarbel/Python_Gaze_Prediction/'
gt , images = mpl.dirtomdfbatchmsra(in_dir)
images = ['0010.jpg']
out_dir = '/home/nyarbel/Python_Gaze_Prediction/'
param_path = '/home/nyarbel/Python_Gaze_Prediction/seg_para.npy'
mpl.save_fseg_segmentations_MSRA(images,in_dir,out_dir,param_path)

