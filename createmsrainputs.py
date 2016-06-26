import dirtomdflib as dmdf
import mdfin
from skimage import io
import dill
import scipy as sp
import os

def msradirtomdfin(dir_path,NSP):
    [gt_maps,images] =  dmdf.dirtomdfbatchmsra(dir_path)
 #   mean_image = sp.zeros([227,227,3],dtype = sp.uint64)
 #   for i in range(0,images.__len__()):#(images.__len__()-1)):
 #       mean_image = mean_image + sp.misc.imresize(io.imread(dir_path+images[i]),[227,227,3])
 #   mean_image = mean_image/images.__len__()
 #   sp.misc.imsave(os.getcwd()+'/mean_image.jpg',mean_image)
    mean_image = io.imread('mean_image.jpg')
    out_dir = './mdfinputs/'
    for i in range(0, (images.__len__())):
        img = io.imread(dir_path+images[i])
        gt = io.imread(dir_path+gt_maps[i])
        record = mdfin.im2mdfin(img,NSP,mean_image,gt)
        outfile = out_dir+images[i][0:-4]+'mdf.out'
        filehandle = open(outfile,'wb')
        dill.dump(record,filehandle)
        filehandle.close()
    
        
        

    
    