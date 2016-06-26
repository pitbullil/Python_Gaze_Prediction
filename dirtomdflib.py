# -*- coding: utf-8 -*-
import os
def dirtomdfbatchmsra(dirpath):
    image_ext = 'jpg'
    images = [fn for fn in os.listdir(dirpath) if fn.endswith(image_ext)]
    images.sort()
    gt_ext = 'png'
    gt_maps = [fn for fn in os.listdir(dirpath) if fn.endswith(gt_ext)]
    gt_maps.sort()
    return gt_maps,images
    