from os import listdir
from os.path import isfile, join
from os import environ
environ["OPENCV_IO_ENABLE_JASPER"] = "true"
environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import torch
import numpy as np
from torchvision import transforms as T
import os
mypath = r"../../../../ssd/hypersim_sample/compose_noises"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(len(onlyfiles))

mean = 0.0
meansq = 0.0
i=0
for f in onlyfiles:
    if i%1000 == 0:
        print(i)
    depth = os.path.join(mypath,f)
    depth = cv2.imread(depth, -1)
    depth = T.ToTensor()(np.float32(depth))
    mean += depth.mean()
    meansq += (depth**2).mean()
    i+=1

print("mean: ",mean / len(onlyfiles))
print("meansq: ", meansq/ len(onlyfiles))
print("std: ", torch.sqrt(meansq / len(onlyfiles)  - ((mean) /len(onlyfiles))**2 ))