import cv2
import numpy as np 
import pickle

with open("/home/vignesh/Desktop/pets2/11_04_backsub.pickle",'rb') as f1:
    imgdict1 = pickle.load(f1)


with open("/home/vignesh/Desktop/pets2/11_03_backsub.pickle",'rb') as f2:
    imgdict2 = pickle.load(f2) 

bias = 2000

