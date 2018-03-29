import numpy as np
import cv2
import sys
import os
import pickle
import shutil
import gc


def data_prep():

    root = "/media/vignesh/Seagate Backup Plus Drive/Research work/pets-2014 arena database/11_04/TRK_RGB_1"
    path = os.path.join(root, "")

    dic = {}
    frameno = 0

    for path, subdirs, files in os.walk(root):
        for filename in files:
            frame_name = filename.split('.')[0]
            if ((int(frame_name)>1377185170222) and (int(frame_name)<1377185175622)) :
                dic[frame_name] = 1
            else:
            	dic[frame_name] = 0
          
            print frameno
            frameno+=1


    with open("/home/vignesh/Desktop/pets/data.pickle", 'r+') as f:
        pickle.dump(dic, f)

if __name__ == "__main__":
    data_prep()
    gc.collect()
    
