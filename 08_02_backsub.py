import numpy as np
import cv2
import sys
import os
import pickle
import shutil
import gc


def data_prep():

    root = "/home/vignesh/Desktop/pets2/08_02_backsub"
    path = os.path.join(root, "")

    dic = {}
    frameno = 0

    for path, subdirs, files in os.walk(root):
        for filename in files:
            frame_name = filename.split('.')[0]
            if ((int(frame_name)>4379) and (int(frame_name)<4586)) :
                dic[frame_name] = 1
            else:
            	dic[frame_name] = 0
          
            print frameno
            frameno+=1


    with open("/home/vignesh/Desktop/pets2/08_02_backsub.pickle", 'r+') as f:
        pickle.dump(dic, f)

if __name__ == "__main__":
    data_prep()
    gc.collect()
    
