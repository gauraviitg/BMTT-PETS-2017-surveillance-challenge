import numpy as np
#import cv2
import sys
import os
import pickle
import shutil
import gc


def data_prep():

    root = "/home/amit/Desktop/vignesh/11_04_backsub2"
    path = os.path.join(root, "")

    dic = {}
    frameno = 0

    for path, subdirs, files in os.walk(root):
        for filename in files:
            frame_name = filename.split('.')[0]
            if ((int(frame_name)>=1056 and int(frame_name)<=1316) or (int(frame_name)>=1748 and int(frame_name)<=1918) ) :
                dic[frame_name] = 0
            #else:
            	#dic[frame_name] = 0
          
            print frameno
            frameno+=1


    with open("/home/amit/Desktop/vignesh/11_04_backsub2.pickle", 'r+') as f:
        pickle.dump(dic, f)

if __name__ == "__main__":
    data_prep()
    gc.collect()
    
