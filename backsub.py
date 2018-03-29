import cv2
import pickle
import numpy as np 
import os



with open("/home/vignesh/Desktop/pets2/08_02_data.pickle",'rb') as f1:
    imgdict = pickle.load(f1)

fgbg = cv2.createBackgroundSubtractorMOG2()
count= 4001
for imgname in sorted(imgdict.keys()):

    filename = "/home/vignesh/Desktop/pets2/08_02/TRK_RGB_2/"+imgname+'.jpg'
    # dest =  "/home//Desktop/backsub2/"+str(count)+'.jpg'
    
    dest =  "/home/vignesh/Desktop/pets2/08_02_backsub/"+str(count)+'.jpg'
    if os.path.exists(filename):

		img = cv2.imread(filename)
		fgmask = fgbg.apply(img)
		cv2.imwrite(dest,fgmask)

		count+=1



