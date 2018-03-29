import cv2
import numpy as np 
import pickle

with open("/home/vignesh/Desktop/pets2/08_02_data.pickle",'rb') as f1:
    imgdict1 = pickle.load(f1)


with open("/home/vignesh/Desktop/pets2/08_02_backsub.pickle",'rb') as f2:
    imgdict2 = pickle.load(f2)    


key1 = imgdict1.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    
key2 = sorted(keylist1)
keylist2=[]
for i in key2:
    newkey2 = str(i)
    keylist2.append(newkey2)  

imgdict1_keys = keylist2
# print(imgdict1_keys)
key1 = imgdict2.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    
key2 = sorted(keylist1)
keylist2=[]
for i in key2:
    newkey2 = str(i)
    keylist2.append(newkey2)

imgdict2_keys = keylist2      
# print(imgdict2_keys)

cnt = 4001

for key1,key2 in zip(imgdict1_keys,imgdict2_keys):
	
    filename1 ='/home/vignesh/Desktop/pets2/08_02/TRK_RGB_2/'+key1+'.jpg'
    filename2 ='/home/vignesh/Desktop/pets2/08_02_backsub/'+key2+'.jpg' 
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    img3 = img1*(img2/255)
    filename3 = '/home/vignesh/Desktop/pets2/08_02_merge/'+str(cnt)+'.jpg' 
    cv2.imwrite(filename3,img3)
    cnt+=1
