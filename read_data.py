#import cv2
import pickle
import numpy as np
import PIL
from PIL import Image
import os.path
import sys
# import cv2


def get_train_data(chunk, img_row, img_col):
    # print(" \n get train data - running")
    X_train = []
    Y_train = []
    with open("/home/amit/Desktop/vignesh/allmerge2.pickle",'rb') as f1:
        spatial_train_data=pickle.load(f1)
    try:
        for imgname in chunk:
            
            filename = "/home/amit/Desktop/vignesh/allmerge/"+str(imgname)+'.jpg' 
            if os.path.exists(filename) == True:
              # print(filename)
              # img = cv2.imread(filename)
              # img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
              
              a = Image.open(filename)
              # print("image opened")
              a = a.resize((img_row,img_col), PIL.Image.ANTIALIAS)
              # print("resized")
              img = np.asarray(a)
              # print("converted")
              img = np.rollaxis(img.astype(np.float32),2) 
              # print("rolled")
              X_train.append(img)
              # print("image appended")
              # print("X_train shape is {0}".format(len(X_train)))
              Y_train.append(spatial_train_data[imgname])
              # print("Y_train shape is {0}".format(len(Y_train)))
        
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        # print(Y_train.shape)
        # print(" \n get train data - finished")
        return X_train,Y_train
    except:
        X_train=None
        Y_train=None
        print(" \n get train data exception- finished")
        return X_train,Y_train

def get_train_data_11_04(chunk, img_row, img_col):
    print(" \n get train data - running")
    X_train = []
    Y_train = []
    with open("/home/amit/Desktop/vignesh/11_04_backsub.pickle",'rb') as f1:
        spatial_train_data=pickle.load(f1)
    try:
        for imgname in chunk:
            
            filename = "/home/amit/Desktop/vignesh/11_04_merge/"+str(imgname)+'.jpg' 
            if os.path.exists(filename) == True:
              print(filename)
              # img = cv2.imread(filename)
              # img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
              
              a = Image.open(filename)
              print("image opened")
              a = a.resize((img_row,img_col), PIL.Image.ANTIALIAS)
              print("resized")
              img = np.asarray(a)
              print("converted")
              img = np.rollaxis(img.astype(np.float32),2) 
              print("rolled")
              X_train.append(img)
              print("image appended")
              # print("X_train shape is {0}".format(len(X_train)))
              Y_train.append(spatial_train_data[imgname])
              # print("Y_train shape is {0}".format(len(Y_train)))
        
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        print(Y_train.shape)
        print(" \n get train data - finished")
        return X_train,Y_train
    except:
        X_train=None
        Y_train=None
        print(" \n get train data exception- finished")
        return X_train,Y_train



def get_train_data_11_03(chunk, img_row, img_col):
    print(" \n get train data - running")
    X_train = []
    Y_train = []
    with open("/home/amit/Desktop/vignesh/11_03_backsub.pickle",'rb') as f1:
        spatial_train_data=pickle.load(f1)
    try:
        for imgname in chunk:
            
            filename = "/home/amit/Desktop/vignesh/11_03_merge/"+str(imgname)+'.jpg' 
            if os.path.exists(filename) == True:
              print(filename)
              # img = cv2.imread(filename)
              # img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
              
              a = Image.open(filename)
              print("image opened")
              a = a.resize((img_row,img_col), PIL.Image.ANTIALIAS)
              print("resized")
              img = np.asarray(a)
              print("converted")
              img = np.rollaxis(img.astype(np.float32),2) 
              print("rolled")
              X_train.append(img)
              print("image appended")
              # print("X_train shape is {0}".format(len(X_train)))
              Y_train.append(spatial_train_data[imgname])
              # print("Y_train shape is {0}".format(len(Y_train)))
        
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        print(Y_train.shape)
        print(" \n get train data - finished")
        return X_train,Y_train
    except:
        X_train=None
        Y_train=None
        print(" \n get train data exception- finished")
        return X_train,Y_train

def get_train_data_08_02(chunk, img_row, img_col):
    print(" \n get train data - running")
    X_train = []
    Y_train = []
    with open("/home/amit/Desktop/vignesh/08_02_backsub.pickle",'rb') as f1:
        spatial_train_data=pickle.load(f1)
    try:
        for imgname in chunk:
            
            filename = "/home/amit/Desktop/vignesh/08_02_merge/"+str(imgname)+'.jpg' 
            if os.path.exists(filename) == True:
              print(filename)
              # img = cv2.imread(filename)
              # img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
              
              a = Image.open(filename)
              print("image opened")
              a = a.resize((img_row,img_col), PIL.Image.ANTIALIAS)
              print("resized")
              img = np.asarray(a)
              print("converted")
              img = np.rollaxis(img.astype(np.float32),2) 
              print("rolled")
              X_train.append(img)
              print("image appended")
              # print("X_train shape is {0}".format(len(X_train)))
              Y_train.append(spatial_train_data[imgname])
              # print("Y_train shape is {0}".format(len(Y_train)))
        
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        print(Y_train.shape)
        print(" \n get train data - finished")
        return X_train,Y_train
    except:
        X_train=None
        Y_train=None
        print(" \n get train data exception- finished")
        return X_train,Y_train

def get_test_data(chunk, img_row, img_col):
    # print(" \n get test data - running")
    # print(len(chunk))
    X_test = []
    Y_test = []
    with open("/home/amit/Desktop/vignesh/allmerge2.pickle",'rb') as f1:
        spatial_test_data=pickle.load(f1)
    try:
        for imgname in chunk:
            
            filename = "/home/amit/Desktop/vignesh/allmerge/"+imgname+'.jpg'
            if os.path.exists(filename) == True:
              # print(filename)
              # img = cv2.imread(filename)
              # img = np.rollaxis(cv2.resize(img,(img_row,img_col)).astype(np.float32),2)
               
              a = Image.open(filename)

              a = a.resize((img_row,img_col), PIL.Image.ANTIALIAS)
              img = np.asarray(a)
              img = np.rollaxis(img.astype(np.float32),2) 
              X_test.append(img)
              Y_test.append(spatial_test_data[imgname])
              

        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)
        # print(" \n get test data - finished")
        return X_test,Y_test
    except:
        X_test=None
        Y_test=None
        print(" \n get test data exception - finished")
        return X_test,Y_test

if __name__ == '__main__':
      gc.collect()        
