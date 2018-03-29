from read_data import get_train_data, get_test_data, get_train_data_08_02, get_train_data_11_03, get_train_data_11_04
import random
import numpy as np
import pickle
import h5py
import multiprocessing as mp 
import tensorflow as tf
import PIL
from PIL import Image
import theano
from keras import backend as K

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn import svm

def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    print("giving chunks")
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def getTestData(chunk,img_rows,img_cols):
    X_test,Y_test = get_test_data(chunk,img_rows,img_cols)
    if (X_test!=None):
        #X_train/=255
        X_test = X_test-np.average(X_test)
        
    return (X_test,Y_test)

def getTrainData(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
        print(Y_train.shape)
    return (X_train, Y_train)

def getTrainDataSVM(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        # Y_train=np_utils.to_categorical(Y_train,nb_classes)
        print(Y_train.shape)
    return (X_train, Y_train)

def getTrainData_11_04(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data_11_04(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
        print(Y_train.shape)
    return (X_train, Y_train)

def getTrainData_11_03(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data_11_03(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
        print(Y_train.shape)
    return (X_train, Y_train)    

def getTrainData_08_02(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data_08_02(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
        print(Y_train.shape)
    return (X_train, Y_train)

def getTrainDataSVM_11_04(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data_11_04(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        
        print(Y_train.shape)
    return (X_train, Y_train)    

def getTrainDataSVM_11_03(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data_11_03(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        
        print(Y_train.shape)
    return (X_train, Y_train)  

def getTrainDataSVM_08_02(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data_08_02(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        
        print(Y_train.shape)
    return (X_train, Y_train)          

def spatial(img_rows,img_cols,weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_cols,img_rows))) # img row, img col, 3 for tf 
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   # weights_path = 'vgg16_weights.h5'
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # print(len(model.layers))
    # f = h5py.File(weights_path)
    # print(f.attrs['nb_layers'])
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #     # we don't look at the last (fully-connected) layers in the savefile
    #        break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()

    model.add(Dense(4096,activation='relu'))
    #convout1 = Activation('relu')
    #model.add(convout1)
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    print('Model loaded.')

    return model


batch_size = 10
img_rows= 224
img_cols= 224
model=[]
fcout=[]
# model=spatial(img_rows,img_cols,'vgg16_weights.h5')
model=spatial(img_rows,img_cols)

learnrate1=0.0001
learnrate2=0.0001 
epochs = 3
instance_count=0
nb_classes=2
chunk_size= 1385

sgd = SGD(lr=learnrate1,decay=1e-6, momentum=0.9, nesterov=True)

print 'Compiling model...'
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])

print 'Loading train dictionary...'
with open("/home/amit/Desktop/vignesh/allmerge.pickle",'rb') as f1:
    spatial_train_data=pickle.load(f1)

key1 = spatial_train_data.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    

key2 = sorted(keylist1)

keylist2=[]
for i in key2:
    if (i > 2000):
        newkey2 = str(i)
        # print(newkey2)
        keylist2.append(newkey2)

keys = keylist2
print(" total number of images being trained is {0}".format(len(keys)))

for chunk in chunks(keys,chunk_size):
    X_chunk,Y_chunk=getTrainData(chunk,nb_classes,img_rows,img_cols)
    if (X_chunk!=None and Y_chunk!=None):
        print("fitting model")        
        model.fit(X_chunk, Y_chunk, verbose=1, batch_size = batch_size, nb_epoch = epochs,validation_split=0.2)
        instance_count = instance_count + chunk_size
        print instance_count
        model.save_weights('spatial_stream_model_demo.h5',overwrite=True)
        # if instance_count%584==0:
        
      
convout1_f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[35].output])
convfeat = []
Y_svm =[]

for chunk in chunks(keys, 5):
    X_data,Y_data = getTrainDataSVM(chunk,nb_classes,img_rows,img_cols)
    if (X_chunk!=None and Y_chunk!=None):

        feat = convout1_f([X_data,0])[0]
        
        # print(len(feat))
        # print(len(convfeat))
        convfeat.append(feat)
        Y_svm.append(Y_data)

convfeat1 = np.asarray(convfeat)
Y_svm = np.asarray(Y_svm)

print(convfeat1.shape)  
print(Y_svm.shape)

convfeat1 = convfeat1.reshape(-1,convfeat1.shape[2])
Y_svm = Y_svm.reshape(-1)
print(convfeat1.shape)  
print(Y_svm.shape)

lin_clf = svm.LinearSVC()
lin_clf.fit(convfeat1, Y_svm)
print(lin_clf)

# Test the model

def spatial2(img_rows,img_cols,weights_path=None):

    model2 = Sequential()
    model2.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols))) # img row, img col, 3 for tf 
    model2.add(Convolution2D(64, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(64, 3, 3, activation='relu'))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(128, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(128, 3, 3, activation='relu'))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(256, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(256, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(256, 3, 3, activation='relu'))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu'))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu'))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu'))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))
    
   # weights_path = 'vgg16_weights.h5'
    model2.add(Flatten())
    model2.add(Dense(4096, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(4096,activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(2, activation='softmax'))
    print('Model loaded.')

    return model2

model2 = spatial2(img_rows,img_cols,'spatial_stream_model_demo.h5')
print 'Compiling model2...'
model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])

# write code for opening pickle file

print 'Loading test dictionary...'
with open("/home/amit/Desktop/vignesh/allmerge.pickle",'rb') as f1:
    spatial_test_data=pickle.load(f1)

key1 = spatial_test_data.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    

key2 = sorted(keylist1)
keylist2=[]
for i in key2:
    if i < 2000:
        newkey2 = str(i)
        keylist2.append(newkey2)

testkeys = keylist2
testfeat = []
testchunk = 9
gtlabel =[]
print("number of testkeys = {0}".format(len(testkeys)))


for chunk in chunks(testkeys, testchunk):
    X_test,Y_test = getTestData(chunk,img_rows,img_cols)
    if (X_test!=None):

        feat = convout1_f([X_test,0])[0]
        
        # print(len(feat))
        # print(len(testfeat))
        testfeat.append(feat)
        gtlabel.append(Y_test)

testfeat1 = np.asarray(testfeat)
print(testfeat1.shape)
testfeat2 = testfeat1.reshape(-1,testfeat1.shape[2])
print(testfeat2.shape)

gtlabel = np.asarray(gtlabel)
print(gtlabel.shape)
gtlabel = gtlabel.reshape(-1)
print(gtlabel.shape)

print(testfeat2.shape)
dec = lin_clf.decision_function(testfeat2)
print("decision shape is {0}".format(dec.shape))
pred = lin_clf.predict(testfeat2)
print(pred)
for i,j in enumerate(pred):
    if j==1:
        print(i)

print(gtlabel)
accuracy = 0
cnt = 0
for i,j in zip(pred,gtlabel):
    if i==j:
        accuracy+=1

    cnt+=1    
print(accuracy)
print(cnt)
print(accuracy/float(cnt))


   # 1we got (1,5,4096) and finally we get (1,n,4096)

# dataX =[]
# dataY =[]
# seq_length = 3
# for i in range(chunk_size-seq_length):
#     seqin = convfeat1[0][i:i+seq_length] 
#     seqout = convfeat1[0][i+seq_length]
#     dataX.append(seqin)
#     dataY.append(Y_data[i+seq_length])

# n_patterns = len(dataX)
# print(n_patterns)

# # reshape X to be [samples, time steps, features]
# X = np.reshape(dataX, (n_patterns, seq_length, 4096))
# Y = np.asarray(dataY)
# print(Y.shape)
# print(X.shape)


# print("LSTM")
# model2 = Sequential()
# model2.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2])))
# # model2.add(LSTM(512))
# model2.add(Dropout(0.2))
# model2.add(Dense(2,activation='softmax'))

# sgd = SGD(lr=learnrate2, momentum=0.9, nesterov=True)

# print 'Compiling model2...'
# model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])

# print ' Fitting model2...'
# model2.fit(X, Y, verbose=1, batch_size = 2, nb_epoch= 5 ,validation_split=0)
# model2.save_weights('lstm_weights.h5',overwrite=True)







