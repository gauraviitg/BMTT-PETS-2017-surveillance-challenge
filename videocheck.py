from read_data import get_train_data, get_test_data, get_sample_data
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

def getSampleData(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_sample_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        # X_train/=255
        X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
    return (X_train,Y_train)

def getTestData(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_test_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        #X_train/=255
        X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
    return (X_train,Y_train)

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
        
        print(Y_train.shape)
    return (X_train, Y_train)    

def test(model, nb_epoch, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size):
    keys=spatial_test_data.keys()
    random.shuffle(keys)
    X_test,Y_test = getTestData(keys[1:50],nb_classes,img_rows,img_cols)
    return (X_test, Y_test)

def spatial(img_rows,img_cols,weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols))) # img row, img col, 3 for tf 
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
    print(len(model.layers))
    f = h5py.File(weights_path)
    print(f.attrs['nb_layers'])
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
           break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    model.add(Dense(4096,activation='relu'))
    #convout1 = Activation('relu')
    #model.add(convout1)
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    print('Model loaded.')

    return model


batch_size = 20
img_rows=224
img_cols=224
model=[]
fcout=[]

learnrate1=0.001
learnrate2=0.001 
epochs = 10
instance_count=0
nb_classes=2
chunk_size=584
sgd = SGD(lr=learnrate1, momentum=0.9, nesterov=True)


model=spatial(img_rows,img_cols,'spatial_stream_model_demo.h5')

print 'Compiling model...'
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])



key1 = spatial_train_data.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    

key2 = sorted(keylist1)
keylist2=[]
for i in key2:
    newkey2 = str(i)
    keylist2.append(newkey2)

keys = keylist2

convout1_f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[35].output])
convfeat = []
Y_svm = []

for chunk in chunks(keys, 73):
    X_data,Y_data=getTrainDataSVM(chunk,nb_classes,img_rows,img_cols)
    if (X_chunk!=None and Y_chunk!=None):

        feat = convout1_f([X_data,0])[0]
        
        print(len(feat))
        print(len(convfeat))
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

dec = lin_clf.decision_function(X_test)
print(dec.shape)

