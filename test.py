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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
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
        # print(Y_train.shape)
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
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))
    print('Model loaded.')

    return model


batch_size = 10
img_rows= 224
img_cols= 224
model=[]
fcout=[]
learnrate1=0.0001
learnrate2=0.0001 
epochs = 5
loop = 5
instance_count=0
nb_classes=2
chunk_size= 1692  #  change this for separate cases   11_04 -- 729 , 11_04_RGB2 =  , 11_03 -- 329, 11_03_RGB2 =  ,  08-02 -- 1056, 08_02_RGB1 = 307
seq_length = 10
nfeat = 1024


model=spatial(img_rows,img_cols)
# model.load_weights('spatial_stream_test.h5')
sgd = SGD(lr=learnrate1,decay=1e-6, momentum=0.9, nesterov=True)

print 'Compiling model...'
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])

print 'Loading train dictionary...'
with open("/home/amit/Desktop/vignesh/allmerge3.pickle",'rb') as f1:
    spatial_train_data=pickle.load(f1)

key1 = spatial_train_data.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
# keys_cnn = np.random.shuffle(keys)

for i in range(loop):
    np.random.shuffle(sorted(keylist1))
    key2 = keylist1
    keylist2=[]
    for i in key2:
        if (i > 1000):         # change the limits for training <4000 to train 11_04 and 11_03,  >2000 to train 11_03 and 08_02,  <2000 and >4000 to train 11_04 and 08_02
            newkey2 = str(i)
            # print(newkey2)
            keylist2.append(newkey2)

    keys = keylist2                    # Training keys 
    print(" total number of images being trained is {0}".format(len(keys)))
    X_chunk,Y_chunk=getTrainData(keys,nb_classes,img_rows,img_cols)

    if (X_chunk!=None and Y_chunk!=None):
        print("fitting model")        
        model.fit(X_chunk, Y_chunk, verbose=1, batch_size = batch_size, nb_epoch = 1,validation_split=0.2)
        model.save_weights('spatial_stream_test.h5',overwrite=True)
        # if instance_count%584==0:
   

# FEATURE OUTPUT FUNCTION    
convout1_f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[35].output])
convfeat = []
Y_0 =[]

for chunk in chunks(sorted(keys), 9):   # change the value of 2 for different cases --- to train on 11_04 and 11_03 = 2 , other cases try 5
    X_data,Y_data = getTrainDataSVM(chunk,nb_classes,img_rows,img_cols)
    if (X_chunk!=None and Y_chunk!=None):

        feat = convout1_f([X_data,0])[0]
        # print(len(feat))
        # print(len(convfeat))
        convfeat.append(feat)
        Y_0.append(Y_data)

convfeat = np.asarray(convfeat)
Y_0 = np.asarray(Y_0)              
Y_0 = Y_0.reshape(-1)                                # Only labels
Y_1 = np_utils.to_categorical(Y_0,nb_classes)        # One hot vector labels 
Y_1 = np.asarray(Y_1)

print(convfeat.shape)  
print(Y_0.shape)

convfeat = convfeat.reshape(-1,convfeat.shape[2])      # Reshaping to get (number_of_training_images, nfeat)

print(convfeat.shape)  
print(Y_1.shape)

# ---------------- end of cnn architecture ---------------
dataX =[]
dataY =[]

for i in range(chunk_size-seq_length):       # can change seq_length
    seqin = convfeat[i:i+seq_length] 
    seqout = convfeat[i+seq_length]

    dataX.append(seqin)
    dataY.append(Y_1[i+seq_length])
    
n_patterns = len(dataX)
print(n_patterns)
        # reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, nfeat))
Y = np.asarray(dataY)
print(X.shape)
print(Y.shape)


# model.add(Bidirectional(LSTM(64)))
print("LSTM")
model2 = Sequential()
model2.add(LSTM(1024, return_sequences = True, input_shape=(X.shape[1], X.shape[2])))
model2.add(Dropout(0.2))
model2.add(Bidirectional(LSTM(1024)))
model2.add(Dropout(0.2))                          # can change dropout
model2.add(Dense(2,activation='softmax'))

sgd = SGD(lr=learnrate2,decay = 1e-6, momentum=0.9, nesterov=True)
# model2.load_weights('lstm_weights_test.h5')
print 'Compiling model2...'
model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])

print ' Fitting model2...'
model2.fit(X, Y, verbose=1, batch_size = batch_size, nb_epoch = epochs ,validation_split=0.2)
model2.save_weights('lstm_weights_test.h5',overwrite=True)

# ----------------------------- end of LSTM architecture -----------------------------
lstm_f = K.function([model2.layers[0].input, K.learning_phase()], [model2.layers[3].output])

lstmfeat = lstm_f([X,0])[0]
Y_2 = Y_0[seq_length:]

lstmfeat = np.asarray(lstmfeat)
Y_2 = np.asarray(Y_2)

lin_clf = svm.LinearSVC()
lin_clf.fit(lstmfeat, Y_2)
print(lin_clf)

# TESTING.....................

print 'Loading test dictionary...'
with open("/home/amit/Desktop/vignesh/allmerge2.pickle",'rb') as f1:
    spatial_test_data=pickle.load(f1)

key1 = spatial_test_data.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    

key2 = sorted(keylist1)
keylist2=[]
for i in key2:
    if (i < 1000):                # change limits of indices for other cases
        newkey2 = str(i)
        keylist2.append(newkey2)
                     
testkeys = keylist2
testfeat = []
testchunk = 9                # change for other cases - to test on 11_04 = 729 = choose 9, 11_03 = 329 = choose 7, 08_02 = 1056 = choose 8
gtlabel =[]
sptestx = []
sptesty = []
print("number of testkeys = {0}".format(len(testkeys)))

for chunk in chunks(testkeys, testchunk):
    X_test,Y_test = getTestData(chunk,img_rows,img_cols)
    if (X_test!=None):
        
        sptestx.append(X_test)          # For cnn
        feat = convout1_f([X_test,1])[0]
        # print(len(feat))
        # print(len(testfeat))
        testfeat.append(feat)           # For lstm
        gtlabel.append(Y_test)           

testfeat1 = np.asarray(testfeat)
print(testfeat1.shape)

testfeat2 = testfeat1.reshape(-1,testfeat1.shape[2])
print(testfeat2.shape)

gtlabel = np.asarray(gtlabel)
print(gtlabel.shape)

gtlabel = gtlabel.reshape(-1)
gt1 = np_utils.to_categorical(gtlabel,nb_classes) 
gt1 = np.asarray(gt1)

sptestx = np.asarray(sptestx)
print(sptestx.shape)

sptestX = sptestx.reshape(len(testkeys),3,img_rows,img_cols)
print(sptestX.shape)
print(testfeat2.shape)
print(gt1.shape)
print(testfeat2.shape)

dataXtest = []
dataYtest = []

for i in range(len(testkeys)-seq_length):
    seqin = testfeat2[i:i+seq_length] 
    seqout = testfeat2[i+seq_length]

    dataXtest.append(seqin)
    dataYtest.append(gtlabel[i+seq_length])
    
n_patterns1 = len(dataXtest)
print(n_patterns1)
# for i in range(10):
#     dataYtest.insert(0,0)
        # reshape X to be [samples, time steps, features]
Xfintest = np.reshape(dataXtest, (n_patterns1, seq_length, nfeat))
Yfintest = np.asarray(dataYtest)
print(Xfintest.shape)
print(Yfintest.shape)

lstmtest = lstm_f([Xfintest,1])[0]

dec = lin_clf.decision_function(lstmtest)
pred0 = lin_clf.predict(lstmtest)
print(np.asarray(pred0).shape)
# print("decision shape is {0}".format(dec.shape))
pred1 = model.predict_classes(sptestX, batch_size = batch_size)      # pred1 == CNN
pred1 = np.asarray(pred1)
pred2 = model2.predict_classes(Xfintest, batch_size = batch_size)    # pred2 == CNN+LSTM
# pred2 = pred2arr.tolist()
# for i in range(10):
#     pred2.insert(0,0)
pred2 = np.asarray(pred2)
print(pred1.shape)
print(pred2.shape)

for i,j in enumerate(pred0):
    if j==1:
        print(i)
print("----------------------------------------------------")
for i,j in enumerate(pred1):
    if j==1:
        print(i)
print("----------------------------------------------------")
for i,j in enumerate(pred2):
    if j==1:
        print(i)                


accuracy0 = 0
cnt0 = 0
 
for i,j in zip(pred0,Yfintest):
    if i==j:
        accuracy0+=1

    cnt0+=1    
print(accuracy0)
print(cnt0)
print(accuracy0/float(cnt0))

accuracy1 = 0
cnt1 = 0

# pred1 = np_utils.to_categorical(pred,nb_classes) 
for i,j in zip(pred1,Yfintest):
    if i==j:
        accuracy1+=1

    cnt1+=1    
print(accuracy1)
print(cnt1)
print(accuracy1/float(cnt1))

accuracy2 = 0
cnt2 = 0
for i,j in zip(pred2,Yfintest):
    if i==j:
        accuracy2+=1

    cnt2+=1    
print(accuracy2)
print(cnt2)
print(accuracy2/float(cnt2))

# output = []
# index = 10
# pred3 = (pred1[seq_length:]+pred2)/float(2)
# for i,j in pred3:
#     if j>0.5:            # Vary threshold other than 0.5
#         print(index)
#         output.append(1)
#     else:
#         output.append(0)
#     index+=1        
# # print(gtlabel)
# # print(Yfintest)
# accuracy3 = 0
# cnt3 = 0
# # pred1 = np_utils.to_categorical(pred,nb_classes) 
# for i,j in zip(output,Yfintest):
#     if i==j:
#         accuracy3+=1

#     cnt3+=1    
# print(accuracy3)
# print(cnt3)
# print(accuracy3/float(cnt3))