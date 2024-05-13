import tensorflow as tf
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
from helper import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

windowSize = 9

X, y = load('Samson')

 
Band = X.shape[0] 
Endmember = y.shape[0]

K = X.shape[0]
Nx = int(np.sqrt(X.shape[1]))
Ny = int(np.sqrt(X.shape[1]))
X= np.reshape(X, (Band,Nx,Ny))
X = np.transpose(X, (1, 2, 0))

y= np.reshape(y, (Endmember,Nx,Ny))
y = np.transpose(y, (1, 2, 0))

K = 13
X,pca = applyPCA(X,numComponents=K)
X, y = createImageCubes(X, y, windowSize=windowSize)

test_ratio = 0.8
L = K

Xtrain, ytrain, Xtest, ytest = splitTrainTestSet(X, y, test_ratio)

windowSize = 9
Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)

S = windowSize
output_units = y.shape[1]

## input layer
input_layer = Input((S, S, L, 1))
lr = 0.0001 
decay = 1e-04
rate = 0.03
## convolutional layers
conv_layer1 = Conv3D(filters=128, kernel_size=(3, 3, 7), activation='tanh')(input_layer)
conv_layer1 = Dropout(rate)(conv_layer1)
conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 5), activation='tanh')(conv_layer1)
conv_layer2 = Dropout(rate)(conv_layer2)
conv3d_shape = conv_layer2._keras_shape
conv_layer2 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer2)
conv_layer2 = Conv2D(filters=32, kernel_size=(3,3), activation='tanh')(conv_layer2)
conv_layer3 = Dropout(rate)(conv_layer2)

flatten_layer = Flatten()(conv_layer3)


output_layer = Dense(units=output_units, activation='softmax')(flatten_layer)

model = Model(inputs=input_layer, outputs=output_layer)

model.summary()

# compiling the model
adam = Adam(lr, decay)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# checkpoint
filepath = "Samson-best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x=Xtrain, y=ytrain, epochs=50, callbacks=callbacks_list)

model.load_weights("Samson-best-model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


X = X.reshape(-1, windowSize, windowSize, K, 1)
Y_pred_all = model.predict(X)


np.savetxt('./Samsonresult/Y_pred_all.txt', Y_pred_all, delimiter=",")




