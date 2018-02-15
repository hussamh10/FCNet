#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os 
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
from hussam_data import getData as gd

def getFCNet():

    input01 = Input((480, 640, 1))
    input02 = Input((480, 640, 1))
    
    dense01 = Dense(224, activation='relu')(input01)
    dense02 = Dense(224, activation='relu')(input02)

    dense03 = Dense(224, activation='relu')(dense01)
    dense04 = Dense(224, activation='relu')(dense02)

    merge01 = merge([dense03, dense04], mode='concat')

    dense05 = Dense(224, activation='relu')(merge01)
    dense06 = Dense(1, activation='relu')(dense05)

    model = Model(inputs=[input01, input02], output=dense06)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()
    
    return model

def train():
    i, l = gd(start = 1, end = 200)

    TensorBoard(log_dir='./Graph', histogram_freq=0, 
            write_graph=True, write_images=True)

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model = getFCNet()

    model_checkpoint = ModelCheckpoint('fcnet.hdf5', monitor='loss',verbose=1, save_best_only=True)

    model.fit(i, l, batch_size=1, epochs=1000, verbose=1,validation_split=0.2, shuffle=True, callbacks=[tbCallBack, model_checkpoint])

train()
