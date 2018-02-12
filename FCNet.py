import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, GlobalAveragePooling2D, concatenate, Reshape, Flatten, MaxPooling2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
from hussam_data import getData

def getData():
    pass

def getFCNet():
    input1 = Input((224, 224))
    input2 = Input((224, 224))

    dense1 = Dense(224, activation='relu')(input1)
    dense2 = Dense(224, activation='relu')(input2)

    merge1 = merge([dense1, dense2], mode = 'concat')

    dense3 = Dense(224, activation='relu')(merge1)
    dense4 = Dense(224, activation='sigmoid')(dense3)
    
    model = Model(inputs=[input1, input2], output=dense4)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    return model

getFCNet()
    
