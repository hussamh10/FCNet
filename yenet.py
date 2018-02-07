import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, GlobalAveragePooling2D, concatenate, Reshape
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras

from unet import getUnet
from ynet import get_unet as getYnet

from hussam_data import getData

def load_data():
    imgs_train, labels_train = getData(end = 400, start=1)
    imgs_test, labels_test = getData(200, start = 190)

    return imgs_train, labels_train, imgs_test

def getYENet(r = 224, c = 224):

    ynet_out, audio_inputs, ynet_inputs = getYnet(r, c)
    unet_out, unet_inputs = getUnet(r, c)

    merge01 = merge([ynet_out, unet_out], mode = 'concat', concat_axis=3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge01)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = [ynet_inputs, audio_inputs, unet_inputs], output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    print(model.summary)

    return model
	
def getFCNet(r=224,c=224):

    ynet_out, audio_inputs, ynet_inputs = getYnet(r, c)
    unet_out, unet_inputs = getUnet(r, c)

    concat01 = concatenate([ynet_out, unet_out])
    flatten01 = GlobalAveragePooling2D()(concat01)
    dense01 = Dense(224 * 224, activation='softmax')(flatten01)
    reshape01 = Reshape([224, 224, 1])(dense01)

    model = Model(inputs = [ynet_inputs, audio_inputs, unet_inputs], output = reshape01)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    print(model.summary)

    return model

def train():
    TensorBoard(log_dir='./Graph', histogram_freq=0, 
            write_graph=True, write_images=True)

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    print("loading data")

    imgs_train, imgs_mask_train, imgs_test = load_data()
    print("loading data done")

    model = getYENet()
    print("got yenet")

    model_checkpoint = ModelCheckpoint('yenet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    print('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=1000, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, tbCallBack])

    print('predict test data')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

def trainFC():
    TensorBoard(log_dir='./Graph', histogram_freq=0, 
            write_graph=True, write_images=True)

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    print("loading data")

    imgs_train, imgs_mask_train, imgs_test = load_data()
    print("loading data done")

    model = getFCNet()
    print("got fcnet")

    model_checkpoint = ModelCheckpoint('fcnet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    print('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=1000, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, tbCallBack])

    print('predict test data')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

	
trainFC()