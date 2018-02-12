from matplotlib import pyplot as plt 
from cv2 import imread
import numpy as np

import numpy as np
import cv2

def getImage(i, source, main_dir):
    name = str(i) + '.jpg'

    print(main_dir + source + '\\' + name)

    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start=0):
    uimgs = getUnetData(end, start)
    yimgs, yauds = getYnetData(end, start)

    labels = getLabels(end, start)

    inputs = [yimgs, yauds, uimgs]

    return inputs, labels

def getUnetData(end, start=0, dir='data\\unet\\'): #data/unet/imgs/1.jpg
    imgs = []
    img_src1 = 'imgs\\1'
    img_src2 = 'imgs\\2'

    for i in range(start, end):
        image1 = getImage(i, img_src1, dir)
        image2 = getImage(i, img_src2, dir)
        image1 = image1.reshape((224, 224, 1))
        image2 = image2.reshape((224, 224, 1))

        image = np.concatenate([image1, image2], axis=2)
        imgs.append(image)
        
    imgs = np.array(imgs)
    return imgs

def getLabels(end, start=0, dir='data\\ynet\\'): #data/ynet/labels
    labels = []
    label_src = 'labels'

    for i in range(start, end):
        labels.append(getImage(i, label_src, dir))

    labels = np.array(labels)
    return labels

def getYnetData(end, start=0, main_dir='data\\ynet\\'): #data/ynet/imgs/1.jpg
    img_src = 'imgs'
    audio_src = 'audio'

    imgs = []
    audios = []

    for i in range(start, end):
        imgs.append(getImage(i, img_src, main_dir))
        audios.append(getImage(i, audio_src, main_dir))
        
    imgs = np.array(imgs)
    audios = np.array(audios)
    return imgs, audios
