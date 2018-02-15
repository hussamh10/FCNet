from matplotlib import pyplot as plt 
from cv2 import imread
import numpy as np

import numpy as np
import cv2

def getImage(i, source, main_dir, ext):
    name = str(i) + ext

    print(main_dir + source + '\\' + name)

    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.reshape(480, 640, 1)
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start=0):
    in1, in2 = getInputs(end, start)
    labels = getLabels(end, start)

    inputs = [in1, in2]

    return inputs, labels

def getInputs(end, start=0, dir='data\\'): #data/unet/imgs/1.jpg
    imgs1 = []
    imgs2 = []

    for i in range(start, end):
        imgs1.append(getImage(i, img_src1, dir, '.png'))
        imgs2.append(getImage(i, img_src2, dir, '.png'))


    imgs1 = np.array(imgs1)
    imgs2 = np.array(imgs2)
    return imgs1, imgs2

def getLabels(end, start=0, dir='data\\'): #data/ynet/labels
    labels = []
    label_src = 'labels\\'

    for i in range(start, end):
        labels.append(getImage(i, label_src, dir, '.jpg'))

    labels = np.array(labels)
    return labels
