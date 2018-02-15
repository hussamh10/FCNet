from cv2 import imread
import numpy as np

def getImage(i, source, main_dir, ext):
    name = str(i) + ext

    print(main_dir + source + '\\' + name)

    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.reshape(480, 640, 1)
    img = img.astype('float32')
    img /= 255
    return img

def generate(size):

    dir = 'data\\'
    img_src1 = 'enet_out\\'
    img_src2 = 'ynet_out\\'
    label_src = 'labels\\'
    i = 1

    while i < size:
        imgs1 = getImage(i, img_src1, dir, '.png')
        imgs2 = getImage(i, img_src2, dir, '.png')
        labels = getImage(i, label_src, dir, '.jpg')

        imgs1 = imgs1.reshape(1, imgs1.shape[0], imgs1.shape[1], imgs1.shape[2])
        imgs2 = imgs2.reshape(1, imgs2.shape[0], imgs2.shape[1], imgs2.shape[2])
        labels = labels.reshape(1, labels.shape[0], labels.shape[1], labels.shape[2])

        print(imgs1.shape)
        
        i += 1

        yield([imgs1, imgs2], labels)

