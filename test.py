import cv2
from hussam_data import getData
import numpy as np
from matplotlib import pyplot as plt

def testHussamData():

    i, l = getData(30, 29)

    yimg = i[0]
    yaud = i[1]
    uimg = i[2]
    limg = l

    uimg1, uimg2 = np.split(uimg, 2, axis=2)

    yimg = yimg.reshape((224, 224))
    yaud = yaud.reshape((224, 224))
    uimg1 = uimg1.reshape((224, 224))
    uimg2 = uimg2.reshape((224, 224))
    limg = limg.reshape((224, 224))

    print(yimg.shape, yaud.shape, uimg1.shape, uimg2.shape, limg.shape)
    
    plt.imshow(yimg)
    plt.savefig('out\\yimg.jpg')

    plt.imshow(yaud)
    plt.savefig('out\\yaud.jpg')

    plt.imshow(uimg1)
    plt.savefig('out\\uimg1.jpg')

    plt.imshow(uimg2)
    plt.savefig('out\\uimg2.jpg')

    plt.imshow(limg)
    plt.savefig('out\\limg.jpg')

testHussamData()
    


