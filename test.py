import tensorflow as tf
import cv2 as cv
import numpy as np
import hussam_data as data
from matplotlib import pyplot as plt

i, l = data.getData(2, start=1)

u = i[-1]
u = u.reshape((224, 224, 2))

print(u.shape)


print(u[:,:,1].shape)

plt.imshow(u[:,:,0])

plt.savefig('out\\' + '1.jpg')

plt.imshow(u[:,:,1])

plt.savefig('out\\' + '2.jpg')

