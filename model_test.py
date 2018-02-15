from fcnet import getFCNet
from hussam_data import getData as gd
from matplotlib import pyplot as plt
import keras

def test(start):
    md = getFCNet()
    md.load_weights('fcnet.hdf5')

    h, hy = gd(start = start, end = start + 1)

    del hy
    
    print("predicting")

    hp = md.predict(h)

    i = start
    for p in hp:
        plt.imshow(p.reshape((480, 640)))
        plt.savefig('' + 'out\\' + str(i) + '.png')
        print(''  + 'out\\' + str(i) + '.png')
        i += 1

    print("Done")

for i in range(200, 300):
    test(i)
