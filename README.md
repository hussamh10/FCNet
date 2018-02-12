# YENet
A combination of Unet and Ynet [FYP]

Instructions to run:

1. Clone using: `git clone https://github.com/hussamh10/YENet.git`
2. Create a Graph folder in the YENet dir
3. Add the data folder from [here](https://drive.google.com/open?id=1oXgl1cTaG7EDcolbMO9IANJV3DUW-ij8) to the dir
4. activate the tensorflow env using: `activate tensorflow`
5. Start the training `python yenet.py` (not to be confused with ynet.py)
6. Start the tensorboard (A little time after an epoch has completed) in an tensorflow activated env
7. Make sure in the same dir as Graph
8. Start tb using `tensorboard --logdir=Graph\`
