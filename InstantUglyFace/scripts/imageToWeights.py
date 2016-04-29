from __future__ import division
import CaffeModel
from scipy import ndimage
from scipy import misc
import scipy
import numpy as np


def loadMask(imagePath):
    img = misc.imread(imagePath).astype(np.float)
    img /= 255
    return img


def copyImageToWeights(net, maskData, layerName):
    # [2, 1, 0]
    print "net.params[layerName][0].data.shape", net.params[layerName][0].data.shape
    data = net.params[layerName][0].data
    data *= 0
    # net.params[layerName][0].data[...] = maskData.reshape(net.params[layerName][0].data.shape)
    for idx in range(3):
        data[idx, 0, idx, :] = maskData[:, :, idx].flat


net = CaffeModel.load("/home/wookjin/vggLauraFiles/miniLCLpretrainPhase2MaxPool/network.prototxt",
                      "/home/wookjin/vggLauraFiles/miniLCLpretrainPhase2MaxPool/models_outoutfrozen/_1e-9_momentum0.9_batch100_iter_6600.caffemodel")
imagePath = "/home/wookjin/vggLauraFiles/miniLCLpretrainPhase2MaxPool/mask.png"
maskData = loadMask(imagePath)
# print("maskData: %s" % maskData[1])
# print("maskData[-1]: %s" % maskData[-1, -1, -1])
# print("maskData.shape: %o" % maskData.shape)
print "maskData.shape:", maskData.shape
for layerName in ["data_nomean_masked", "data_reconstruction_masked"]:
    print("Copying mask " + imagePath + " to weights of layer" + layerName)
    copyImageToWeights(net,
        maskData,
        layerName)

net.forward()
im = net.blobs['data_reconstruction_masked'].data
im_correct = net.blobs['data_reconstruction'].data
#scipy.misc.imsave('fwdim.png', im[0,[2, 0, 1]])
idx = 0
X=np.zeros((3,120,0))
for idx in range(10):
    catWith = np.hstack((im_correct[idx, [2, 1, 0]], im[idx, [2, 1, 0]]))
    # catWith = np.hstack((im_correct[idx, [2, 1, 0]]))
    X=np.concatenate((X,catWith),2)

scipy.misc.imsave('tests/fwdim.png', X)
# net.save("/home/wookjin/vggLauraFiles/miniLCLpretrainPhase2MaxPool/modelWithMask.caffemodel")

print "Done"
