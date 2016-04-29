import os
import numpy as np
import sys
import scipy
caffePath = '/home/wookjin/caffe-not-nyf-nyf-with_py/'
sys.path.insert(0, caffePath + 'distribute/python/')
import caffe
import pickle
from collections import OrderedDict
# pathToPrototxt = 'VGG_FACE_oreo.prototxt'
# pathToModel = 'oreoSolver__iter_22000.caffemodel'
# pathToPrototxt = 'VGG_FACE_oreo_old.prototxt'
# pathToModel = 'ohreoSolver__iter_6000.caffemodel'
pathToPrototxt = 'network.prototxt'
pathToModel = 'models/_1e-10_PIE_train200front_lmdb__iter_10000.caffemodel'


net = caffe.Net(pathToPrototxt, pathToModel)
#net.set_mode_gpu()
#net.params['deconv1_2'][0].data.flat = net.params['conv1_1'][0].data
fwd = net.forward()

im = net.blobs['data-deconv'].data
im_correct = net.blobs['data'].data
#scipy.misc.imsave('fwdim.png', im[0,[2, 0, 1]])
idx = 0
X=np.zeros((3,120,0))
for idx in range(10):
    catWith = np.hstack((im_correct[idx, [2, 1, 0]], im[idx, [2, 1, 0]]))
    # catWith = np.hstack((im_correct[idx, [2, 1, 0]]))
    X=np.concatenate((X,catWith),2)

# scipy.misc.imsave('fwdim.png', X)


#for key in  net.params.keys():
#    blob = net.params[key]
#    if blob[0].data.shape[0]>1:
#        print(key + ': ' + str(blob[0].data.shape))


