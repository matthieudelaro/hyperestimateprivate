# setup paths  with: export LD_LIBRARY_PATH=/home/wookjin/caffe-with_py/distribute/lib/:$LD_LIBRARY_PATH
# test imports with: /caffe-with_py/distribute/python$ python2 classify.py

pathToPrototxt = 'ItWorks/VGG_FACE_cherryChocolateChip.prototxt'
# pathToModel = 'ItWorks/_iter_37925_withDropout.caffemodel'

# pathToPrototxt = 'VGG_FACE_rockyRoad.prototxt'
# pathToModel = '_iter_50113.caffemodel'

# caffe3 with python
caffe3_root = '/home/wookjin/caffe-with_py/'  # this file is expected to be in {caffe_root}/examples

# caffe1 with python
caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples

from pprint import pprint
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image

# Make sure that caffe is on the python path:
# caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe3_root + 'python')

import caffe


print "Caffe imported."



caffe.set_mode_cpu()
# net = caffe.Classifier(pathToPrototxt, pathToModel)
def inspect(pathToPrototxtParam):
    print "Inspecting ", pathToPrototxtParam
    net = caffe.Net(pathToPrototxtParam, caffe.TEST)
    print "\n\nnet\n", net
    print "\n\nnet.params\n", net.params
    print "\n\nnet.params.items()[0]\n", net.params.items()[0]
    print "\n\nnet.params.keys()\n", net.params.keys()
    print "\n\nnet.params['conv1_1']\n", net.params['conv1_1']
    print "\n\nnet.params['conv1_1']...\n", dir(net.params['conv1_1'])
    print "\n\nnet.params['conv1_1']...\n", vars(net.params['conv1_1'])
    print "\n\nnet.params['conv1_1']...\n", pprint(net.params['conv1_1'], indent="2")
    print "\n\nnet.params['conv1_1']...\n", len(net.params['conv1_1'])
    print "\n\nnet.params['conv1_1']...\n", net.params['conv1_1'][0]
    print ""
    # print "blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys())
    # print "params {}".format(net.params.keys())
    # for index, key in enumerate(net.params.keys()):
    #     # print key, net.params[index][0:3], net.params[key][0:3]
    #     print net.params[key][0]

    print "shapes of the blobs"
    for layerIndex, key in enumerate(net.params.keys()):
        # print key, net.params[index][0:3], net.params[key][0:3]
        print key
        layer = net.params[key]
        for blobIndex in range(len(layer)):
            blob = net.params[key][blobIndex]
            average = np.mean(blob.data)
            # print "\t", blob.data.shape, np.flatten(blob.data)[0:3]
            print "\t", blob.data.shape, "average=", average, blob.data.flatten()[0:3]
    # configure plotting
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'

# inspect(pathToPrototxt)
# inspect('/home/wookjin/vggLauraFiles/VGG_FACE_FS_50_tmp.prototxt')
