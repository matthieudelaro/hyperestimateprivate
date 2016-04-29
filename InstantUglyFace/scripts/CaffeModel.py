import sys
# caffe3 with python
caffe3_root = '/home/wookjin/caffe-with_py/'  # this file is expected to be in {caffe_root}/examples
# caffe1 with python
caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples
caffe1l_root = '/home/wookjin/caffe1WithLocallyConLayers-with_py/'  # this file is expected to be in {caffe_root}/examples
caffe_root = caffe1l_root  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np

# see http://stackoverflow.com/a/32509003
def load(pathToPrototxtParam, pathToCaffemodelParam, verbose=False):
    """Load the network in memory"""
    if verbose:
        print "Loading ", pathToPrototxtParam, pathToCaffemodelParam
    net = caffe.Net(pathToPrototxtParam, pathToCaffemodelParam)
    return net


def inspect(net, verbose=False):
    """Return the list of the layers, with their shapes, their names, ..."""
    if verbose:
        print "Inspecting network"
        print "\n\nnet.params\n", net.params
        print "\n\nnet.params.keys()\n", net.params.keys()
        print ""
        print "Shapes of the blobs"
    layers = []
    for layerIndex, key in enumerate(net.params.keys()):
        if verbose:
            print key
        layer = net.params[key]
        blobs = []
        for blobIndex in range(len(layer)):
            blob = net.params[key][blobIndex]
            average = np.mean(blob.data)
            blobs.append(blob.data.shape)
            if verbose:
                print "\t", "shape=", blob.data.shape, \
                      "average=", average, \
                      "beginning=", blob.data.flatten()[0:3]
        layers.append((key, blobs))
    layersNames = []
    for l in net._layer_names:
        layersNames.append(l)
    if verbose:
        print "All layers (from private variable):"
        for l in layersNames:
            print l
    return layers


def quickTests():
    # net = load('/home/wookjin/vggLauraFiles/matthieu/VGG_FACE_FS_tmpMatthieuNew__withConvFS6.50_insteadOf__fullyConnected_fc6.50.prototxt',
    #         '/home/wookjin/vggLauraFiles/matthieu/modifiedFCtoCVMatthieu_smarter__turnedInto__leftPartOfNetwork_convolutionalFacespace_FS6.50.caffemodel',
    #         True)
    net = load('/home/wookjin/vggLauraFiles/matthieu/wholeNetwork_oreo.prototxt',
    # net = load('/home/wookjin/vggLauraFiles/VGG_FACE_oreo.prototxt',
               '/home/wookjin/vggLauraFiles/matthieu/modifiedFCtoCVMatthieu_smarter__turnedInto__leftPartOfNetwork_convolutionalFacespace_FS6.50.caffemodel',
               # '/home/wookjin/vggLauraFiles/leftPartOfNetwork_convolutionalFacespace_FS6.50.caffemodel',
            True)
    layers = inspect(net, True)
    print "layers:", layers
    return net

if __name__ == '__main__':
    res = quickTests()

