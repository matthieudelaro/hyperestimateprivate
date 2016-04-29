# Make sure that caffe is on the python path:
# caffe3 with python
caffe3_root = '/home/wookjin/caffe-with_py/'  # this file is expected to be in {caffe_root}/examples

# caffe1 with python
caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples
caffe_root = caffe1_root  # this file is expected to be in {caffe_root}/examples

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import pickle
import numpy as np

# # Load the original network and extract the fully-connected layers' parameters.
# net = caffe.Net('/home/wookjin/caffe_obsolete/caffe-not-nyf-nyf_old/examples/mnist/lenet_train_test.prototxt',
#                 '/home/wookjin/caffe_obsolete/caffe-not-nyf-nyf_old/examples/mnist/lenet_iter_10000.caffemodel')
# net.set_phase_test()
# params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
# fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

# for fc in params:
    # print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

def inspect(pathToPrototxtParam, pathToCaffemodelParam):
    print "Inspecting ", pathToPrototxtParam, pathToCaffemodelParam
    net = caffe.Net(pathToPrototxtParam, pathToCaffemodelParam)
    print "\n\nnet\n", net
    print "\n\nnet.params\n", net.params
    print "\n\nnet.params.items()[0]\n", net.params.items()[0]
    print "\n\nnet.params.keys()\n", net.params.keys()
    # print "\n\nnet.params['conv1_1']\n", net.params['conv1_1']
    # print "\n\nnet.params['conv1_1']...\n", dir(net.params['conv1_1'])
    # print "\n\nnet.params['conv1_1']...\n", vars(net.params['conv1_1'])
    # print "\n\nnet.params['conv1_1']...\n", pprint(net.params['conv1_1'], indent="2")
    # print "\n\nnet.params['conv1_1']...\n", len(net.params['conv1_1'])
    # print "\n\nnet.params['conv1_1']...\n", net.params['conv1_1'][0]
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
    return net
    # configure plotting
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'




def loadSavedAsNumpy1(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
        return obj


def testsLoad():
    loaded = loadSavedAsNumpy1("/home/wookjin/caffe-with_py/distribute/python/test.net")

    print "shapes of the blobs"
    for layerIndex, key in enumerate(loaded.keys()):
        # print key, net.params[index][0:3], net.params[key][0:3]
        print key
        layer = loaded[key]
        data = layer
        print "\t", data.shape, data.flatten()[0:3]
        # for blobIndex in range(len(layer)):
        #     blob = loaded[blobIndex]
        #     # print "\t", blob.data.shape, np.flatten(blob.data)[0:3]
        #     print "\t", blob.data.shape, blob.data.flatten()[0:3]
    # print(loaded.keys())

# inspect('/home/wookjin/caffe_obsolete/caffe-not-nyf-nyf_old/examples/mnist/lenet_train_test.prototxt',
#         '/home/wookjin/caffe_obsolete/caffe-not-nyf-nyf_old/examples/mnist/lenet_iter_10000.caffemodel')

def downgrade(pathToPrototxtParam, pathToCaffemodelParam, special=False):
    loaded = loadSavedAsNumpy1("/home/wookjin/caffe-with_py/distribute/python/test.net")
    net = caffe.Net(pathToPrototxtParam, pathToCaffemodelParam)

    print "loaded keys:", loaded.keys()
    print "Net keys   :", net.params.keys()

    print "downgrading..."
    for layerIndex, key in enumerate(net.params.keys()):
        # print key, net.params[index][0:3], net.params[key][0:3]
        print "\nLayer", key
        layer = net.params[key]
        print "layer", layer
        print "layer[0]", layer[0]
        # print "layer[0].data", layer[0].data
        print "layer[0].data.shape", layer[0].data.shape

        if key in loaded.keys():
            print "loaded[key].shape", loaded[key].shape
            reshaped = loaded[key].reshape(layer[0].data.shape)
            print "loaded[key].shape after reshaping", reshaped.shape

            layer[0].data[...] = reshaped
            # print "loaded[key]", loaded[key]
            # layer[0] = loaded[key][:]
            # for blobIndex in range(len(layer)):
            #     blob = net.params[key][blobIndex]
            #     # print "\t", blob.data.shape, np.flatten(blob.data)[0:3]
            #     print "\t", blob.data.shape, blob.data.flatten()[0:3]
        else:
            print "Key", key, " from numpy save does not exist in caffemodel => ignored."
        outputPath = 'downgraded.caffemodel'
        net.save(outputPath)
        print "Saved to", outputPath

def copyLayerToLayer(pathToPrototxtParam, pathToCaffemodelParam, fromLayer, toLayer, shapeLayer):
    loaded = loadSavedAsNumpy1("/home/wookjin/caffe-with_py/distribute/python/test.net")
    net = caffe.Net(pathToPrototxtParam, pathToCaffemodelParam)

    print "loaded keys:", loaded.keys()
    print "Net keys   :", net.params.keys()

    print "downgrading..."
    for layerIndex, key in enumerate(net.params.keys()):
        # print key, net.params[index][0:3], net.params[key][0:3]
        print "\nLayer", key
        layer = net.params[key]
        # print "layer", layer
        # print "layer[0]", layer[0]
        # print "layer[0].data.shape", layer[0].data.shape

        if key in loaded.keys():
            # print "loaded[key].shape", loaded[key].shape
            reshaped = loaded[key].reshape(layer[0].data.shape)
            print "loaded[key].shape after reshaping", reshaped.shape

            layer[0].data[...] = reshaped

            if key == fromLayer:
                # net.params[toLayer] = layer
                print layer
                layer2 = caffe._caffe.BlobVec(layer)
                print layer2
                print "Duplicating", fromLayer, "into", toLayer
            # print "loaded[key]", loaded[key]
            # layer[0] = loaded[key][:]
            # for blobIndex in range(len(layer)):
            #     blob = net.params[key][blobIndex]
            #     # print "\t", blob.data.shape, np.flatten(blob.data)[0:3]
            #     print "\t", blob.data.shape, blob.data.flatten()[0:3]
        else:
            print "Key", key, " from numpy save does not exist in caffemodel => ignored."
    outputPath = 'modifiedFCtoCVMatthieu.caffemodel'
    net.save(outputPath)
    print "Saved to", outputPath


def copyFCLayerToConvLayerFromNets(oldNet, newNet, fromLayer, toLayer):
    originalWeights = oldNet.params[fromLayer][0].data
    newLayer = newNet.params[toLayer]

    newWeights = originalWeights.reshape(newLayer[0].data.shape)
    newLayer[0].data[...] = newWeights

    outputPath = 'modifiedFCtoCVMatthieu_smarter.caffemodel'
    newNet.save(outputPath)
    print "Saved to", outputPath


def turn__FullyConnectedLayer_fc650__into__convolutionalLayer_FS650():
    oldNet = inspect('/home/wookjin/vggLauraFiles/VGG_FACE_FS_tmpMatthieu.prototxt',
            '/home/wookjin/vggLauraFiles/_iter_50113_converted.caffemodel')
    newNet = inspect('/home/wookjin/vggLauraFiles/VGG_FACE_FS_tmpMatthieuNew.prototxt',
            '/home/wookjin/vggLauraFiles/modifiedFCtoCVMatthieu.caffemodel')
    copyFCLayerToConvLayerFromNets(oldNet, newNet,
                     'fc6.50', 'FS6.50')

# inspect('/home/wookjin/vggLauraFiles/with100Features/VGG_FACE_cherryChocolateChip.prototxt',
#         '/home/wookjin/vggLauraFiles/with100Features/_iter_100.caffemodel')


if __name__ == '__main__':
    newNetSmarter = inspect('/home/wookjin/vggLauraFiles/VGG_FACE_FS_tmpMatthieuNew.prototxt',
            '/home/wookjin/vggLauraFiles/modifiedFCtoCVMatthieu_smarter.caffemodel')


# copyLayerToLayer('/home/wookjin/vggLauraFiles/VGG_FACE_FS_tmpMatthieu.prototxt',
#                  '/home/wookjin/vggLauraFiles/_iter_50113_converted.caffemodel',
#                  'fc6.50', 'FS6.50', (1, 1, 50, 2048))
# downgrade('/home/wookjin/vggLauraFiles/VGG_FACE_FS_50_rockyRoad.prototxt',
#         '/home/wookjin/vggLauraFiles/_iter_50113.caffemodel', True)
# downgrade('/home/wookjin/vggLauraFiles/with100Features/VGG_FACE_cherryChocolateChip.prototxt',
#           '/home/wookjin/vggLauraFiles/with100Features/_iter_100.caffemodel')

# pathToModel = '../../../vggLauraFiles/_iter_50113.caffemodel'
# pathToPrototxt = '../../../vggLauraFiles/ItWorks/VGG_FACE_cherryChocolateChip.prototxt'
# testsLoad()


# # Make sure that caffe is on the python path:
# # caffe3 with python
# caffe3_root = '/home/wookjin/caffe-with_py/'  # this file is expected to be in {caffe_root}/examples

# # caffe1 with python
# caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples
# caffe_root = caffe1_root  # this file is expected to be in {caffe_root}/examples

# import sys
# sys.path.insert(0, caffe_root + 'python')

# import caffe

# # Load the original network and extract the fully-connected layers' parameters.
# net = caffe.Net(caffe_root + 'examples/mnist/lenet_train_test.prototxt',
#                 caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel')
# # net.set_phase_test()
# params = ['fc6', 'fc7', 'fc8']
# # fc_params = {name: (weights, biases)}
# fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

# for fc in params:
#     print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

