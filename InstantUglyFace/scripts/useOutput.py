# caffe1 with python
caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples
caffe_root = caffe1_root  # this file is expected to be in {caffe_root}/examples


import numpy as np
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import caffe.draw

try:
    from PIL import Image
except:
    from pillow import Image


def forwardNet(net, inputImagePath, outputImagePath):
    """Load input image from disk, performs a forward pass on the network,
    and saves the output of the network to the specified outputImagePath."""
    # inputImage = np.array(Image.open(inputImagePath))
    inputImage = caffe.io.load_image(inputImagePath)
    # print inputImage
    # outputImage = net.forward_all(data=inputImage)

    # net_full_conv.set_mean('data', np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    # net.set_channel_swap('data', (2,1,0))
    # net_full_conv.set_raw_scale('data', 255.0)
    # make classification map by forward and print prediction indices at each location
    # out = net.forward_all(data=np.asarray([inputImage]))
    # out = net.forward_all(data=np.asarray([inputImage]).reshape((1, 3, 60, 60)))
    # output_blobs = [np.empty((1, 1, 1, 1), dtype=np.float32)]

    # out = net.forward_all(data=np.asarray([net.preprocess('data', inputImage)]))  # runs. Not desired output

    # output_blobs = []
    # out = net.forward_all(data=np.asarray([net.preprocess('data', inputImage)]), blobs=output_blobs)
    # print output_blobs

    out = net.forward(end="fc8.1")
    # out = net.forward(data=np.asarray([net.preprocess('data', np.asarray(inputImage*100).reshape((100, 3, 60, 60)) )]), end="fc8.1")
    # out = net.forward(data=np.asarray([net.preprocess('data', np.asarray(inputImage).reshape((1, 3, 60, 60)) )]), end="fc8.1")
    print "out.keys()", out.keys()
    print "out['fc8.1'].shape", out['fc8.1'].shape
    print "out.keys()", out['fc8.1'][0].shape
    # outputImage = net.forward_all(data=np.asarray([inputImage])
    # print outputImage


def forward(net, inputImagePath, outputImagePath):
    """Load input image from disk, performs a forward pass on the network,
    and saves the output of the network to the specified outputImagePath."""
    # inputImage = np.array(Image.open(inputImagePath))
    inputImage = caffe.io.load_image(inputImagePath)
    outputImage = net.predict([inputImage])
    print outputImage


def drawToFile(net, outputImagePath):
    net.name = "Hello Net"
    caffe.draw.draw_net_to_file(net, outputImagePath)


def dev():
    pathToPrototxt = '/home/wookjin/vggLauraFiles/with100Features/VGG_FACE_cherryChocolateChip.prototxt'
    pathToCaffemodel = '/home/wookjin/vggLauraFiles/with100Features/_iter_100.caffemodel'
    net = caffe.Net(pathToPrototxt, pathToCaffemodel)
    # net = caffe.Classifier(pathToPrototxt, pathToCaffemodel)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_device(0)
    # forwardNet(net, "kitty.png", "output.png")
    drawToFile(net, "graph_of_the_network.png")
    return net

dev()
print "Done"
