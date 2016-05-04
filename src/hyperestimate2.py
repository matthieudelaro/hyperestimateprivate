from __future__ import print_function
# import utils
import unittest

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2 as v2


class Param(object):
    """Defines a hyper param to estimate"""
    def __init__(self, name, rangee, default):
        self.name = name
        self.rangee = rangee
        self.default = default


class Objective(object):
    def __init__(self, pretrainError, trainLoss):
        self.pretrainError = pretrainError
        self.trainLoss = trainLoss


class ArchDef(object):
    """The parameters to estimate, the function to create blocks,
    objectives, ..."""
    def __init__(self, objectives, params):
        self.objectives = objectives
        self.params = params
        for key in self.params:
            self.params[key].name = key


    def createEncoderBlock(self, bottomLayer, id, paramValues):
        # """Create a block (ex: conv conv pool), and return the top layer of it"""
        # """Create a block (ex: conv conv pool), and return it as a Block."""
        """Create a block (ex: conv conv pool), and return it as a list of layers."""
        top = bottomLayer
        layers = []
        # block = None
        for i in range(paramValues["convLayersPerBlock"]):
            # print(to_proto(top))
            conv, relu = convRelu(bottom=top,
                           name=str(id) + "_" + str(i) + "_",
                           ks=paramValues["kernelSize"],
                           nout=paramValues["featuresPerLayer"],
                           stride=paramValues["strideConv"],
                           pad=1
                           )
            layers.append(conv)
            layers.append(relu)
            top = relu
            # print("top", top)
            # if block is None:
                # block = Block(conv, None)
        # top = maxPool(top, ks=2, stride=2)
        layers.append(maxPool(top, ks=2, stride=2))
        return layers

        # block.setHighest(top)
        # return block

    def createDecoderBlock(self, bottomLayer, id, encoderBlock, paramValues):
        # """Create a block (ex: conv conv pool), and return the top layer of it"""
        # """Create a block (ex: conv conv pool), and return it as a Block."""
        """Create a block (ex: unpool deconv decconv), and return it as a list of layers."""
        top = bottomLayer
        # layers = [maxUnpool([bottomLayer, encoderBlock[-1]], ks=2, stride=2)]
        layers = [maxUnpool(bottomLayer, ks=2, stride=2)]
        # block = None
        for i in range(paramValues["convLayersPerBlock"]):
            # print(to_proto(top))
            deconv, relu = deconvRelu(bottom=layers[-1],
                           name=str(id) + "_" + str(i) + "_",
                           ks=paramValues["kernelSize"],
                           nout=paramValues["featuresPerLayer"],
                           stride=paramValues["strideConv"],
                           pad=1
                           )
            layers.append(deconv)
            layers.append(relu)
            # top = relu
            # print("top", top)
            # if block is None:
                # block = Block(conv, None)
        # top = maxPool(top, ks=2, stride=2)
        # layers.append(maxPool(top, ks=2, stride=2))
        return layers

        # block.setHighest(top)
        # return block


class Model(object):
    """A version of the network/params, An attempt to train"""
    def __init__(self, archDef):
        pass


def Hyperestimate(params, archDef):
    """Iterate over values of hyperparams, and find the best."""
    pass

interactive = {}


def dataLayer(layer, tops, sourcePath, meanFilePath):
    # data = net.layer.add()
    layer.name = "data"
    layer.type = "Data"
    layer.data_param.source = sourcePath
    layer.data_param.backend = caffe.proto.caffe_pb2.DataParameter.LMDB
    layer.data_param.batch_size = 64
    layer.transform_param.mean_file = meanFilePath

    for top in tops:
        layer.top.append(top)

    return layer


def testPhase(layer):
    include = layer.include.add()
    include.phase = v2.TEST
    return layer


def trainPhase(layer):
    include = layer.include.add()
    include.phase = v2.TRAIN
    return layer


def plug(lowerLayer, higherLayer):
    higherLayer.bottom.append(lowerLayer.top[len(higherLayer.bottom)])
    return higherLayer


def relu(lowerLayer, layer):
    layer.type = "ReLU"
    layer.name = lowerLayer.name + "_ReLU"
    layer.bottom.append(lowerLayer.top[0])
    layer.top.append(lowerLayer.top[0])
    return layer


def conv(layer, name, ks, nout, pad=0):
    layer.type = "Convolution"
    layer.name = name
    layer.top.append(name)

    paramWeight = layer.param.add()
    paramWeight.lr_mult = 1
    paramWeight.decay_mult = 0

    paramBias = layer.param.add()
    paramBias.lr_mult = 2
    paramBias.decay_mult = 0

    layer.convolution_param.num_output = nout
    layer.convolution_param.pad.append(pad)
    layer.convolution_param.kernel_size.append(ks)
    layer.convolution_param.weight_filler.type = "xavier"
    layer.convolution_param.bias_filler.type = "constant"
    layer.convolution_param.bias_filler.value = 0
    return layer


def maxPool(layer, name, ks, stride=1, outputMask=False):
    layer.type = "Pooling"
    layer.name = name
    layer.top.append(name)
    if outputMask is True:
        layer.top.append(name + "_mask")
    layer.pooling_param.pool = v2.PoolingParameter.MAX
    layer.pooling_param.kernel_size = ks
    layer.pooling_param.stride = stride
    return layer



class TestBasic(unittest.TestCase):
    def setUp(self):
        # self.elements = [0, 1, 2, 4, 8, 16]
        # self.values = bytearray(self.elements)
        # self.bElements = BitOver(self.elements)
        # self.bValues = BitOver(self.values)
        pass

    def test2(self):
        objectives = Objective(0.4, 500000)
        params = {
            "featuresPerLayer": Param("", slice(4, 64, 10), 64),
            "convLayersPerBlock": Param("", slice(1, 5, 1), 2),
            "blocks": Param("", slice(1, 5, 1), 3),
            "kernelSize": Param("", slice(1, 5, 1), 3),
            "strideConv": Param("", slice(1, 1, 1), 3),
            "stridePool": Param("", slice(1, 5, 1), 3)
            }
        print(params)
        archDef = ArchDef(objectives, params)

        net = caffe.proto.caffe_pb2.NetParameter()
        data = testPhase(dataLayer(net.layer.add(), tops=["data", "labels"],
                         sourcePath="here", meanFilePath="here2"))
        top = relu(plug(data, conv(net.layer.add(), "conv1", ks=2, nout=50)), net.layer.add())
        top = plug(top, maxPool(net.layer.add(), "pool1", ks=2))
        # top = plug(top, relu(net.layer.add(), top))
        print("net\n", net)

        # block = archDef.createEncoderBlock(data, 0, {
        #     "featuresPerLayer": 64,
        #     "convLayersPerBlock": 2,
        #     "blocks": 3,
        #     "kernelSize": 3,
        #     "strideConv": 3,
        #     "stridePool": 2
        #     })
        # print("blocks", block[-1])
        # print("blocks", to_proto(block[-1])) # <<
        # print(to_proto(block))
        # self.assertEqual(len(self.elements)*8, self.bElements.len())
        # self.assertEqual(len(self.values)*8, self.bValues.len())

        # middle = L.Convolution(block[-1], kernel_size=4,
                                # num_output=50,
                                # name="middle")

        # unblock = archDef.createDecoderBlock(middle, 0, block, {
        #     "featuresPerLayer": 64,
        #     "convLayersPerBlock": 2,
        #     "blocks": 3,
        #     "kernelSize": 3,
        #     "strideConv": 3,
        #     "stridePool": 2
        #     })
        # print("unblock", to_proto(unblock[-1])) # <<

        # interactive["block"] = block
        # interactive["unblock"] = unblock
        interactive["net"] = net



if __name__ == '__main__':
    unittest.main()
