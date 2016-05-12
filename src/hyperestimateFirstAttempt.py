from __future__ import print_function
# import utils
import unittest

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2


class Param(object):
    """Defines a hyper param to estimate"""
    def __init__(self, name, rangee, default):
        self.name = name
        self.rangee = rangee
        self.default = default

    # def __getitem__(self, key):
    #     if isinstance(key, slice):
    #         key.start
    #         key.step
    #         key.stop
    #     elif isinstance(key, int):
    #         if key < 0:  # Handle negative indices
    #         if key >= len(self):
    #             raise IndexError("The index (%d) is out of range." % key)
    #         return Bit.get(self._data, key)  # Get the data from elsewhere
    #     else:
    #         raise TypeError("Invalid argument type.")


class Objective(object):
    def __init__(self, pretrainError, trainLoss):
        self.pretrainError = pretrainError
        self.trainLoss = trainLoss


# class Block(object):
#     """Represents the layers in a block."""
#     def __init__(self, lowestLayer=None):
#         if lowestLayer is None:
#             self.layers = []
#         else:
#             self.layers = [lowestLayer]

#     def append(self, highestLayer):
#         self.highestLayer = highestLayer

#     def getHighest(self):
#         return self.highestLayer


def convRelu(bottom, name, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group,
                                name=name+"Conv")
    return conv, L.ReLU(conv, in_place=True, name=name+"ReLU")


def maxPool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def deconvRelu(bottom, name, ks, nout, stride=1, pad=0, group=1):
    conv = L.Deconvolution(bottom,
                                name=name+"Deconv")
    return conv, L.ReLU(conv, in_place=True, name=name+"ReLU")


def maxUnpool(bottom, ks, stride=1):
    return L.Unpooling(bottom, unpool=P.Unpooling.MAX, kernel_size=ks, stride=stride, unpool_size=8)


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

class TestBasic(unittest.TestCase):
    def setUp(self):
        # self.elements = [0, 1, 2, 4, 8, 16]
        # self.values = bytearray(self.elements)
        # self.bElements = BitOver(self.elements)
        # self.bValues = BitOver(self.values)
        pass

    def test_len(self):
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

        # data, label = L.Data(source="here", backend=P.Data.LMDB, batch_size=10, ntop=2,
            # transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))
        data = L.Data(source="here", backend=P.Data.LMDB, batch_size=10, ntop=1,
            transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))
        print("data", data)

        block = archDef.createEncoderBlock(data, 0, {
            "featuresPerLayer": 64,
            "convLayersPerBlock": 2,
            "blocks": 3,
            "kernelSize": 3,
            "strideConv": 3,
            "stridePool": 2
            })
        # print("blocks", block[-1])
        print("blocks", to_proto(block[-1]))
        # print(to_proto(block))
        # self.assertEqual(len(self.elements)*8, self.bElements.len())
        # self.assertEqual(len(self.values)*8, self.bValues.len())

        middle = L.Convolution(block[-1], kernel_size=4,
                                num_output=50,
                                name="middle")

        unblock = archDef.createDecoderBlock(middle, 0, block, {
            "featuresPerLayer": 64,
            "convLayersPerBlock": 2,
            "blocks": 3,
            "kernelSize": 3,
            "strideConv": 3,
            "stridePool": 2
            })
        print("unblock", to_proto(unblock[-1]))

        interactive["block"] = block
        interactive["unblock"] = unblock



if __name__ == '__main__':
    unittest.main()
