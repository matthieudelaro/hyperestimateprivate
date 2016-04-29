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


def convRelu(bottom, name, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group,
                                name=name+"Conv")
    return conv, L.ReLU(conv, in_place=True, name=name+"ReLU")


def maxPool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


class ArchDef(object):
    """The parameters to estimate, the function to create blocks,
    objectives, ..."""
    def __init__(self, objectives, params):
        self.objectives = objectives
        self.params = params
        for key in self.params:
            self.params[key].name = key


    def createConvBlock(self, bottomLayer, id, paramValues):
        """Create a block (ex: conv conv pool), and return the top layer of it"""
        top = bottomLayer
        for i in range(paramValues["convLayersPerBlock"]):
            print(to_proto(top))
            conv, relu = convRelu(bottom=top,
                           name=str(id) + "_" + str(i) + "_",
                           ks=paramValues["kernelSize"],
                           nout=paramValues["featuresPerLayer"],
                           stride=paramValues["strideConv"],
                           pad=1
                           )
            top = relu
            print("top", top)
        top = maxPool(top, ks=2, stride=2)
        return top


class Model(object):
    """A version of the network/params, An attempt to train"""
    def __init__(self, archDef):
        pass


def Hyperestimate(params, archDef):
    """Iterate over values of hyperparams, and find the best."""
    pass

block = None

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

        block = archDef.createConvBlock(data, 0, {
            "featuresPerLayer": 64,
            "convLayersPerBlock": 2,
            "blocks": 3,
            "kernelSize": 3,
            "strideConv": 3,
            "stridePool": 2
            })
        print("blocks", block)
        print("blocks", to_proto(block))
        # print(to_proto(block))
        # self.assertEqual(len(self.elements)*8, self.bElements.len())
        # self.assertEqual(len(self.values)*8, self.bValues.len())



if __name__ == '__main__':
    unittest.main()
