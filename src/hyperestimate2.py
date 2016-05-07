from __future__ import print_function
# import utils
import unittest

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2 as v2

import tempfile
import os


def getSGDSolver(solver):
    """Serialize them to files, and loads them from file,
    since PyCaffe doesn't export the constructor of SGDSolver
    from SolverParameter (see _caffe.cpp:306).

    This is a hack, using files instead of memory is sad.

    For some reason, there are errors while parsing when the file
    is opened in 'w+' mode. So the function opens files in 'w+' mode,
    writes, closes, opens again in 'r' mode, reads, and closes.

    Writes solver and network in a unique file."""

    try:
        # WRITE
        solverFileTemp = open("tmp_pycaffe_to_caffe_solver.sh", "w+")
        solverFileTemp.write(str(solver))
        solverFileTemp.close()

        # READ
        solverFileTemp = open("tmp_pycaffe_to_caffe_solver.sh", "r")
        sgdSolver = caffe.SGDSolver(solverFileTemp.name)
        solverFileTemp.close()

        return sgdSolver, sgdSolver.net
    except Error as e:
        print("Could not write/parse to/from file.", e.strerror)


def getSolverNet(solver, net):
    """Serialize them to files, and loads them from file,
    since PyCaffe doesn't export the constructor of SGDSolver
    from SolverParameter (see _caffe.cpp:306).

    This is a hack, using files instead of memory is sad.

    For some reason, there are errors while parsing when the file
    is opened in 'w+' mode. So the function opens files in 'w+' mode,
    writes, closes, opens again in 'r' mode, reads, and closes.

    Writes solver and network in two seperate files."""

    try:
        # WRITE
        solverFileTemp = open("tmp_pycaffe_to_caffe_solver.sh", "w+")
        netFileTemp = open("tmp_pycaffe_to_caffe_net.sh", "w+")
        netFileTemp.write(str(net))

        solver.net = netFileTemp.name
        solverFileTemp.write(str(solver))

        solverFileTemp.close()
        netFileTemp.close()

        # READ
        solverFileTemp = open("tmp_pycaffe_to_caffe_solver.sh", "r")
        netFileTemp = open("tmp_pycaffe_to_caffe_net.sh", "r")

        sgdSolver = caffe.SGDSolver(solverFileTemp.name)

        solverFileTemp.close()
        netFileTemp.close()

        return sgdSolver, sgdSolver.net
    except Error as e:
        print("Could not write/parse to/from file.", e.strerror)


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

    def createEncoderBlock(self, net, id, paramValues, outputMask=False):
        """Create a block (ex: conv conv pool), and return it as a list
        of layers."""
        layers = [net.layer[-1]]
        for i in range(paramValues["convLayersPerBlock"]):
            prefix = str(id) + "_" + str(i) + "_"
            layers.append(plug(layers[-1], conv(net.layer.add(),
                               prefix + "conv",
                               ks=paramValues["kernelSize"],
                               nout=paramValues["featuresPerLayer"],
                               stride=paramValues["strideConv"],
                               pad=1
                               )))
            layers.append(relu(layers[-1], net.layer.add()))

        prefix = str(id) + "_"
        layers.append(plug(layers[-1],
                           maxPool(net.layer.add(), prefix + "pool",
                                   ks=2,
                                   stride=paramValues["stridePool"],
                                   outputMask=outputMask)))
        layers.pop(0)
        return layers

    def createDecoderBlock(self, net, id, encoderBlock, paramValues):
        """Create a block (ex: unpool deconv decconv), and return it
         as a list of layers."""
        # def maxUnpool(layer, name, ks, stride=1):
        middleKernelSize = paramValues["inputSize"] / (2**paramValues["blocks"])
        unpool_size = middleKernelSize * (2**(paramValues["blocks"] - id))
        prefix = str(id) + "_"
        layers = [plug(net.layer[-1],
                       maxUnpool(net.layer.add(),
                                 name=prefix + "unpool",
                                 ks=2,
                                 unpool_size=unpool_size,
                                 stride=paramValues["stridePool"]))]
        plug(encoderBlock[-1], layers[-1])
        for i in range(paramValues["convLayersPerBlock"]-1, -1, -1):
            prefix = str(id) + "_" + str(i) + "_"
            layers.append(plug(layers[-1], conv(net.layer.add(),
                               prefix + "deconv",
                               ks=paramValues["kernelSize"],
                               nout=paramValues["featuresPerLayer"],
                               stride=paramValues["strideConv"],
                               pad=1
                               )))
            layers.append(relu(layers[-1], net.layer.add()))
        return layers


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
    if len(lowerLayer.top) > len(higherLayer.bottom):
        higherLayer.bottom.append(lowerLayer.top[len(higherLayer.bottom)])
    else:
        higherLayer.bottom.append(lowerLayer.top[-1])

    return higherLayer


def relu(lowerLayer, layer):
    layer.type = "ReLU"
    layer.name = lowerLayer.name + "_ReLU"
    layer.bottom.append(lowerLayer.top[0])
    layer.top.append(lowerLayer.top[0])
    return layer


def conv(layer, name, ks, nout, stride, pad=0):
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
    layer.convolution_param.stride.append(stride)
    layer.convolution_param.pad.append(pad)
    layer.convolution_param.kernel_size.append(ks)
    layer.convolution_param.weight_filler.type = "xavier"
    layer.convolution_param.bias_filler.type = "constant"
    layer.convolution_param.bias_filler.value = 0
    return layer


def deconv(layer, name, ks, nout, stride, pad=0):
    layer.type = "Deconvolution"
    layer.name = name
    layer.top.append(name)

    paramWeight = layer.param.add()
    paramWeight.lr_mult = 1
    paramWeight.decay_mult = 0

    paramBias = layer.param.add()
    paramBias.lr_mult = 2
    paramBias.decay_mult = 0

    layer.convolution_param.num_output = nout
    layer.convolution_param.stride.append(stride)
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


def maxUnpool(layer, name, ks, unpool_size, stride=1):
    layer.type = "Unpooling"
    layer.name = name
    layer.top.append(name)
    layer.unpooling_param.unpool = v2.PoolingParameter.MAX
    layer.unpooling_param.kernel_size = ks
    layer.unpooling_param.stride = stride
    layer.unpooling_param.unpool_size = unpool_size
    return layer


def fullyConnected(layer, name, nout):
    layer.type = "InnerProduct"
    layer.name = name
    layer.top.append(name)

    paramWeight = layer.param.add()
    paramWeight.lr_mult = 1
    paramWeight.decay_mult = 0

    paramBias = layer.param.add()
    paramBias.lr_mult = 2
    paramBias.decay_mult = 0

    layer.inner_product_param.num_output = nout
    layer.convolution_param.weight_filler.type = "xavier"
    layer.convolution_param.bias_filler.type = "constant"
    layer.convolution_param.bias_filler.value = 0
    return layer


def dropout(lowerLayer, layer, ratio):
    layer.type = "Dropout"
    layer.name = lowerLayer.name + "_Dropout"
    layer.bottom.append(lowerLayer.top[0])
    layer.top.append(lowerLayer.top[0])

    layer.dropout_param.dropout_ratio = ratio
    return layer


def locallyConnected(layer, name, ks, nout, stride, pad=0):
    layer.type = "Local"
    layer.name = name
    layer.top.append(name)

    paramWeight = layer.param.add()
    paramWeight.lr_mult = 1
    paramWeight.decay_mult = 0

    paramBias = layer.param.add()
    paramBias.lr_mult = 2
    paramBias.decay_mult = 0

    layer.local_param.num_output = nout
    layer.local_param.stride = stride
    layer.local_param.pad = pad
    layer.local_param.kernel_size = ks
    layer.local_param.weight_filler.type = "xavier"
    layer.local_param.bias_filler.type = "constant"
    layer.local_param.bias_filler.value = 0
    return layer


def softmax(layer, name="softmax"):
    layer.type = "SoftmaxWithLoss"
    layer.name = name
    layer.top.append(name)
    return layer


def accuracy(layer, top_k, name=""):
    if name == "":
        name = "accuracy_top_" + str(top_k)
    layer.type = "Accuracy"
    layer.name = name
    layer.top.append(name)
    layer.accuracy_param.top_k = top_k
    return layer


def euclideanLoss(layer, name="L2_Loss"):
    layer.type = "EuclideanLoss"
    layer.name = name
    layer.top.append(name)
    return layer



class TestBasic(unittest.TestCase):
    def setUp(self):
        # self.elements = [0, 1, 2, 4, 8, 16]
        # self.values = bytearray(self.elements)
        # self.bElements = BitOver(self.elements)
        # self.bValues = BitOver(self.values)
        pass

    def previousReconstruct(self):
        caffe.set_mode_cpu()

        objectives = Objective(0.4, 500000)
        params = {
            "featuresPerLayer": Param("", slice(4, 64, 10), 64),
            "convLayersPerBlock": Param("", slice(1, 5, 1), 2),
            "blocks": Param("", slice(1, 5, 1), 3),
            "kernelSize": Param("", slice(1, 5, 1), 3),
            "kernelSizeLocal": Param("", slice(1, 5, 1), 1),
            "strideConv": Param("", slice(1, 1, 1), 1),
            "stridePool": Param("", slice(1, 5, 1), 3),
            "inputSize": Param("", slice(32, 32, 1), 32)
            }
        print(params)
        archDef = ArchDef(objectives, params)


        solver_param = v2.SolverParameter()
        # solver_param.test_iter.append(10)
        # solver_param.test_interval = 50
        solver_param.base_lr = 1e-08
        solver_param.display = 1
        solver_param.max_iter = 50
        solver_param.lr_policy = "fixed"
        solver_param.momentum = 0
        solver_param.weight_decay = 0.004
        solver_param.snapshot = 200
        solver_param.snapshot_prefix = "snapshots/reconstructing_full"
        solver_param.solver_mode = solver_param.CPU
        net = solver_param.net_param


        # net = caffe.proto.caffe_pb2.NetParameter()
        dataTest = testPhase(dataLayer(net.layer.add(), tops=["data"],
                    sourcePath="/dataset/cifar100_lmdb_lab/cifar100_test_lmdb",
                    meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto"))
        dataTrain = trainPhase(dataLayer(net.layer.add(), tops=["data"],
                    sourcePath="/dataset/cifar100_lmdb_lab/cifar100_train_lmdb",
                    meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto"))
        # top = relu(plug(dataTrain, conv(net.layer.add(), "conv1", ks=2, nout=50)), net.layer.add())
        # top = plug(top, maxPool(net.layer.add(), "pool1", ks=2))

        settings = {
            "featuresPerLayer": 64,
            "convLayersPerBlock": 2,
            "blocks": 3,
            "kernelSize": 3,
            "kernelSizeLocal": 1,
            "strideConv": 1,
            "stridePool": 2,
            "inputSize": 32
            }

        blocks = []
        for i in range(settings["blocks"]):
            block = archDef.createEncoderBlock(net, i, settings, outputMask=True)
            blocks.append(block)

        middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
        middleConv = plug(blocks[-1][-1], conv(net.layer.add(),
                                      name="middle_conv",
                                      ks=middleKernelSize,
                                      nout=50,
                                      stride=1
                                      ))
        middleDeconv = plug(middleConv, deconv(net.layer.add(),
                                      name="middle_deconv",
                                      ks=middleKernelSize,
                                      nout=settings["featuresPerLayer"],
                                      stride=settings["strideConv"]
                                      ))
        # print("blocks", block[-1])
        # print("blocks", to_proto(block[-1])) # <<
        # print(to_proto(block))
        # self.assertEqual(len(self.elements)*8, self.bElements.len())
        # self.assertEqual(len(self.values)*8, self.bValues.len())

        # middle = L.Convolution(block[-1], kernel_size=4,
                                # num_output=50,
                                # name="middle")

        unblocks = []
        for i in range(settings["blocks"]-1, -1, -1):
            unblock = archDef.createDecoderBlock(net, i, blocks[i], settings)
            unblocks.append(unblock)
        # unblock = archDef.createDecoderBlock(net, 0, block, settings)
        top = plug(unblocks[-1][-1], locallyConnected(net.layer.add(),
                                      name="reconstruct1",
                                      ks=settings["kernelSizeLocal"],
                                      nout=3,
                                      stride=1
                                      ))
        top = plug(top, locallyConnected(net.layer.add(),
                                      name="reconstruct2",
                                      ks=settings["kernelSizeLocal"],
                                      nout=3,
                                      stride=1
                                      ))
        top = plug(top, plug(dataTrain, euclideanLoss(net.layer.add())))

        print("net\n", net)
        # print("unblock", to_proto(unblock[-1])) # <<

        interactive["blocks"] = blocks
        interactive["unblocks"] = unblocks
        interactive["net"] = net

        print("blocks:")
        for b in blocks:
            for l in b:
                print(l.name)
            print()
        print("unblocks:")
        for b in unblocks:
            print(b[0].unpooling_param.unpool_size)
            for l in b:
                print(l.name)
            print()

        # solver_param = caffe.SGDSolver(solver_param)
        # net.state.phase = v2.TRAIN
        # [solver, net] = getSolverNet(solver_param, net)
        [solver, net] = getSGDSolver(solver_param)
        solver.step(5)

    def testClassify(self):
        caffe.set_mode_cpu()

        objectives = Objective(0.4, 500000)
        params = {
            "featuresPerLayer": Param("", slice(4, 64, 10), 64),
            "convLayersPerBlock": Param("", slice(1, 5, 1), 2),
            "blocks": Param("", slice(1, 5, 1), 3),
            "kernelSize": Param("", slice(1, 5, 1), 3),
            "kernelSizeLocal": Param("", slice(1, 5, 1), 1),
            "strideConv": Param("", slice(1, 1, 1), 1),
            "stridePool": Param("", slice(1, 5, 1), 3),
            "inputSize": Param("", slice(32, 32, 1), 32)
            }
        print(params)
        archDef = ArchDef(objectives, params)


        solver_param = v2.SolverParameter()
        # solver_param.test_iter.append(10)
        # solver_param.test_interval = 50
        solver_param.base_lr = 1
        solver_param.display = 1
        solver_param.max_iter = 50
        solver_param.lr_policy = "fixed"
        solver_param.momentum = 0.9
        solver_param.weight_decay = 0.004
        solver_param.snapshot = 200
        solver_param.snapshot_prefix = "snapshots/classify"
        solver_param.solver_mode = solver_param.CPU
        net = solver_param.net_param


        # net = caffe.proto.caffe_pb2.NetParameter()
        dataTest = testPhase(dataLayer(net.layer.add(), tops=["data", "label"],
                    sourcePath="/dataset/cifar100_lmdb_lab/cifar100_test_lmdb",
                    meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto"))
        dataTrain = trainPhase(dataLayer(net.layer.add(), tops=["data", "label"],
                    sourcePath="/dataset/cifar100_lmdb_lab/cifar100_train_lmdb",
                    meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto"))
        # top = relu(plug(dataTrain, conv(net.layer.add(), "conv1", ks=2, nout=50)), net.layer.add())
        # top = plug(top, maxPool(net.layer.add(), "pool1", ks=2))

        settings = {
            "featuresPerLayer": 64,
            "convLayersPerBlock": 2,
            "blocks": 3,
            "kernelSize": 3,
            "kernelSizeLocal": 1,
            "strideConv": 1,
            "stridePool": 2,
            "inputSize": 32
            }

        blocks = []
        for i in range(settings["blocks"]):
            block = archDef.createEncoderBlock(net, i, settings, outputMask=False)
            blocks.append(block)

        middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
        middleConv = plug(blocks[-1][-1], conv(net.layer.add(),
                                      name="middle_conv",
                                      ks=middleKernelSize,
                                      nout=50,
                                      stride=1
                                      ))

        top = plug(middleConv, fullyConnected(net.layer.add(), name="fc1", nout=1024))
        top = trainPhase(dropout(top, net.layer.add(), ratio=0.5))
        top = relu(top, net.layer.add())

        top = plug(top, fullyConnected(net.layer.add(), name="fc2", nout=100))

        top = plug(dataTrain, plug(top, softmax(net.layer.add())))

        plug(dataTest, plug(top, testPhase(accuracy(net.layer.add(), 1))))
        plug(dataTest, plug(top, testPhase(accuracy(net.layer.add(), 5))))

        print("net\n", net)
        # print("unblock", to_proto(unblock[-1])) # <<

        interactive["blocks"] = blocks
        interactive["net"] = net

        print("blocks:")
        for b in blocks:
            for l in b:
                print(l.name)
            print()

        [solver, net] = getSGDSolver(solver_param)
        solver.step(5)



if __name__ == '__main__':
    unittest.main()
