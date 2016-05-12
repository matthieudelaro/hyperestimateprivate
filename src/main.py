from __future__ import print_function
# import utils
import unittest

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2 as v2

import os
from prototype import *

USE_GPU = False


def main():
    if USE_GPU:
        caffe.set_mode_gpu()
    else:
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
    archDef = ArchDef(objectives, params)

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

    trainArchitecture("0", archDef, settings)


def trainArchitecture(ID, archDef, settings):
    weightsFilePath = pretrainClassification(ID, archDef, settings)
    print("pretraining done: ", weightsFilePath)
    weightsFilePath = trainReconstruct(ID, archDef, settings, weightsFilePath)
    print("training done: ", weightsFilePath)


def pretrainClassification(ID, archDef, settings):
    # create solver
    solver_param = v2.SolverParameter()
    solver_param.base_lr = 1
    solver_param.display = 1
    solver_param.max_iter = 50
    solver_param.lr_policy = "fixed"
    solver_param.momentum = 0.5
    solver_param.weight_decay = 0.004
    solver_param.snapshot = 200
    solver_param.snapshot_prefix = "snapshots/" + ID + "_classify"
    # if USE_GPU:
    #     solver_param.solver_mode = solver_param.GPU
    # else:
    #     solver_param.solver_mode = solver_param.CPU

    # create network
    net_param = caffe.proto.caffe_pb2.NetParameter()

    (dataTrain, dataTest) = dataLayers(net_param)

    # create the conv pool blocks
    blocks = []
    for i in range(settings["blocks"]):
        block = archDef.createEncoderBlock(net_param, i, settings, outputMask=False)
        blocks.append(block)

    # create the middle layer
    middleKernelSize = settings["inputSize"] / (2**settings["blocks"])
    middleConv = plug(blocks[-1][-1], conv(net_param.layer.add(),
                                  name="middle_conv",
                                  ks=middleKernelSize,
                                  nout=1024,
                                  stride=1
                                  ))

    # create additional fully connected layer to classify
    top = plug(middleConv, fullyConnected(net_param.layer.add(), name="fc1", nout=1024))
    top = trainPhase(dropout(top, net_param.layer.add(), ratio=0.5))
    top = relu(top, net_param.layer.add())

    top = plug(top, fullyConnected(net_param.layer.add(), name="fc2", nout=100))

    top = plug(dataTrain, plug(top, softmax(net_param.layer.add())))

    plug(dataTest, plug(top, testPhase(accuracy(net_param.layer.add(), 1))))
    plug(dataTest, plug(top, testPhase(accuracy(net_param.layer.add(), 5))))

    (solverFileName, netFileName) = saveToFiles(ID + "_classify", solver_param, net_param)
    [solver, net] = getSolverNet(solver_param, net_param)

    # train
    iterations = 1000
    solver.step(iterations)

    # save
    weightsFilePath = "snapshots/" + ID + "_classify_" + str(iterations)
    net.save(weightsFilePath)
    return weightsFilePath


def trainReconstruct(ID, archDef, settings, givenWeightsFilePath):
    pass


def dataLayers(net_param):
    dataTest = testPhase(dataLayer(net_param.layer.add(), tops=["data", "label"],
                sourcePath="/dataset/cifar100_lmdb_lab/cifar100_test_lmdb",
                meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto"))
    dataTrain = trainPhase(dataLayer(net_param.layer.add(), tops=["data", "label"],
                sourcePath="/dataset/cifar100_lmdb_lab/cifar100_train_lmdb",
                meanFilePath="/dataset/cifar100_lmdb_lab/mean.binaryproto"))
    return (dataTest, dataTrain)

if __name__ == '__main__':
    main()

