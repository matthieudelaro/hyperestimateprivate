import Prototxt


def inflateDatalayer(self, phase, dataLayer, train=False, test=False):
    dataLayer.type = Prototxt.caffe_pb2.LayerParameter.DATA
    dataLayer.name = "data_to_train_" + phase.targetLayer.name
    # dataLayer.name = phase.inputBlob  # Generates error when the network runs loading pretrained weights. Example : Check failed: target_blobs.size() == source_layer.blobs_size() (0 vs. 2) Incompatible number of blobs for layer conv5_2
    # dataLayer.top.append(dataLayer.name)
    dataLayer.top.append(phase.inputBlob)
    dataLayer.data_param.source = self.featureMapsPrefix + phase.inputBlob
    dataLayer.data_param.backend = Prototxt.caffe_pb2.DataParameter.LMDB
    dataLayer.data_param.batch_size = 100
    if train:
        dataLayer.include.add().phase = Prototxt.caffe_pb2.TRAIN
    if test:
        dataLayer.include.add().phase = Prototxt.caffe_pb2.TEST
    return dataLayer


def generatePrototxtSolver(self, phase):
    prototxtProto = Prototxt.caffe_pb2.NetParameter()

    prototxtProto.name = "uno_training_" + str(phase.number) + \
                         "_to_train_" + phase.targetLayer.name

    # add data layers
    # if input blob is from a data layer, then add this data layer.
    # otherwise make up a data layer
    if len(phase.dataLayers) > 0:
        # TODO : make it flexible as it is meant to be!
        dataLayerTrain = prototxtProto.layers.add()
        dataLayerTrain.CopyFrom(phase.dataLayers[0])
        # dataLayerTest = inflateDatalayer(self, phase, prototxtProto.layers.add(), test=True)
    else:
        dataLayerTrain = inflateDatalayer(self, phase, prototxtProto.layers.add(), train=True)
        dataLayerTest = inflateDatalayer(self, phase, prototxtProto.layers.add(), test=True)

    # add layers
    layerToBeDestinationOfBridge = None
    for oldLayer in phase.layers:
        newLayer = prototxtProto.layers.add()
        newLayer.CopyFrom(oldLayer)

        if oldLayer == phase.bridgeToLayer:
            layerToBeDestinationOfBridge = newLayer


        # TODO : modify weight initialization to xavier
        # if oldLayer == phase.targetLayer:
            # weight_filler = Prototxt.caffe_pb2.ConvolutionParameter.weight_filler
            # weight_filler.type = "xavier"
            # newLayer.convolution_param.weight_filler = weight_filler


            # newLayer.convolution_param.weight_filler.type = "xavier"
            # newLayer.convolution_param.weight_filler.std.remove() = None
        if oldLayer != phase.targetLayer:  # TODO ?
            newLayer.blobs_lr = [0, 0]  # for weights
            # newLayer.blobs_lr.append(0)  # for biases

    # bridge layers
    if layerToBeDestinationOfBridge:
        # get rid of original bottom blob
        while len(layerToBeDestinationOfBridge.bottom) > 0:
            print "removing bottom arg", layerToBeDestinationOfBridge.bottom[0], "from layer", str(layerToBeDestinationOfBridge.name)
            del layerToBeDestinationOfBridge.bottom[0]
        # bridge outputs of layer to inputs of other layer
        for top in phase.bridgeLayer.top:
            print "adding   bottom arg", top, "to layer", str(layerToBeDestinationOfBridge.name)
            layerToBeDestinationOfBridge.bottom.append(top)

    # add loss layer
    lossLayer = prototxtProto.layers.add()
    lossLayer.type = Prototxt.caffe_pb2.LayerParameter.EUCLIDEAN_LOSS
    lossLayer.name = "loss"
    lossLayer.top.append(lossLayer.name)
    lossLayer.bottom.append(phase.layers[-1].top[-1])
    lossLayer.bottom.append(dataLayerTrain.top[-1])

    # generate solver
    solverProto = Prototxt.caffe_pb2.SolverParameter()
    solverProto.net = self._getPhaseProtoName(phase)
    # solverProto.test_iter.append(1000)
    # solverProto.test_interval = 10000
    solverProto.base_lr = 1e-7
    solverProto.lr_policy = "step"  # TODO: find and use enum instead of using hard coded string
    solverProto.gamma = 0.1
    solverProto.stepsize = 5000
    solverProto.display = 10
    solverProto.max_iter = 100000
    solverProto.momentum = 0.9
    solverProto.weight_decay = 0.0005
    solverProto.solver_mode = Prototxt.caffe_pb2.SolverParameter.GPU
    solverProto.snapshot_prefix = self._getPhaseSnapshotsPrefix(phase)
    solverProto.snapshot = 10000

    # print "\nprototxtProto:\n", prototxtProto
    # print "\nsolverProto:\n", solverProto

    return prototxtProto, solverProto


def quickTests():
    p, s = generatePrototxtSolver(None, None)
    return p, s

if __name__ == '__main__':
    p, s = quickTests()
