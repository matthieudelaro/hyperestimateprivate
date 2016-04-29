import CaffeModel
from HelperClasses import Phase
import Prototxt


def determinePhases(self):
    phases = []

    # find the index of the layer of center of the network
    iBegin = None
    iCenter = None
    iEnd = None
    layers = self.wholeNetProto.layers
    # layers = CaffeModel.inspect(self.wholeNet, verbose=False)
    for i, layer in enumerate(layers):
        layerName = str(layer.name)
        if layerName == self.beginLayerName:
            iBegin = i
        elif layerName == self.centerLayerName:
            iCenter = i
        elif layerName == self.endLayerName:
            iEnd = i
    if iBegin is None or iCenter is None or iEnd is None:
        raise ("Could not find begin, center, or end layer",
               self.beginLayerName, self.centerLayerName, self.endLayerName)
    else:
        print "begin", iBegin, "center", iCenter, "end", iEnd

    iLeft = iCenter  # bottom of the expanding window
    iRight = iCenter+1  # top of the expanding window
    phaseCounter = 0
    while iLeft >= iBegin and iRight <= iEnd:  # while expanding window has not reach its limit
        # save last phase expansion window
        iLeftPrevious = iLeft
        iRightPrevious = iRight

        phase = Phase(phaseCounter)
        if phaseCounter != 0:
            phase.bridgeLayer = layers[iLeft]
            phase.bridgeToLayer = layers[iRight]

        # expand to create the next phase
        # explore leftward, ie bottomward
        while layers[iLeft].type != Prototxt.caffe_pb2.LayerParameter.CONVOLUTION:
            if iLeft < iBegin:
                raise Exception("Could not find a convolution layer while "
                                + "looking for one toward the bottom of the network.\n"
                                + str(phase))
            phase.layers.insert(0, layers[iLeft])  # push_front
            iLeft -= 1
        phase.layers.insert(0, layers[iLeft])  # push_front

        # explore rightward, ie topward
        while not ((iRight > iEnd)
            or (phase.targetLayer and layers[iRight+1].type == Prototxt.caffe_pb2.LayerParameter.UNPOOLING) \
            or (phase.targetLayer and layers[iRight+1].type == Prototxt.caffe_pb2.LayerParameter.DECONVOLUTION)):
            phase.layers.append(layers[iRight])
            if layers[iRight].type == Prototxt.caffe_pb2.LayerParameter.DECONVOLUTION:
                phase.targetLayer = layers[iRight]
            iRight += 1
        if iRight < iEnd:
            phase.layers.append(layers[iRight])
        if not phase.targetLayer:
            raise Exception("Could not find a deconvolution or pooling layer while "
                            + "looking for one toward the top of the network.\n"
                            + str(phase))

        # add the name of the blob which has to be fed to this phase
        phase.inputBlob = str(layers[iLeft-1].top[0])
        # if input blob is from a data layer, then add this data layer.
        # otherwise make up a data layer
        # phase.dataLayers = []  # often there are two of them: train and test
        for layer in layers:
            if layer.type == Prototxt.caffe_pb2.LayerParameter.DATA and \
               phase.inputBlob in layer.top:
                phase.dataLayers.append(layer)

        print phase

        # end of the phase creation
        phases.append(phase)
        iLeft -= 1
        iRight += 1
        phaseCounter += 1

        # if phaseCounter == 4:
    # divideBy0 = 0/0


    # botwardIndex = iCenter - 1
    # topwardIndex = iCenter + 1
    # while iBegin <= botwardIndex and topwardIndex <= iEnd:
    #     pass
    #     botwardIndex -= 1
    #     topwardIndex += 1

    # Mock until real version is done
    if False:
        phases = []

        # phase 1
        mockPhase = Phase()
        mockPhase.number = 0
        mockPhase.layers = []
        for l in layers:
            if l.name in ['FS6.50', 'relu6', 'FS6.50-deconv', 'FS6.50-deconv-relu']:
                mockPhase.layers.append(l)
        mockPhase.inputBlob = 'pool5'
        for l in layers:
            if l.name == 'FS6.50-deconv':
                mockPhase.targetLayer = l
                break
        mockPhase.bridgeToLayer = None
        phases.append(mockPhase)

        # phase 2
        mockPhase = Phase()
        mockPhase.number = 1
        mockPhase.layers = []
        for l in layers:
            if l.name in ['conv5_3', 'relu5_3', 'pool5', 'unpool5', 'deconv5_1', 'debn5_1', 'derelu5_1']:
                mockPhase.layers.append(l)
        mockPhase.inputBlob = 'conv5_2'
        for l in layers:
            if l.name == 'deconv5_1':
                mockPhase.targetLayer = l
                break
        mockPhase.bridgeBlob = "pool5"
        for l in layers:
            if l.name == 'pool5':
                mockPhase.bridgeLayer = l
                break
        for l in layers:
            if l.name == 'unpool5':
                mockPhase.bridgeToLayer = l
                mockPhase.bridgeReplaceBlob = "FS6.50-deconv"
                break
        phases.append(mockPhase)

    # print "mockPhase", mockPhase
    # print "mockPhase.layers", mockPhase.layers
    return phases


def typeOf(layerName):
    """Returns the type of the layer. Ex: 'conv5_3' => 'conv'"""
    pass
