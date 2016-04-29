
class NetInfo(object):
    def __init__(self, prototxt="", model="", solver=""):
        self.prototxt = prototxt
        self.model = model
        self.solver = solver


class Phase(object):
    def __init__(self, number, targetLayer=None, bridgeLayer=None, bridgeToLayer=None, layers=None, inputBlob=None):
        self.number = number
        self.targetLayer = targetLayer
        self.bridgeLayer = bridgeLayer
        self.bridgeToLayer = bridgeToLayer
        if not layers:
            self.layers = []
        else:
            self.layers = layers
        self.inputBlob = inputBlob
        self.dataLayers = []

    def __str__(self):
        out = ""
        out += "Phase " + str(self.number)
        out += "\n\ttarget layer:"
        if self.targetLayer:
            out += self.targetLayer.name
        out += "\n\timplying layers:" + str([str(l.name) for l in self.layers])
        out += "\n\tinput blob:" + str(self.inputBlob)
        if len(self.dataLayers) > 0:
            out += "\n\tinput layers:" + str([str(l.name) for l in self.dataLayers])
        else:
            out += "\n\tinput layers: None. Create fake ones from lmdbs"

        out += "\n\tbridge:"
        if self.bridgeLayer and self.bridgeToLayer:
            out += str(self.bridgeLayer.name) + " to " + str(self.bridgeToLayer.name)
        else:
            out += "None"
        return out

    pass
