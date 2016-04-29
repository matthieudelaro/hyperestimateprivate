import CaffeModel
import UnoTraining_generatePrototxtSolver
import UnoTraining_determinePhases
from HelperClasses import NetInfo
import Prototxt
import os.path
import pprint

pp = pprint.PrettyPrinter(indent=4)


class UnoTraining(object):
    def save(self, pathPrefix):
        """saves itself to given path."""
        pass

    def loadFrom(path):
        """Resume UnoTraining from file."""
        pass

    def resume(self, debug=False, verbose=True):
        self._phaseLoop(debug=debug, verbose=verbose)

    def _getPhaseFolderName(self, phase):
        return os.path.join(self.outputFolder, "phase_" + str(phase.number) + "_for_" + str(phase.targetLayer.name))

    def _getPhaseProtoName(self, phase):
        return os.path.join(self._getPhaseFolderName(phase), "network.prototxt")

    def _getPhaseSolverName(self, phase):
        return os.path.join(self._getPhaseFolderName(phase), "solver.prototxt")

    def _getPhaseSnapshotsPrefix(self, phase):
        return os.path.join(self._getPhaseFolderName(phase), "_")

    def init(self, wholeNetInfo, beginLayer, centerLayer, endLayer, outputFolder, featureMapsPrefix):
        self.wholeNetInfo = wholeNetInfo
        self.wholeNetProto = Prototxt.readPrototxtFile(self.wholeNetInfo.prototxt)
        self.beginLayerName = beginLayer
        self.centerLayerName = centerLayer
        self.endLayerName = endLayer
        self.outputFolder = outputFolder
        self.featureMapsPrefix = featureMapsPrefix

    def train(self, debug=False, verbose=True):
        self.phases = self._determinePhases()
        self._phaseLoop(debug=debug, verbose=verbose)

    def _phaseLoop(self, debug=False, verbose=True):
        # wholeNet = CaffeModel.load(wholeNetInfo.prototxt, wholeNetInfo.model, verbose=False)
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        wholeNet = None  # TODO
        for phase in self.phases:
            self._performPhase(wholeNet, phase)
        pass

    def _performPhase(self, wholeNet, phase, debug=False, verbose=True):
        print "Generating files for phase ", phase.number
        print "target layer:", phase.targetLayer.name
        print "implying layers:",
        pp.pprint([str(l.name) for l in phase.layers])
        prototxtProto, solverProto = self._generatePrototxtSolver(phase)

        if not os.path.exists(self._getPhaseFolderName(phase)):
            os.makedirs(self._getPhaseFolderName(phase))
        with open(self._getPhaseProtoName(phase), "w+") as outputFile:
            outputFile.write(str(prototxtProto))
        with open(self._getPhaseSolverName(phase), "w+") as outputFile:
            outputFile.write(str(solverProto))
        print "Files saved to", self._getPhaseFolderName(phase)
        print ""

    def _generatePrototxtSolver(self, phase):
        return UnoTraining_generatePrototxtSolver.generatePrototxtSolver(self, phase)

    def _determinePhases(self):
        return UnoTraining_determinePhases.determinePhases(self)


def quickTests():
    netInfo = NetInfo('/home/wookjin/vggLauraFiles/matthieu/wholeNetwork_oreo.prototxt',
                      '/home/wookjin/vggLauraFiles/matthieu/modifiedFCtoCVMatthieu_smarter__turnedInto__leftPartOfNetwork_convolutionalFacespace_FS6.50.caffemodel')
    uno = UnoTraining()
    uno.init(netInfo, beginLayer='conv1_1', centerLayer="FS6.50",
             endLayer="data-deconv",
             # endLayer="derelu1_2",
             outputFolder='./tests/',
             # outputFolder='/home/wookjin/vggLauraFiles/UnoTraining',
             featureMapsPrefix='/home/wookjin/FeatureMaps/featuremap_lmdb_')
    uno.train(debug=True, verbose=True)
    return uno


if __name__ == '__main__':
    uno = quickTests()
