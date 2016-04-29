import sys
# caffe3 with python
caffe3_root = '/home/wookjin/caffe-with_py/'  # this file is expected to be in {caffe_root}/examples
# caffe1 with python
caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples
caffe_root = caffe1_root  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe


from caffe.proto import caffe_pb2
from google.protobuf import text_format


# https://learningcarrot.wordpress.com/2015/10/25/protocol-buffers-in-caffe/
def readPrototxtFile(filepath):
    solver_config = caffe.proto.caffe_pb2.NetParameter()
    #TODO how to read proto file?
    return _readProtoFile(filepath, solver_config)


def readProtoSolverFile(filepath):
    solver_config = caffe.proto.caffe_pb2.SolverParameter()
    #TODO how to read proto file?
    return _readProtoFile(filepath, solver_config)


def getLayers(filepath):
    return readPrototxtFile(filepath).layers


def _readProtoFile(filepath, parser_object):
    file = open(filepath, "r")

    if not file:
        raise "ERROR (" + filepath + ")!"

    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object


def quickTests():
    protoPath = '/home/wookjin/vggLauraFiles/matthieu/wholeNetwork_oreo.prototxt'
    proto = object()
    proto = readPrototxtFile(protoPath)
    layers = getLayers(protoPath)
    print [str(l.name) for l in layers]
    return layers

if __name__ == '__main__':
    res = quickTests()
