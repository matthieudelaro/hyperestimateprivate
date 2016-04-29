import sys
# caffe3 with python
caffe3_root = '/home/wookjin/caffe-with_py/'  # this file is expected to be in {caffe_root}/examples
# caffe1 with python
caffe1_root = '/home/wookjin/caffe-not-nyf-nyf-with_py/'  # this file is expected to be in {caffe_root}/examples
caffe1l_root = '/home/wookjin/caffe1WithLocallyConLayers/'  # this file is expected to be in {caffe_root}/examples
caffe_root = caffe1_root  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

import caffe
import numpy as np
import sys
import scipy
import os


def convert(fromFile):
    [baseName, exension] = os.path.splitext(fromFile)
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( fromFile , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    scipy.misc.imsave(baseName + '.png', arr[0, [2, 1, 0]])
    np.save(baseName + '.npy' , out )
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python BinaryProtoToNumpy.py meanfiles.binaryproto,[mean2,mean3..]")
        sys.exit()
    else:
        for arg in sys.argv[1:]:
            convert(arg)
