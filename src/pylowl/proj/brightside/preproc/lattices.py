#!/usr/bin/env python


import sys
from glob import glob
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from concrete.structure.ttypes import TokenLattice


def print_lattice(path):
    for path in paths:
        with open(path, 'rb') as f:
            transportIn = TTransport.TFileObjectTransport(f)
            protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
            lattice = TokenLattice()
            lattice.read(protocolIn)
            if lattice.cachedBestPath is not None:
                print ' '.join([t.text for t in lattice.cachedBestPath.tokenList])


if __name__ == '__main__':
    import sys
    for pat in sys.argv[1:]:
        for path in glob(pat):
            print_lattice(path)
