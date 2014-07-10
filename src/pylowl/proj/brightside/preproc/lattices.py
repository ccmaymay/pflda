#!/usr/bin/env python

import sys
from glob import glob
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from concrete.structure.ttypes import TokenLattice

locs = glob(sys.argv[1])
for loc in locs:
    with open(loc, 'rb') as f:
        transportIn = TTransport.TFileObjectTransport(f)
        protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
        lattice = TokenLattice()
        lattice.read(protocolIn)
        if lattice.cachedBestPath is not None:
            print [t.text for t in lattice.cachedBestPath.tokenList]
