#!/usr/bin/env python


import os
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from concrete.structure.ttypes import TokenLattice


def load_lattice_best_tokens(path):
    with open(path, 'rb') as f:
        transportIn = TTransport.TFileObjectTransport(f)
        protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
        lattice = TokenLattice()
        lattice.read(protocolIn)
        best_path = lattice.cachedBestPath
        if best_path is None:
            return None
        else:
            token_list = best_path.tokenList
            if token_list is None:
                return None
            else:
                return [t.text for t in token_list]


def load_aggregate_lattice_best_tokens(input_dir):
    for video_dir_name in os.listdir(input_dir):
        video_dir = os.path.join(input_dir, video_dir_name)
        tokens = []
        for frame_filename in os.listdir(video_dir):
            frame_path = os.path.join(video_dir, frame_filename)
            lattice_best = load_lattice_best_tokens(frame_path)
            if lattice_best is not None:
                tokens.extend(lattice_best)
        if tokens:
            yield (tokens, video_dir)


def import_lattices(input_dir, output_dir):
    for (agg_tokens, video_dir) in load_aggregate_lattice_best_tokens(input_dir):
        print agg_tokens, video_dir


if __name__ == '__main__':
    import sys
    import_lattices(*sys.argv[1:])
