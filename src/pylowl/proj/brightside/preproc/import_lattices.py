#!/usr/bin/env python


import os
from pylowl.proj.brightside.corpus import load_lattice_best_tokens, write_concrete_docs, Document


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


def load_docs_from_concrete_lattices(input_dir):
    for (agg_tokens, video_dir) in load_aggregate_lattice_best_tokens(input_dir):
        yield Document(agg_tokens, id=video_dir)


def import_lattices(input_dir, output_dir):
    write_concrete_docs(load_docs_from_concrete_lattices(input_dir),
                        output_dir)


if __name__ == '__main__':
    import sys
    import_lattices(*sys.argv[1:])
