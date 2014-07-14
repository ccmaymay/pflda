#!/usr/bin/env python


import sys
import os
from pylowl.proj.brightside.corpus import load_lattice_best_tokens, write_concrete_docs, Document


def load_aggregate_lattice_best_tokens(input_dir):
    for video_dir_name in os.listdir(input_dir):
        video_dir = os.path.join(input_dir, video_dir_name)
        if os.path.isdir(video_dir):
            tokens = []
            for frame_filename in os.listdir(video_dir):
                frame_path = os.path.join(video_dir, frame_filename)
                if os.path.isfile(frame_path):
                    lattice_best = None
                    try:
                        lattice_best = load_lattice_best_tokens(frame_path)
                    except:
                        sys.stderr.write('Warning: failed to load lattice from %s' % frame_path)
                    else:
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
    import_lattices(*sys.argv[1:])
