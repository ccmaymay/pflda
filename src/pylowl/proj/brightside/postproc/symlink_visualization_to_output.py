#!/usr/bin/env python


import os
from glob import glob


def symlink_to_dir_force(dest_dir, *sources):
    for source in sources:
        basename = os.path.basename(source)
        dest = os.path.join(dest_dir, basename)
        if os.path.isfile(dest) or os.path.islink(dest):
            os.remove(dest)
        os.symlink(source, dest)


def js_and_html_paths(parent_dir):
    return (
        glob(os.path.join(parent_dir, '*.js'))
        + glob(os.path.join(parent_dir, '*.html'))
    )

def symlink_model_visualization_to_output(littleowl_dir, model_output_dir,
                                          model_name, *global_sources):
    postproc_dir = os.path.join(
        littleowl_dir,
        'src/pylowl/proj/brightside',
        model_name,
        'postproc'
    )
    local_sources = tuple(js_and_html_paths(postproc_dir))
    sources = global_sources + local_sources

    for model_output_subdir_name in os.listdir(model_output_dir):
        model_output_subdir = os.path.join(
            model_output_dir,
            model_output_subdir_name
        )
        if os.path.isdir(model_output_subdir):
            symlink_to_dir_force(model_output_subdir, *sources)


def symlink_visualization_to_output(pylowl_dir, brightside_output_dir):
    postproc_dir = os.path.join(pylowl_dir, 'proj/brightside/postproc')
    global_sources = tuple(js_and_html_paths(postproc_dir))
    for model_name in os.listdir(brightside_output_dir):
        model_output_dir = os.path.join(brightside_output_dir, model_name)
        if os.path.isdir(model_output_dir):
            symlink_model_visualization_to_output(pylowl_dir, model_output_dir,
                                                  model_name, *global_sources)


if __name__ == '__main__':
    import sys
    symlink_visualization_to_output(*sys.argv[1:])
