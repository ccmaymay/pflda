#!/usr/bin/env python


import os


def symlink_to_dir_force(dest_dir, *sources):
    for source in sources:
        basename = os.path.basename(source)
        dest = os.path.join(dest_dir, basename)
        if os.path.isfile(dest):
            os.remove(dest)
        os.symlink(source, dest)


def symlink_model_visualization_to_output(littleowl_dir, output_dir, model_name, *global_sources):
    model_output_dir = os.path.join(output_dir, model_name)
    subgraphs_html_path = os.path.join(
        littleowl_dir,
        'src/pylowl/proj/brightside',
        model_name,
        'postproc/subgraphs.html'
    )
    sources = global_sources + (subgraphs_html_path,)
    for model_output_subdir_name in os.listdir(model_output_dir):
        model_output_subdir = os.path.join(
            model_output_dir,
            model_output_subdir_name
        )
        if os.path.isdir(model_output_subdir):
            symlink_to_dir_force(model_output_subdir, *sources)


def symlink_visualization_to_output(littleowl_dir, output_dir):
    graph_html_path = os.path.join(littleowl_dir,
        'src/pylowl/proj/brightside/postproc/graph.html')
    d3_js_path = os.path.join(littleowl_dir,
        'src/pylowl/proj/brightside/postproc/d3.v3.js')
    global_sources = (graph_html_path, d3_js_path)

    symlink_model_visualization_to_output(littleowl_dir, output_dir, 'm0',
                                          *global_sources)
    symlink_model_visualization_to_output(littleowl_dir, output_dir, 'm1',
                                          *global_sources)


if __name__ == '__main__':
    import sys
    symlink_visualization_to_output(*sys.argv[1:])
