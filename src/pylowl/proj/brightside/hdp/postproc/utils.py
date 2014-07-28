from pylowl.proj.brightside.utils import parent_package_name
from pylowl.proj.brightside.hdp.postproc.generate_d3_graph import generate_d3_graph
from pylowl.proj.brightside.hdp.postproc.generate_d3_subgraphs import generate_d3_subgraphs
from pylowl.proj.brightside.hdp.postproc.generate_d3_user_subgraphs import generate_d3_user_subgraphs
import pkg_resources
import shutil
import os


def postprocess(output_dir):
    brightside_postproc = parent_package_name(parent_package_name(__package__)) + '.postproc'

    generate_d3_graph(output_dir, os.path.join(output_dir, 'graph.json'))
    for basename in ('d3.v3.js', 'core.js', 'graph.html'):
        shutil.copy(pkg_resources.resource_filename(brightside_postproc, basename),
            os.path.join(output_dir, basename))

    generate_d3_subgraphs(output_dir, os.path.join(output_dir, 'subgraphs.json'))
    generate_d3_user_subgraphs(output_dir, os.path.join(output_dir, 'user_subgraphs.json'))
    for basename in ('subgraphs.html', 'user_subgraphs.html'):
        shutil.copy(pkg_resources.resource_filename(__package__, basename),
            os.path.join(output_dir, basename))
