import sys

from scipy import sparse
from pymatreader import read_mat

from amftrack.util.sys import temp_path
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
import scipy.sparse
import pickle
import pandas as pd


def process(args):
    directory = str(args[1])

    i = int(args[-1])
    op_id = int(args[-2])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()
    directory_name = folder_list[i]
    path_snap = directory + directory_name
    skel = read_mat(path_snap + "/Analysis/skeleton_pruned_realigned.mat")["skeleton"]
    skeleton = scipy.sparse.dok_matrix(skel)

    # nx_graph_poss=[generate_nx_graph(from_sparse_to_graph(skeleton)) for skeleton in skels_aligned]
    # nx_graphs_aligned=[nx_graph_pos[0] for nx_graph_pos in nx_graph_poss]
    # poss_aligned=[nx_graph_pos[1] for nx_graph_pos in nx_graph_poss]
    # nx_graph_pruned=[clean_degree_4(prune_graph(nx_graph),poss_aligned[i])[0] for i,nx_graph in enumerate(nx_graphs_aligned)]
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    # Optional, to remove spurs
    nx_graph, pos = remove_spurs(nx_graph, pos)

    nx_graph_pruned = clean_degree_4(nx_graph, pos)[0]
    path_save = path_snap + "/Analysis/nx_graph_pruned.p"
    print(path_save)
    pickle.dump((nx_graph_pruned, pos), open(path_save, "wb"))


if __name__ == "__main__":
    process(sys.argv)
