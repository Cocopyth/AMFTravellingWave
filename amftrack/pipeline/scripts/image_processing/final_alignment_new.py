import sys

import networkx as nx

from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_all_nodes,
    get_all_edges,
)
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.util.register_videos import (
    find_optimal_R_and_t,
    objective_function,
    average_min_distance_to_set_fast,
)
from amftrack.util.sys import temp_path
import scipy.io as sio
from pymatreader import read_mat
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_graph import (
    sparse_to_doc,
    get_degree3_nodes,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
import pandas as pd
import os


def process(args):

    i = int(args[-1])
    op_id = int(args[-2])
    directory = str(args[1])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()
    folders = run_info.sort_values("datetime")

    select = folders.iloc[i : i + 2]
    directory_name = folder_list[i + 1]
    path_snap = directory + directory_name
    exp = Experiment(directory)
    exp.load(select, suffix="")
    for t in range(exp.ts):
        exp.load_tile_information(t)
        exp.save_location = ""

        load_study_zone(exp)

        # load_graphs(exp, directory_targ,indexes = [0])
        edges = get_all_edges(exp, t)

        weights = {
            (edge.begin.label, edge.end.label): edge.length_um(t) for edge in edges
        }
        nx.set_edge_attributes(exp.nx_graph[t], weights, "length")
        G = exp.nx_graph[t]
        components = nx.connected_components(G)
        S = [G.subgraph(c).copy() for c in components]
        selected = [g for g in S if g.size(weight="length") >= 1e3]
        len_connected = [
            (nx_graph.size(weight="weight") / 10**6) for nx_graph in selected
        ]
        print(len_connected)
        G = selected[0]
        for g in selected[1:]:
            G = nx.compose(G, g)
        # Find the largest connected component

        # Create a new graph representing the largest connected component
        largest_component_graph = G
        exp.nx_graph[t] = largest_component_graph
    dists = []
    rottrans = []
    for order in [(0, 1), (1, 0)]:
        pixels1 = [
            node.pos(order[0])
            for node in get_all_nodes(exp, order[0])
            if node.degree(order[0]) == 3
        ]
        pixels2 = [
            node.pos(order[1])
            for node in get_all_nodes(exp, order[1])
            if node.degree(order[1]) == 3
        ]
        X = np.array(pixels1)
        Y = np.array(pixels2)
        Rfound, tfound = find_optimal_R_and_t(X, Y)
        transformed_source = np.dot(Rfound, X.T).T + tfound

        # Compute distances to the nearest target points
        dist = average_min_distance_to_set_fast(transformed_source, Y)
        dists.append(dist)
        rottrans.append((Rfound, tfound))
    index = np.argmin(dists)
    Rfound, tfound = rottrans[index]
    order = [(0, 1), (1, 0)][index]
    if order == (1, 0):
        t_init = tfound
        Rot_init = Rfound
    else:
        Rot_init, t_init = np.linalg.inv(Rfound), np.dot(np.linalg.inv(Rfound), -tfound)
    print(Rot_init, t_init)
    sio.savemat(path_snap + "/Analysis/transform_new.mat", {"R": Rot_init, "t": t_init})


if __name__ == "__main__":
    process(sys.argv)
