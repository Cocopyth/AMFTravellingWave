import sys

from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    get_width_info_new,
)


from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.util.sys import temp_path
import pickle
import networkx as nx
import pandas as pd


def process(args):
    directory = str(args[1])
    skip = eval(args[2])
    resolution = eval(args[3])
    i = int(args[-1])
    op_id = int(args[-2])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    # plate = list(run_info['PrincePos'])[i]
    # Sometime plate in param file is inconsistent with folder name...
    # plate = int(list(run_info['folder'])[i].split('_')[-1][5:])
    folder_list = list(run_info["folder"])
    folder_list.sort()
    directory_name = folder_list[i]
    plate = int(directory_name.split("_")[-1][5:])

    exp = Experiment(directory)
    exp.load(run_info.loc[run_info["folder"] == directory_name], suffix="")
    path_snap = directory + directory_name

    (G, pos) = exp.nx_graph[0], exp.positions[0]
    edge_test = get_width_info_new(exp, 0, resolution=resolution, skip=skip)
    nx.set_edge_attributes(G, edge_test, "width")
    print(i, f"saving {path_snap}")
    pickle.dump((G, pos), open(f"{path_snap}/Analysis/nx_graph_pruned_width.p", "wb"))


if __name__ == "__main__":
    process(sys.argv)
