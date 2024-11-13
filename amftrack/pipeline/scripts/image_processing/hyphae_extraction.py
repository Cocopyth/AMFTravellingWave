import sys

import os
from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    resolve_anastomosis_crossing_by_root,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    save_graphs,
)
from amftrack.pipeline.functions.image_processing.experiment_util import get_ROI
import pandas as pd
import json
from time import time_ns
from amftrack.util.sys import temp_path

directory = str(sys.argv[1])
limit = int(sys.argv[2])
version = str(sys.argv[3])
suffix = str(sys.argv[4])
lim_considered = int(sys.argv[5])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
# print(run_info['unique_id'])
plates = list(set(run_info["unique_id"].values))
print(i, op_id, run_info)
plates.sort()
unique_id = plates[i]
folders = run_info.loc[run_info["unique_id"] == unique_id]
folders = folders.sort_values("datetime")
# folders = folders.iloc[:5]
corrupted_rotation = folders.loc[
    ((folders["/Analysis/transform.mat"] == False))
    & (folders["/Analysis/transform_corrupt.mat"])
]["folder"]

folder_list = list(folders["folder"])
folder_list.sort()
indexes = [folder_list.index(corrupt_folder) for corrupt_folder in corrupted_rotation]
indexes = [index for index in indexes if index < limit]
indexes.sort()
indexes += [limit]
indexes = [limit]

start = 0
labeled = suffix == "_labeled"
for index in indexes:
    stop = index
    select_folder_names = folder_list[start:stop]
    plate = int(folder_list[0].split("_")[-1][5:])
    # confusion between plate number and position in Prince
    exp = Experiment(directory)
    select_folders = folders.loc[run_info["folder"].isin(select_folder_names)]
    print(select_folders["unique_id"])
    exp.load(select_folders)
    exp.dates.sort()
    # when no width is included
    # width_based_cleaning(exp)
    if labeled:
        resolve_anastomosis_crossing_by_root(exp, lim_considered=lim_considered)
    # get_mother(exp.hyphaes)
    # solve_two_ends = resolve_ambiguity_two_ends(exp_clean.hyphaes)
    # solved = solve_degree4(exp_clean)
    # clean_obvious_fake_tips(exp_clean)
    dates = exp.dates
    exp.dates.sort()
    save_graphs(exp)
    exp.nx_graph = None
    dirName = f"{directory}Analysis_{unique_id}_{index}_Version{version}"
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    # hyphs, gr_inf = save_hyphaes(
    #     exp, f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/"
    # )
    # exp.save(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/")
    exp.save_location = dirName
    exp.pickle_save(f"{dirName}/")
    select_folders.to_json(f"{dirName}/folder_info.json")
    ROI = get_ROI(exp, 0)
    polygon_geojson = ROI.__geo_interface__

    # Save the GeoJSON to a file
    with open(f"{dirName}/ROI.geojson", "w") as geojson_file:
        json.dump(polygon_geojson, geojson_file)
    start = stop
