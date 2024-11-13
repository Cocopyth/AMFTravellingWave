"""
The utils for tests are here.
There are several sorts of utils for tests:
- boolean functions to skip tests that haven't the data requiered
- function that create complicate objects like Experiment or Mock objects
"""

import cv2
import numpy as np
import os
import pickle
from amftrack.util.sys import (
    test_path,
    get_current_folders,
    update_plate_info,
    update_analysis_info,
    get_analysis_info,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    load_graphs,
)
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.util.dbx import get_dropbox_folders

video_path = os.path.join(test_path, "flow_processing")
from amftrack.pipeline.launching.run import run_transfer


def is_equal_seq(list_of_list_1, list_of_list_2):
    for i in range(len(list_of_list_1)):
        if not np.all(list_of_list_1[i] == list_of_list_2[i]):
            return False
    return True


def has_video():
    """
    Check if there is a flow_processing present to run tests on videos.
    The flow_processing should be several images of the same size in directory `flow_processing`.
    """
    video_path = os.path.join(test_path, "flow_processing")
    if os.path.isdir(video_path):
        return True
    return False


def make_video() -> None:
    """
    Makes a fake flow_processing and a flow_processing directory used to test flow_processing utils
    """
    video_path = os.path.join(test_path, "flow_processing")
    if not os.path.isdir(video_path):
        os.mkdir(video_path)

        im1 = np.triu(np.ones((10, 10)) * 100)
        im2 = np.triu(np.ones((10, 10)) * 10)
        for i in range(2):
            cv2.imwrite(os.path.join(video_path, f"image{i}.png"), im1)
        cv2.imwrite(os.path.join(video_path, f"image{2}.png"), im2)


def make_image() -> np.array:
    image = np.full(shape=(250, 300, 3), fill_value=255, dtype=np.int8)
    image[100:120, :, :] = 100
    image[:100, :, :] = 50
    image[:, 200:220, :] = 0
    return image


def has_test_repo():
    "Tests if the general test repository is present"
    return os.path.isdir(test_path)


def make_test_repo():
    if not os.path.isdir(test_path):
        os.mkdir(test_path)


def has_test_plate():
    "Check if a Prince plate is present, for tests that use it"
    if has_test_repo():
        if len(os.listdir(test_path)) > 0:
            # TODO(FK): check if this is really a testing plate instead
            return True
    else:
        return False

def make_folders(directory):
    update_plate_info(directory)
    folder_df = get_current_folders(directory)
    return folder_df

def make_experiment_object():
    "Build an experiment object using the plate that is in the test repository."
    directory = test_path
    plate_name = "111111_20600101"  # TODO(FK): find the name automaticaly (can be different based on the person)
    update_plate_info(directory)
    folder_df = get_current_folders(directory)
    selected_df = folder_df.loc[folder_df["unique_id"] == plate_name]
    if len(selected_df) < 4:
        all_folders_drop = get_dropbox_folders("/DATA/PRINCE", True)
        folders_drop = all_folders_drop.loc[all_folders_drop["unique_id"] == plate_name]
        run_transfer(
            "from_drop.py",
            [directory],
            folders_drop,
        )
    exp = Experiment(directory)
    exp.load(selected_df, suffix="")
    for t in range(len(exp.folders)):
        exp.load_tile_information(t)
    return exp


def make_experiment_object_analysis(directory = test_path):
    "Build an experiment object using the plate that is in the test repository."
    plate_name = "111111_20600101"  # TODO(FK): find the name automaticaly (can be different based on the person)
    update_analysis_info(directory)

    analysis_info = get_analysis_info(directory)
    select = analysis_info.loc[analysis_info["unique_id"] == plate_name]
    path_exp = f'{directory}{select["path_exp"].iloc[0]}'
    exp = pickle.load(open(path_exp, "rb"))
    load_graphs(exp, directory, range(exp.ts), post_process=True)
    for t in range(len(exp.folders)):
        exp.load_tile_information(t)
    exp.save_location = "/".join(path_exp.split("/")[:-1])
    load_study_zone(exp)

    return exp


def make_experiment_object_multi():
    "Build an experiment object using the plate that is in the test repository."
    directory = test_path
    plate_name = "111111_20600101"  # TODO(FK): find the name automaticaly (can be different based on the person)
    update_plate_info(directory)
    folder_df = get_current_folders(directory)
    selected_df = folder_df.loc[folder_df["unique_id"] == plate_name]
    if len(selected_df) < 4:
        all_folders_drop = get_dropbox_folders("/DATA/PRINCE", True)
        folders_drop = all_folders_drop.loc[all_folders_drop["unique_id"] == plate_name]
        run_transfer(
            "from_drop.py",
            [directory],
            folders_drop,
        )
    exp = Experiment(directory)
    exp.load(selected_df, suffix="")
    for t in range(len(exp.folders)):
        exp.load_tile_information(t)
    return exp


class EdgeMock:
    """
    A fake Edge object without having to make an experiment object
    """

    def __init__(self, list_coord):
        self.list_coord = list_coord

    def pixel_list(self, t):
        return self.list_coord
