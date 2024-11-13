import sys
import os

from amftrack.util.sys import get_dirname, temp_path
import pandas as pd
import ast
from scipy import sparse
import scipy.io as sio
import cv2
import imageio.v2 as imageio
import numpy as np
import os
from time import time
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_no_external,
    run_back_sub,
    bowler_hat,
)
from amftrack.sparse_util import zhang_suen_thinning


from amftrack.util.sys import get_dates_datetime, get_dirname
import shutil


def process(args):

    i = int(args[-1])
    op_id = int(args[-2])
    hyph_width = int(args[1])
    perc_low = float(args[2])
    perc_high = float(args[3])
    minlow = float(args[4])
    minhigh = float(args[5])

    directory = str(args[6])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()
    directory_name = folder_list[i]
    print(directory_name)
    path_snap = os.path.join(directory, directory_name)
    path_tile = os.path.join(path_snap, "Img/TileConfiguration.txt.registered")
    try:
        tileconfig = pd.read_table(
            path_tile,
            sep=";",
            skiprows=4,
            header=None,
            converters={2: ast.literal_eval},
            skipinitialspace=True,
        )
    except:
        print("error_name")
        path_tile = os.path.join(path_snap, "Img/TileConfiguration.registered.txt")
        tileconfig = pd.read_table(
            path_tile,
            sep=";",
            skiprows=4,
            header=None,
            converters={2: ast.literal_eval},
            skipinitialspace=True,
        )
    dirName = path_snap + "/Analysis"
    try:
        os.mkdir(path_snap + "/Analysis")
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    t = time()
    xs = [c[0] for c in tileconfig[2]]
    ys = [c[1] for c in tileconfig[2]]
    name = tileconfig[0][0]
    imname = "/Img/" + name.split("/")[-1]
    im = imageio.imread(directory + directory_name + imname)
    dim = (
        int(np.max(ys) - np.min(ys)) + max(im.shape),
        int(np.max(xs) - np.min(xs)) + max(im.shape),
    )
    ims = []
    skel = np.zeros(dim, dtype=bool)
    params = [30]
    for index, name in enumerate(tileconfig[0]):
        # for index, name in enumerate(list_debug):
        print(name)
        imname = "/Img/" + name.split("/")[-1]
        im = imageio.imread(directory + directory_name + imname)
        imname2 = "/Img/" + name.split("/")[-1]
        im2 = imageio.imread(directory + directory_name + imname2)
        bowled2 = bowler_hat(-im2, 32, params)
        im[bowled2 <= 0.09] = np.maximum(im[bowled2 <= 0.09], 250)
        shape = im.shape
        print("segmenting")
        segmented = extract_skel_no_external(
            im, [hyph_width], perc_low, perc_high, minlow, minhigh
        )
        # low = np.percentile(-im+255, perc_low)
        # high = np.percentile(-im+255, perc_high)
        # segmented = filters.apply_hysteresis_threshold(-im+255, low, high)
        boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
            tileconfig[2][index][1] - np.min(ys)
        )
        skel[
            boundaries[1] : boundaries[1] + shape[0],
            boundaries[0] : boundaries[0] + shape[1],
        ] += segmented.astype(bool)
    print("number to reduce : ", np.sum(skel == 1), np.sum(skel == 0))
    skel = zhang_suen_thinning(skel)
    # skel_sparse = sparse.lil_matrix(skel)
    sio.savemat(
        path_snap + "/Analysis/skeleton.mat",
        {"skeleton": sparse.csc_matrix(skel)},
    )
    print("time=", time() - t)


if __name__ == "__main__":
    process(sys.argv)
