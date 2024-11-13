import sys

from amftrack.util.sys import temp_path
import scipy.io as sio
from pymatreader import read_mat
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_graph import (
    sparse_to_doc,
    get_degree3_nodes,
)
import open3d as o3d
from cycpd import rigid_registration
import pandas as pd
import os


def process(args):

    i = int(args[-1])
    op_id = int(args[-2])
    thresh = int(args[1])
    directory = str(args[2])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()

    print(run_info)
    dates_datetime_chosen = folder_list[i : i + 2]
    print("========")
    print(f"Matching plate {dates_datetime_chosen[0]} at dates {dates_datetime_chosen}")
    print("========")
    dates = dates_datetime_chosen

    dilateds = []
    skels = []
    skel_docs = []
    for directory_name in dates:
        path_snap = os.path.join(directory, directory_name)
        skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
        skel = skel_info["skeleton"]
        skels.append(skel)
        skel_doc = sparse_to_doc(skel)
        skel_docs.append(skel_doc)
    isnan = True
    for order in [(0, 1), (1, 0)]:
        skeleton1, skeleton2 = skel_docs[order[0]], skel_docs[order[1]]
        skelet_pos = np.array(list(skeleton1.keys()))
        samples = np.random.choice(skelet_pos.shape[0], len(skeleton2.keys()) // 100)
        X = np.transpose(skelet_pos[samples, :])
        skelet_pos = np.array(list(skeleton2.keys()))
        samples = np.random.choice(skelet_pos.shape[0], len(skeleton2.keys()) // 100)
        Y = np.transpose(skelet_pos[samples, :])
        reg = rigid_registration(
            **{
                "X": np.transpose(X.astype(float)),
                "Y": np.transpose(Y.astype(float)),
                "scale": False,
                "tolerance": 1e-5,
                "w": 1e-5,
            }
        )
        out = reg.register()
        Rfound = reg.R[0:2, 0:2]
        tfound = np.dot(Rfound, reg.t[0:2])
        if order == (0, 1):
            t_init = -tfound
            Rot_init = Rfound
        else:
            Rot_init, t_init = np.linalg.inv(Rfound), np.dot(
                np.linalg.inv(Rfound), tfound
            )
        sigma2 = reg.sigma2
        if sigma2 >= thresh:
            print("========")
            print(
                f"Failed to match plate {dates_datetime_chosen[0]} at dates {dates_datetime_chosen}"
            )
            print("========")
            continue
        isnan = np.isnan(tfound[0])
        if isnan:
            continue
        skeleton1, skeleton2 = skel_docs[0], skel_docs[1]
        X = np.transpose(np.array(get_degree3_nodes(skeleton1)))
        Y = np.transpose(np.array(get_degree3_nodes(skeleton2)))
        X = np.insert(X, 2, values=0, axis=0)
        Y = np.insert(Y, 2, values=0, axis=0)
        print(X.shape, Y.shape)
        vectorX = o3d.utility.Vector3dVector(np.transpose(X))
        vectorY = o3d.utility.Vector3dVector(np.transpose(Y))
        source = o3d.geometry.PointCloud(vectorX)
        target = o3d.geometry.PointCloud(vectorY)
        threshold = 200
        trans_init = np.asarray(
            [
                [Rot_init[0, 0], Rot_init[0, 1], 0, t_init[0]],
                [Rot_init[1, 0], Rot_init[1, 1], 0, t_init[1]],
                [0, 0, 1, 0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        print(reg_p2p)
        Rfound = reg_p2p.transformation[0:2, 0:2]
        tfound = reg_p2p.transformation[0:2, 3]
        print(Rfound, tfound)
        X, Y = X[0:2, :], Y[0:2, :]
        Yrep = np.transpose(np.transpose(np.dot(Rfound, X)) + tfound)
        # fig=plt.figure(figsize=(10,9))
        # ax = fig.add_subplot(111)
        # ax.scatter(np.transpose(Yrep)[:,0],np.transpose(Yrep)[:,1])
        # ax.scatter(np.transpose(Y)[:,0],np.transpose(Y)[:,1])
        break

    if not isnan:
        sio.savemat(path_snap + "/Analysis/transform.mat", {"R": Rfound, "t": tfound})
    else:
        sio.savemat(
            path_snap + "/Analysis/transform_corrupt.mat",
            {"R": np.array([[1, 0], [0, 1]]), "t": np.array([0, 0])},
        )


if __name__ == "__main__":
    process(sys.argv)
