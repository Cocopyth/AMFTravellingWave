import sys
from amftrack.util.sys import temp_path
import pandas as pd
from pymatreader import read_mat
import cv2
from amftrack.pipeline.functions.image_processing.extract_skel import bowler_hat
import numpy as np
from scipy import sparse
from time import time_ns
from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload
import imageio.v2 as imageio
import os

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
select = select.sort_values("datetime")
print(select["folder"])
imgs = []

kernel = np.ones((3, 3), np.uint8)
itera = 1

texts = list(select["folder"])
fontScale = 3
color = (0, 255, 255)
id_unique = (
    str(int(select["Plate"].iloc[0]))
    + "_"
    + str(int(str(select["CrossDate"].iloc[0]).replace("'", "")))
)
for k, path in enumerate(select["total_path"]):
    print(path)
    # print(folder)
    path_snap = path
    # skel_info = read_mat(path_snap + "/Analysis/skeleton_masked.mat")
    skel_info = read_mat(path_snap + "/Analysis/skeleton.mat")

    skel = skel_info["skeleton"]
    im = read_mat(path_snap + "/Analysis/raw_image.mat")["raw"]
    size_image = im.shape
    if k == 0:
        resize = (2624, int(2624 * size_image[0] / size_image[1]))
    compressed = sparse.csr_matrix(
        (
            [1] * len(skel.nonzero()[0]),
            (skel.nonzero()[0] // 15, skel.nonzero()[1] // 15),
        ),
        (5800, 11000),
    )
    skel_comp = cv2.dilate(
        compressed.toarray().astype(np.uint8), kernel, iterations=itera
    )
    blackAndWhiteImage3 = cv2.cvtColor(skel_comp, cv2.COLOR_GRAY2BGR)
    blackAndWhiteImage3[skel_comp > 0] = (0, 0, 125)
    blackAndWhiteImage3[skel_comp == 0] = (255, 255, 255)
    blackAndWhiteImage3 = blackAndWhiteImage3[: im.shape[0], : im.shape[1]]
    bc_rm_f = cv2.cvtColor(
        cv2.normalize(-im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    bc_rm_f = bc_rm_f[: blackAndWhiteImage3.shape[0], : blackAndWhiteImage3.shape[1]]

    added_image = cv2.addWeighted(blackAndWhiteImage3, 0.3, bc_rm_f, 0.5, 0)
    imgs.append(cv2.resize(added_image, resize))

for i, img in enumerate(imgs):
    anchor = img.shape[0] // 10, img.shape[1] // 10
    cv2.putText(
        img=img,
        text=texts[i],
        org=anchor,
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=fontScale,
        color=color,
        thickness=3,
    )


time = time_ns()
save_path_temp = os.path.join(temp_path, f"{time}.mp4")
imageio.mimsave(save_path_temp, imgs)
dir_drop = "DATA/PRINCE_ANALYSIS"
upload_path = f"/{dir_drop}/{id_unique}/{id_unique}_skelet.mp4"
upload(save_path_temp, upload_path)
print(upload_path, len(imgs), len(select))
os.remove(save_path_temp)
