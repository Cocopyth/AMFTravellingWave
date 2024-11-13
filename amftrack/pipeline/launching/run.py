import os
import pandas as pd
from amftrack.util.sys import (
    path_code,
    update_plate_info_local,
    get_current_folders_local,
    temp_path,
    fiji_path,
)
import imageio
from typing import List

from time import time_ns
from tqdm.autonotebook import tqdm
import subprocess
import pickle


def make_stitching_loop(directory: str, dirname: str, op_id: int) -> None:
    """
    For the acquisition directory `directory`, prepares the stitching script used to stitch the image.
    """
    # TODO(FK): handle the case where there is no stiching_loops directory
    a_file = open(
        os.path.join(path_code, "pipeline/scripts/stitching_loops/stitching_loop.ijm"),
        "r",
    )

    list_of_lines = a_file.readlines()
    list_of_lines[4] = f"mainDirectory = \u0022{directory}\u0022 ;\n"
    list_of_lines[29] = f"\t if(startsWith(list[i],\u0022{dirname}\u0022)) \u007b\n"

    file_name = f"{temp_path}/stitching_loops/stitching_loop{op_id}.ijm"
    a_file = open(file_name, "w")
    a_file.writelines(list_of_lines)
    a_file.close()


def run_stitch(directory: str, folders: pd.DataFrame) -> None:
    """
    Runs the stiching loop.
    :param directory: is the folder containing the directories for each acquisition
    :param folder: is a pandas dataframe with information on the folder
    """
    # TODO(FK): Should encapsulate and factorize to have a version working with str instead of pd frame
    folder_list = list(folders["folder"])
    # TODO(FK): raise error if empty list
    folder_list.sort()
    with tqdm(total=len(folder_list), desc="stitched") as pbar:
        for folder in folder_list:
            op_id = time_ns()
            im = imageio.imread(f"{directory}/{folder}/Img/Img_r03_c05.tif")
            for x in range(1, 11):
                for y in range(1, 16):
                    strix = str(x) if x >= 10 else f"0{x}"
                    striy = str(y) if y >= 10 else f"0{y}"
                    path = f"{directory}/{folder}/Img/Img_r{strix}_c{striy}.tif"
                    if not os.path.isfile(path):
                        f = open(path, "w")
                    if os.path.getsize(path) == 0:
                        imageio.imwrite(path, im * 0)
            make_stitching_loop(directory, folder, op_id)
            command = [
                fiji_path,
                "--mem=8000m",
                "--headless",
                "--ij2",
                "--console",
                "-macro",
                f"{temp_path}/stitching_loops/stitching_loop{op_id}.ijm",
            ]
            print(" ".join(command))
            process = subprocess.run(command)
            pbar.update(1)


def run(
    code: str,
    args: List,
    folders: pd.DataFrame,
    loc_code="pipeline/scripts/image_processing/",
    pyt_vers="python3",
) -> None:
    """
    Run the chosen script `code` localy.
    :param code: name of the script file such as "prune.py", it has to be in the image_processing file
    :param args: list of arguments used by the script
    """
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    folder_list = list(folders["folder"])
    folder_list.sort()
    args_str = [str(arg) for arg in args]
    with tqdm(total=len(folder_list), desc="folder_treated") as pbar:
        for index, folder in enumerate(folder_list):
            command = (
                [f"{pyt_vers}", f"{path_code}{loc_code}{code}"]
                + args_str
                + [f"{op_id}", f"{index}"]
            )
            print(" ".join(command))
            process = subprocess.run(command, stdout=subprocess.DEVNULL)
            pbar.update(1)


def run_function(
    process,
    args: List,
    folders: pd.DataFrame,
    per_unique_id=False,
    sequential_process=False,
) -> None:
    """
    Run the chosen script `code` localy.
    :param code: name of the script file such as "prune.py", it has to be in the image_processing file
    :param args: list of arguments used by the script
    """
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    folder_list = list(folders["folder"])
    folder_list.sort()
    args += [op_id, 0]
    if per_unique_id:
        unique_ids = list(set(folders["unique_id"].values))
        with tqdm(total=len(unique_ids), desc="plates_treated") as pbar:
            for index, folder in enumerate(unique_ids):
                args[-1] = index
                process(args)
                pbar.update(1)
    else:
        with tqdm(total=len(folder_list), desc="folder_treated") as pbar:
            if sequential_process:
                last_index = len(folder_list) - 1
            else:
                last_index = len(folder_list)
            for index, folder in enumerate(folder_list[:last_index]):
                args[-1] = index
                process(args)
                pbar.update(1)


def run_post_process(
    code: str,
    list_f,
    list_args: List,
    args,
    folders: pd.DataFrame,
    loc_code="pipeline/scripts/post_processing/",
    pyt_vers="python3",
) -> None:
    """
    Run the chosen script `code` localy.
    :param code: name of the script file such as "prune.py", it has to be in the image_processing file
    :param args: list of arguments used by the script
    """
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    pickle.dump((list_f, list_args), open(f"{temp_path}/{op_id}.pick", "wb"))

    folder_list = list(folders["folder_analysis"])
    folder_list.sort()
    args_str = [str(arg) for arg in args]
    with tqdm(total=len(folder_list), desc="folder_treated") as pbar:
        for index, folder in enumerate(folder_list):
            command = (
                [f"{pyt_vers}", f"{path_code}{loc_code}{code}"]
                + args_str
                + [f"{op_id}", f"{index}"]
            )
            print(" ".join(command))
            process = subprocess.run(command, stdout=subprocess.DEVNULL)
            pbar.update(1)


def run_all_time(
    code: str,
    args: List,
    folders: pd.DataFrame,
    loc_code="pipeline/scripts/image_processing/",
    pyt_vers="python3",
) -> None:
    """
    Run the chosen script `code` localy.
    :param code: name of the script file such as "prune.py", it has to be in the image_processing file
    :param args: list of arguments used by the script
    """
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    plates = set(folders["Plate"].values)
    folder_list = list(folders["folder"])
    folder_list.sort()
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    with tqdm(total=len(plates), desc="plate treated") as pbar:
        for index, plate in enumerate(plates):
            command = (
                [f"{pyt_vers}", f"{path_code}{loc_code}{code}"]
                + args_str
                + [f"{op_id}", f"{index}"]
            )
            print(" ".join(command))
            process = subprocess.run(command, stdout=subprocess.DEVNULL)
            pbar.update(1)


def run_transfer(code: str, args: List, folders: pd.DataFrame) -> None:
    """
    Run the chosen script `code` localy.
    :param code: name of the script file such as "prune.py", it has to be in the transfer/scripts/ folder
    :param args: list of arguments used by the script
    """
    run(code, args, folders, loc_code="transfer/scripts/")


if __name__ == "__main__":
    # directory = "/data/felix/width1/full_plates/"  # careful: must have the / at the end
    # # update_plate_info(directory)
    # folder_df = get_current_folders(directory)
    # folders = folder_df.loc[folder_df["Plate"] == "907"]
    # time = "3:00:00"
    # threshold = 0.1
    # args = [threshold, directory]
    # run("prune_skel.py", args, folders)

    # directory = "/data/felix/width1/full_plates/"  # careful: must have the / at the end
    # time = "2:00:00"
    # args = [directory]
    # run("extract_nx_graph.py", args, folders)

    ### TEST 1
    # directory = "/data/felix/plate1050/"
    # update_plate_info_local(directory)
    # folder_df = get_current_folders_local(directory)
    # folders = folder_df.loc[folder_df["Plate"] == "1050"]
    # run_stitch(directory, folders)

    ### TEST 2
    from amftrack.util.sys import storage_path

    directory = os.path.join(storage_path, "plate1050")
    update_plate_info_local(directory)
    folder_df = get_current_folders_local(directory)
    folders = folder_df.loc[folder_df["Plate"] == "1050"]

    time = "3:00:00"
    low = 30
    high = 80
    extend = 30
    args = [low, high, extend, directory]
    run("extract_skel.py", args, folders)
