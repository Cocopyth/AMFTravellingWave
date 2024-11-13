"""Utils for defining environement variables and defining paths"""

from scipy import sparse
import numpy as np
import os
from datetime import datetime
from amftrack.pipeline.functions.image_processing.extract_graph import (
    sparse_to_doc,
)
import cv2
import json
import pandas as pd
from amftrack.util.dbx import download, upload, load_dbx, env_config
from tqdm.autonotebook import tqdm
from time import time_ns
from decouple import Config, RepositoryEnv
from pymatreader import read_mat
import shutil
import hashlib


DOTENV_FILE = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + "/local.env"
)
env_config = Config(RepositoryEnv(DOTENV_FILE))

path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

storage_path = env_config.get("STORAGE_PATH")
conda_path = env_config.get("CONDA_PATH")

fiji_path = env_config.get("FIJI_PATH")
test_path = os.path.join(
    storage_path,
    "test/",
)  # repository used for tests
pastis_path = env_config.get("PASTIS_PATH")
temp_path = env_config.get("TEMP_PATH")

slurm_path = env_config.get("SLURM_PATH")
ml_path = os.path.join(storage_path, "models")
if not os.path.isdir(ml_path):
    os.mkdir(ml_path)
slurm_path = env_config.get("SLURM_PATH")
slurm_path_transfer = env_config.get("SLURM_PATH_transfer")
dropbox_path = env_config.get("DROPBOX_PATH")
path_analysis = env_config.get("PATH_ANALYSIS")


def pad_number(number):
    """
    Convert number to string and padd with a zero
    Ex:
    1 -> 01
    23 -> 23
    """
    if number < 10:
        return f"0{number}"
    else:
        return str(number)


def get_path(date, plate, skeleton, row=None, column=None, extension=".mat"):
    root_path = (
        r"//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE"
    )
    date_plate = f"/2020{date}"
    plate = f"_Plate{plate}"
    if skeleton:
        end = "/Analysis/Skeleton" + extension
    else:
        end = "/Img" + f"/Img_r{pad_number(row)}_c{pad_number(column)}.tif"
    return root_path + date_plate + plate + end


def get_dates_datetime(directory, plate):
    listdir = os.listdir(directory)
    list_dir_interest = [
        name
        for name in listdir
        if name.split("_")[-1] == f'Plate{0 if plate<10 else ""}{plate}'
    ]
    ss = [name.split("_")[0] for name in list_dir_interest]
    ff = [name.split("_")[1] for name in list_dir_interest]
    dates_datetime = [
        datetime(
            year=int(ss[i][:4]),
            month=int(ss[i][4:6]),
            day=int(ss[i][6:8]),
            hour=int(ff[i][0:2]),
            minute=int(ff[i][2:4]),
        )
        for i in range(len(list_dir_interest))
    ]
    dates_datetime.sort()
    return dates_datetime


def get_dirname(date, folders):
    select = folders.loc[folders["datetime"] == date]["folder"]
    assert len(select) == 1
    return select.iloc[0]


def shift_skeleton(skeleton, shift):
    shifted_skeleton = sparse.dok_matrix(skeleton.shape, dtype=bool)
    for pixel in skeleton.keys():
        #             print(pixel[0]+shift[0],pixel[1]+shift[1])
        if (
            skeleton.shape[0] > np.ceil(pixel[0] + shift[0]) > 0
            and skeleton.shape[1] > np.ceil(pixel[1] + shift[1]) > 0
        ):
            shifted_pixel = (
                np.round(pixel[0] + shift[0]),
                np.round(pixel[1] + shift[1]),
            )
            shifted_skeleton[shifted_pixel] = 1
    return shifted_skeleton


def transform_skeleton_final_for_show(skeleton_doc, Rot, trans):
    skeleton_transformed = {}
    transformed_keys = np.round(
        np.transpose(np.dot(Rot, np.transpose(np.array(list(skeleton_doc.keys())))))
        + trans
    ).astype(np.int)
    i = 0
    for pixel in list(transformed_keys):
        i += 1
        skeleton_transformed[(pixel[0], pixel[1])] = 1
    skeleton_transformed_sparse = sparse.lil_matrix((27000, 60000))
    for pixel in list(skeleton_transformed.keys()):
        i += 1
        skeleton_transformed_sparse[(pixel[0], pixel[1])] = 1
    return skeleton_transformed_sparse


def get_skeleton(exp, boundaries, t, directory):
    i = t
    plate = exp.prince_pos
    listdir = os.listdir(directory)
    dates = exp.dates
    date = dates[i]
    directory_name = get_dirname(date, exp.folders)
    path_snap = directory + directory_name
    skel = read_mat(path_snap + "/Analysis/skeleton_pruned_realigned.mat")
    skelet = skel["skeleton"]
    skelet = sparse_to_doc(skelet)
    Rot = skel["R"]
    trans = skel["t"]
    skel_aligned = transform_skeleton_final_for_show(
        skelet, np.array([[1, 0], [0, 1]]), np.array([0, 0])
    )
    output = skel_aligned[
        boundaries[2] : boundaries[3], boundaries[0] : boundaries[1]
    ].todense()
    kernel = np.ones((5, 5), np.uint8)
    output = cv2.dilate(output.astype(np.uint8), kernel, iterations=2)
    return (output, Rot, trans)


def get_param(
    folder, directory
):  # Very ugly but because interfacing with Matlab so most elegant solution.
    # TODO(FK)
    path_snap = os.path.join(directory, folder)
    file1 = open(os.path.join(path_snap, "param.m"), "r")
    Lines = file1.readlines()
    ldict = {}
    for line in Lines:
        to_execute = line.split(";")[0]
        relation = to_execute.split("=")
        if len(relation) == 2:
            ldict[relation[0].strip()] = relation[1].strip()
        # exec(line.split(';')[0],globals(),ldict)
    files = [
        "/Img/TileConfiguration.txt.registered",
        "/Analysis/skeleton_compressed.mat",
        "/Analysis/skeleton_masked_compressed.mat",
        "/Analysis/skeleton_pruned_compressed.mat",
        "/Analysis/transform.mat",
        "/Analysis/transform_corrupt.mat",
        "/Analysis/skeleton_realigned_compressed.mat",
        "/Analysis/nx_graph_pruned.p",
        "/Analysis/nx_graph_pruned_width.p",
        "/Analysis/nx_graph_pruned_labeled.p",
    ]
    for file in files:
        ldict[file] = os.path.isfile(path_snap + file)  # TODO(FK) change here
    return ldict


def update_plate_info_local(directory: str) -> None:
    """
    An acquisition repositorie has a param.m file inside it.
    :param directory: full path to a directory containing acquisition directories
    """
    target = env_config.get("DATA_PATH")
    listdir = os.listdir(directory)
    info_path = os.path.join(storage_path, "data_info.json")
    # TODO(FK): Crashes when there is no basic file
    # try:
    #     with open(target) as f:
    #         plate_info = json.load(f)
    # except:
    #     s = "/home/ipausers/kahane/Wks/AMFtrack/template_data_info.json"
    #     dest = os.path.join(storage_path, "data_info.json")
    #     shutil.copy(s, dest)
    #     with open(target) as f:
    #         plate_info = json.load(f)
    with open(info_path) as f:
        plate_info = json.load(f)

    # TOFIX
    for folder in listdir:
        if os.path.isfile(os.path.join(directory, folder, "param.m")):
            params = get_param(folder, directory)
            ss = folder.split("_")[0]
            ff = folder.split("_")[1]
            date = datetime(
                year=int(ss[:4]),
                month=int(ss[4:6]),
                day=int(ss[6:8]),
                hour=int(ff[0:2]),
                minute=int(ff[2:4]),
            )
            params["date"] = datetime.strftime(date, "%d.%m.%Y, %H:%M:")
            params["folder"] = folder
            total_path = os.path.join(directory, folder)
            plate_info[total_path] = params
    with open(target, "w") as jsonf:
        json.dump(plate_info, jsonf, indent=4)


def update_plate_info(
    directory: str, local=True, strong_constraint=True, suffix_data_info=""
) -> None:
    """*
    1/ Download `data_info.json` file containing all information about acquisitions.
    2/ Add all acquisition files in the `directory` path to the `data_info.json`.
    3/ Upload the new version of data_info (actuliased) to the dropbox.
    An acquisition repositorie has a param.m file inside it.
    """
    # TODO(FK): add a local version without dropbox modification
    listdir = os.listdir(directory)
    source = f"/data_info.json"
    target = os.path.join(temp_path, f"data_info{suffix_data_info}.json")
    if local:
        plate_info = {}
    else:
        download(source, target, end="")
        plate_info = json.load(open(target, "r"))
    with tqdm(total=len(listdir), desc="analysed") as pbar:
        for folder in listdir:
            path_snap = os.path.join(directory, folder)
            if os.path.exists(os.path.join(path_snap, "Img")):
                sub_list_files = os.listdir(os.path.join(path_snap, "Img"))
                is_real_folder = os.path.isfile(os.path.join(path_snap, "param.m"))
                if strong_constraint:
                    is_real_folder *= (
                        os.path.isfile(
                            os.path.join(path_snap, "Img", "Img_r03_c05.tif")
                        )
                        * len(sub_list_files)
                        >= 100
                    )
                if is_real_folder:
                    params = get_param(folder, directory)
                    ss = folder.split("_")[0]
                    ff = folder.split("_")[1]
                    date = datetime(
                        year=int(ss[:4]),
                        month=int(ss[4:6]),
                        day=int(ss[6:8]),
                        hour=int(ff[0:2]),
                        minute=int(ff[2:4]),
                    )
                    params["date"] = datetime.strftime(date, "%d.%m.%Y, %H:%M:")
                    params["folder"] = folder
                    total_path = os.path.join(directory, folder)
                    plate_info[total_path] = params
            pbar.update(1)
    with open(target, "w") as jsonf:
        json.dump(plate_info, jsonf, indent=4)
    if not local:
        upload(target, f"{source}", chunk_size=256 * 1024 * 1024)


def get_data_info(local=False, suffix_data_info=""):
    source = f"/data_info.json"
    target = os.path.join(temp_path, f"data_info{suffix_data_info}.json")

    if not local:
        download(source, target, end="")
    data_info = pd.read_json(target, convert_dates=True).transpose()
    if len(data_info) > 0:
        data_info.index.name = "total_path"
        data_info.reset_index(inplace=True)
        data_info["unique_id"] = (
            data_info["Plate"].astype(str).astype(int).astype(str)
            + "_"
            + data_info["CrossDate"].str.replace("'", "").astype(str)
        )

        data_info["datetime"] = pd.to_datetime(
            data_info["date"], format="%d.%m.%Y, %H:%M:"
        )
    return data_info


def get_current_folders_local(directory: str) -> pd.DataFrame:
    """
    :param directory: full path to a directory containing acquisition files
    """
    plate_info = pd.read_json(
        os.path.join(storage_path, "data_info.json"), convert_dates=True
    ).transpose()
    plate_info.index.name = "total_path"
    plate_info.reset_index(inplace=True)
    listdir = os.listdir(directory)
    selected_df = plate_info.loc[
        np.isin(plate_info["folder"], listdir)  # folder exist
        & (
            plate_info["total_path"]
            == plate_info["folder"].apply(
                lambda x: os.path.join(directory, x)
            )  # folder is registered
        )
    ]
    return selected_df


def get_current_folders(
    directory: str, local=True, suffix_data_info=""
) -> pd.DataFrame:
    """
    Returns a pandas data frame with all informations about the acquisition files
    inside the directory.
    WARNING: directory must finish with '/'
    """
    # TODO(FK): solve the / problem
    plate_info = get_data_info(local, suffix_data_info)
    listdir = os.listdir(directory)
    if len(plate_info) > 0:
        return plate_info.loc[
            np.isin(plate_info["folder"], listdir)
            & (plate_info["total_path"] == directory + plate_info["folder"])
        ]
    else:
        return plate_info


def get_folders_by_plate_id(plate_id, begin=0, end=-1, directory=None):
    data_info = get_data_info() if directory is None else get_current_folders(directory)
    folders = data_info.loc[
        10**8 * data_info["Plate"] + data_info["CrossDate"] == plate_id
    ]
    dates_datetime = [
        datetime.strptime(row["date"], "%d.%m.%Y, %H:%M:")
        for index, row in folders.iterrows()
    ]
    dates_datetime.sort()
    dates_datetime_select = dates_datetime[begin:end]
    dates_str = [
        datetime.strftime(date, "%d.%m.%Y, %H:%M:") for date in dates_datetime_select
    ]
    select_folders = folders.loc[np.isin(folders["date"], dates_str)]
    return select_folders


def update_analysis_info(directory, suffix_analysis_info=""):
    listdir = os.listdir(directory)
    analysis_dir = [fold for fold in listdir if fold.split("_")[0] == "Analysis"]
    infos_analysed = {}
    for folder in analysis_dir:
        metadata = {}
        version = folder.split("_")[-1]
        op_id = int(folder.split("_")[-2])
        dt = datetime.fromtimestamp(op_id // 1000000000)
        path = f"{directory}{folder}/folder_info.json"
        if os.path.exists(path):
            infos = pd.read_json(path, dtype={"unique_id": str})
            if len(infos) > 0:
                column_interest = [
                    column for column in infos.columns if column[0] != "/"
                ]
                metadata["version"] = version
                for column in column_interest:
                    if column != "folder":
                        info = str(infos[column].iloc[0])
                        metadata[column] = info
                metadata["date_begin"] = datetime.strftime(
                    infos["datetime"].iloc[0], "%d.%m.%Y, %H:%M:"
                )
                metadata["date_end"] = datetime.strftime(
                    infos["datetime"].iloc[-1], "%d.%m.%Y, %H:%M:"
                )
                metadata["number_timepoints"] = len(infos)
                metadata["path_exp"] = f"{folder}/experiment.pick"
                metadata["path_global_hypha_info"] = f"{folder}/global_hypha_info.json"
                metadata["path_time_hypha_info"] = f"{folder}/time_hypha_info"
                metadata["path_time_plate_info"] = f"{folder}/time_plate_info.json"
                metadata["path_global_plate_info"] = f"{folder}/global_plate_info.json"
                metadata["date_run_analysis"] = datetime.strftime(
                    dt, "%d.%m.%Y, %H:%M:"
                )
                infos_analysed[folder] = metadata
        else:
            print(folder)
    target = os.path.join(temp_path, f"analysis_info{suffix_analysis_info}.json")

    with open(target, "w") as jsonf:
        json.dump(infos_analysed, jsonf, indent=4)


def get_analysis_info(directory, suffix_analysis_info=""):
    target = os.path.join(temp_path, f"analysis_info{suffix_analysis_info}.json")

    analysis_info = pd.read_json(target, convert_dates=True).transpose()
    analysis_info.index.name = "folder_analysis"
    analysis_info.reset_index(inplace=True)
    return analysis_info


def get_analysis_folders(path=path_analysis):
    analysis_folders_list = []  # Store each infos DataFrame here

    for dire in os.walk(path):
        name_analysis = dire[0].split(os.sep)[-1].split("_")
        if name_analysis[0] == "Analysis":
            analysis_dir = dire[0]
            path_save = os.path.join(analysis_dir, "folder_info.json")
            if os.path.exists(path_save):
                folders_plate = pd.read_json(path_save)
                infos = folders_plate.iloc[0][1:10].copy()  # Copy to avoid SettingWithCopyWarning
                infos["total_path"] = analysis_dir
                infos["time_plate"] = os.path.isfile(
                    os.path.join(analysis_dir, "time_plate_info.json")
                )
                infos["global_hypha"] = os.path.isfile(
                    os.path.join(analysis_dir, "global_hypha_info.json")
                )
                infos["time_hypha"] = os.path.isdir(
                    os.path.join(analysis_dir, "time_hypha_info.json")
                )
                infos["num_folders"] = len(folders_plate)

                # Append the infos DataFrame to the list
                analysis_folders_list.append(infos)

    # Concatenate all infos DataFrames at once
    analysis_folders = pd.concat(analysis_folders_list, axis=1).transpose().reset_index(drop=True)
    analysis_folders["unique_id"] = (
            analysis_folders["Plate"].astype(str)
            + "_"
            + analysis_folders["CrossDate"].astype(str).str.replace("'", "")
    )

    return analysis_folders

def get_time_plate_info_from_analysis(analysis_folders, use_saved=True):
    plates_in = analysis_folders["unique_id"].unique()
    plates_in.sort()
    ide = hashlib.sha256(np.sum(plates_in).encode("utf-8")).hexdigest()
    path_save_info = os.path.join(temp_path, f"time_plate_info_{ide}")
    path_save_folders = os.path.join(temp_path, f"folders_{ide}")

    if os.path.exists(path_save_info) and use_saved:
        time_plate_info = pd.read_json(path_save_info)
        folders = pd.read_json(path_save_folders)
        return folders, time_plate_info

    analysis_dirs = analysis_folders["total_path"]
    time_plate_info_list = []  # Store each table as a separate DataFrame in a list
    folders_list = []  # Store each folders_plate DataFrame here

    for analysis_dir in analysis_dirs:
        path_save = os.path.join(analysis_dir, "time_plate_info.json")
        table = pd.read_json(path_save).transpose().fillna(-1)

        path_save = os.path.join(analysis_dir, "folder_info.json")
        folders_plate = pd.read_json(path_save).reset_index()

        # Calculate time since beginning and add as new columns
        time_since_begin = folders_plate["datetime"] - folders_plate["datetime"].iloc[0]
        table["time_since_begin"] = time_since_begin
        table["time_since_begin_h"] = time_since_begin / np.timedelta64(1, "h")
        table["timestep"] = folders_plate.index.values
        table["folder"] = folders_plate["folder"]
        table["unique_id"] = folders_plate["unique_id"]
        table["datetime"] = folders_plate["datetime"]

        # Add optional columns if they exist
        for column in ["PrincePos", "root", "strain", "medium"]:
            if column in folders_plate:
                table[column] = folders_plate[column]

        time_plate_info_list.append(table)  # Append to the list instead of concatenating immediately
        folders_list.append(folders_plate)  # Append folders_plate to list

    # Concatenate all tables at once
    time_plate_info = pd.concat(time_plate_info_list, ignore_index=True)
    folders = pd.concat(folders_list, ignore_index=True)

    # Save to JSON files
    time_plate_info.to_json(path_save_info)
    folders.to_json(path_save_folders)

    return folders, time_plate_info


def get_global_hypha_info_from_analysis(analysis_folders, use_saved=True):
    plates_in = analysis_folders["unique_id"].unique()
    plates_in.sort()
    ide = hashlib.sha256(np.sum(plates_in).encode("utf-8")).hexdigest()
    path_save_info = os.path.join(temp_path, f"global_hypha_info_{ide}")
    path_save_folders = os.path.join(temp_path, f"folders_{ide}")

    if os.path.exists(path_save_info) and use_saved:
        global_hypha_info = pd.read_json(path_save_info)
        folders = pd.read_json(path_save_folders)
        return (folders, global_hypha_info)
    analysis_dirs = analysis_folders["total_path"]
    global_hypha_info = pd.DataFrame()
    folders = pd.DataFrame()
    for analysis_dir in analysis_dirs:
        path_save = os.path.join(analysis_dir, "global_hypha_info.json")
        if os.path.exists(path_save):
            table = pd.read_json(path_save)
            table = table.transpose()
            table = table.fillna(-1)
            path_save = os.path.join(analysis_dir, "folder_info.json")
            folders_plate = pd.read_json(path_save)
            folders_plate = folders_plate.reset_index()
            table = table.reset_index()
            table["unique_id"] = folders_plate["unique_id"].iloc[0]
            table["PrincePos"] = folders_plate["PrincePos"].iloc[0]
            table["root"] = folders_plate["root"].iloc[0]
            table["strain"] = folders_plate["strain"].iloc[0]
            table["medium"] = folders_plate["medium"].iloc[0]
            global_hypha_info = pd.concat([global_hypha_info, table], ignore_index=True)
            folders = pd.concat(
                [folders.copy(), folders_plate.copy()], axis=0, ignore_index=True
            )
    global_hypha_info.to_json(path_save_info)
    folders.to_json(path_save_folders)
    return (folders, global_hypha_info)


def get_time_hypha_info_from_analysis(analysis_folders, use_saved=True):
    plates_in = analysis_folders["unique_id"].unique()
    plates_in.sort()
    ide = hashlib.sha256(np.sum(plates_in).encode("utf-8")).hexdigest()
    path_save_info = os.path.join(temp_path, f"time_hypha_info_{ide}")
    path_save_folders = os.path.join(temp_path, f"folders_{ide}")

    if os.path.exists(path_save_info) and use_saved:
        time_hypha_infos = pd.read_json(path_save_info)
        folders = pd.read_json(path_save_folders)
        return (folders, time_hypha_infos)
    analysis_dirs = analysis_folders["total_path"]
    folders = pd.DataFrame()
    time_hypha_infos = []
    for analysis_dir in analysis_dirs:
        path_time_hypha = os.path.join(analysis_dir, "time_hypha_info")
        if os.path.exists(path_time_hypha):
            path_save = os.path.join(analysis_dir, "folder_info.json")
            folders_plate = pd.read_json(path_save)
            folders_plate = folders_plate.reset_index()
            folders_plate = folders_plate.sort_values("datetime")
            json_paths = os.listdir(path_time_hypha)
            tables = []
            for path in json_paths:
                index = int(path.split("_")[-1].split(".")[0])
                line = folders_plate.iloc[index]
                try:
                    table = pd.read_json(os.path.join(path_time_hypha, path))
                except:
                    print(os.path.join(path_time_hypha, path))
                    continue
                table = table.transpose()
                table = table.fillna(-1)
                table["time_since_begin_h"] = (
                    line["datetime"] - folders_plate["datetime"].iloc[0]
                )
                table["folder"] = line["folder"]
                table["Plate"] = line["Plate"]
                table["unique_id"] = line["unique_id"]
                table["datetime"] = line["datetime"]
                table["PrincePos"] = line["PrincePos"]
                table["root"] = line["root"]
                table["strain"] = line["strain"]
                table["medium"] = line["medium"]
                tables.append(table)
            time_hypha_info_plate = pd.concat(tables, axis=0, ignore_index=True)
            time_hypha_info_plate.reset_index(inplace=True, drop=True)
            time_hypha_infos.append(time_hypha_info_plate)
            folders = pd.concat([folders, folders_plate], axis=0, ignore_index=True)
    time_hypha_info = pd.concat(time_hypha_infos, axis=0, ignore_index=True)
    time_hypha_info.to_json(path_save_info)
    folders.to_json(path_save_folders)
    return (folders, time_hypha_info)


def get_time_edge_info_from_analysis(analysis_folders, use_saved=True):
    plates_in = analysis_folders["unique_id"].unique()
    plates_in.sort()
    ide = hashlib.sha256(np.sum(plates_in).encode("utf-8")).hexdigest()
    path_save_info = os.path.join(temp_path, f"time_edge_info_{ide}")
    path_save_folders = os.path.join(temp_path, f"folders_{ide}")

    if os.path.exists(path_save_info) and use_saved:
        time_edge_infos = pd.read_json(path_save_info)
        folders = pd.read_json(path_save_folders)
        return (folders, time_edge_infos)
    analysis_dirs = analysis_folders["total_path"]
    folders = pd.DataFrame()
    time_edge_infos = []
    for analysis_dir in analysis_dirs:
        path_time_edge = os.path.join(analysis_dir, "time_edge_info")
        if os.path.exists(path_time_edge):
            path_save = os.path.join(analysis_dir, "folder_info.json")
            folders_plate = pd.read_json(path_save)
            folders_plate = folders_plate.reset_index()
            folders_plate = folders_plate.sort_values("datetime")
            json_paths = os.listdir(path_time_edge)
            tables = []
            for path in json_paths:
                index = int(path.split("_")[-1].split(".")[0])
                line = folders_plate.iloc[index]
                try:
                    table = pd.read_json(os.path.join(path_time_edge, path))
                except:
                    print(os.path.join(path_time_edge, path))
                    continue
                table = table.transpose()
                table = table.fillna(-1)
                table["time_since_begin_h"] = (
                    line["datetime"] - folders_plate["datetime"].iloc[0]
                )
                table["folder"] = line["folder"]
                table["Plate"] = line["Plate"]
                table["unique_id"] = line["unique_id"]
                table["datetime"] = line["datetime"]
                for column in ["PrincePos", "root", "strain", "medium"]:
                    try:
                        table[column] = line[column]
                    except KeyError:
                        continue

                tables.append(table)
            time_edge_info_plate = pd.concat(tables, axis=0, ignore_index=True)
            time_edge_info_plate.reset_index(inplace=True, drop=True)
            time_edge_infos.append(time_edge_info_plate)
            folders = pd.concat([folders, folders_plate], axis=0, ignore_index=True)
    time_edge_info = pd.concat(time_edge_infos, axis=0, ignore_index=True)
    time_edge_info.to_json(path_save_info)
    folders.to_json(path_save_folders)
    return (folders, time_edge_info)


def get_time_plate_info_long_from_analysis(analysis_folders, use_saved=True):
    plates_in = analysis_folders["unique_id"].unique()
    plates_in.sort()
    ide = hashlib.sha256(np.sum(plates_in).encode("utf-8")).hexdigest()
    path_save_info = os.path.join(temp_path, f"time_plate_info_long_{ide}")
    path_save_folders = os.path.join(temp_path, f"folders_{ide}")

    if os.path.exists(path_save_info) and use_saved:
        time_plate_infos = pd.read_json(path_save_info)
        folders = pd.read_json(path_save_folders)
        return (folders, time_plate_infos)
    analysis_dirs = analysis_folders["total_path"]
    folders = pd.DataFrame()
    time_plate_infos = []
    for analysis_dir in analysis_dirs:
        path_time_plate = os.path.join(analysis_dir, "time_plate_info_long")
        if os.path.exists(path_time_plate):
            path_save = os.path.join(analysis_dir, "folder_info.json")
            folders_plate = pd.read_json(path_save)
            folders_plate = folders_plate.reset_index()
            folders_plate = folders_plate.sort_values("datetime")
            json_paths = os.listdir(path_time_plate)
            tables = []
            for path in json_paths:
                index = int(path.split("_")[-1].split(".")[0])
                line = folders_plate.iloc[index]
                try:
                    table = pd.read_json(
                        os.path.join(path_time_plate, path), orient="index"
                    )
                except:
                    print(os.path.join(path_time_plate, path))
                    continue
                table = table.transpose()
                table = table.fillna(-1)
                table["time_since_begin_h"] = (
                    line["datetime"] - folders_plate["datetime"].iloc[0]
                )
                table["folder"] = line["folder"]
                table["Plate"] = line["Plate"]
                table["unique_id"] = line["unique_id"]
                table["datetime"] = line["datetime"]
                for column in ["PrincePos", "root", "strain", "medium"]:
                    try:
                        table[column] = line[column]
                    except KeyError:
                        continue

                tables.append(table)
            time_plate_info_plate = pd.concat(tables, axis=0, ignore_index=True)
            time_plate_info_plate = time_plate_info_plate.sort_values("datetime")
            time_plate_info_plate.reset_index(inplace=True, drop=True)
            time_plate_info_plate["timestep"] = time_plate_info_plate.index

            time_plate_infos.append(time_plate_info_plate)
            folders = pd.concat([folders, folders_plate], axis=0, ignore_index=True)
    time_plate_info = pd.concat(time_plate_infos, axis=0, ignore_index=True)
    time_plate_info.to_json(path_save_info)
    folders.to_json(path_save_folders)
    return (folders, time_plate_info)


def get_data_tables(op_id=time_ns(), redownload=True):
    API = str(np.load(os.getenv("HOME") + "/pycode/API_drop.npy"))
    dir_drop = "data_tables"
    root = os.getenv("TEMP")
    # op_id = time_ns()
    if redownload:
        path_save = f"{root}global_hypha_info{op_id}.pick"
        download(f"/{dir_drop}/global_hypha_infos.pick", path_save)
        path_save = f"{root}time_plate_infos{op_id}.pick"
        download(f"/{dir_drop}/time_plate_infos.pick", path_save)
        path_save = f"{root}time_hypha_info{op_id}.pick"
        download(f"/{dir_drop}/time_hypha_infos.pick", path_save)
    path_save = f"{root}time_plate_infos{op_id}.pick"
    time_plate_info = pd.read_pickle(path_save)
    path_save = f"{root}global_hypha_info{op_id}.pick"
    global_hypha_info = pd.read_pickle(path_save)
    path_save = f"{root}time_hypha_info{op_id}.pick"
    time_hypha_info = pd.read_pickle(path_save)
    return (time_plate_info, global_hypha_info, time_hypha_info)


if __name__ == "__main__":
    directory = r"/data/felix/"
    update_plate_info(directory)