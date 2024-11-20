import copy
import pandas as pd
from IPython.display import clear_output
import re
from amftrack.pipeline.functions.transport_processing.high_mag_videos.kymo_class import (
    KymoEdgeAnalysis,
    KymoVideoAnalysis,
)
from amftrack.pipeline.functions.transport_processing.high_mag_videos.plot_data import (
    save_raw_data,
    plot_summary,
)
import matplotlib.patches as mpatches
from pathlib import Path, PurePath
import sys
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite, imread
from tqdm import tqdm
import matplotlib as mpl
from amftrack.util.dbx import (
    upload_folder,
    download,
    read_saved_dropbox_state,
    save_dropbox_state,
    load_dbx,
    download,
    get_dropbox_folders_prince,
    get_dropbox_video_folders,
    get_dropbox_video_folders_new,
)

import logging
import datetime
import numpy as np
import matplotlib.patheffects as pe

#
# # logging.basicConfig(stream=sys.stdout, level=logging.debug)
mpl.rcParams["figure.dpi"] = 200


def month_to_num(x: str):
    """
    Takes a string with the name of a month, and returns that month's corresponding number as a string.
    :param x:   String, preferably of a month
    :return:    Two-digit string number of that month
    """
    months = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    a = x.strip()[:3].lower()
    try:
        ez = months[a]
        return ez
    except:
        raise ValueError("Not a month")


def index_videos_dropbox_new(
    analysis_folder,
    videos_folder,
    dropbox_folder,
    REDO_SCROUNGING=False,
    CREATE_VIDEO_HIERARCHY=True,
    date_start: int = None,
    date_end: int = None,
    plate_names: np.array = None,
):
    """
    Goes through the specified dropbox folder, and collects and organises all the relevant video data for all videos
    stored in that dropbox. Works recursively. On the local machine will also create a folder structure in the
    analysis and videos folder, populating the analysis folder with the video info. Returns a merged pandas dataframe
    with all video information.
    :param analysis_folder:         Local address where video info, and later analysis will be stored
    :param videos_folder:           Local address where raw video data will be stored later
    :param dropbox_folder:          Folder address on the dropbox with all videos you want to process
    :param REDO_SCROUNGING:         Boolean whether to redo the dropbox searching. Dropbox will be searched anyway if cache .json cannot be found
    :param CREATE_VIDEO_HIERARCHY:  Boolean whether to create the video hierarchy. Only set to False if you don't plan on downloading the raw data
    :return:                        Pandas dataframe with all video information
    """

    # Using PathLib allows for more OS-ambiguous functionality

    analysis_folder = Path(analysis_folder)
    dropbox_folder = Path(dropbox_folder)
    videos_folder = Path(videos_folder)

    analysis_json = analysis_folder.joinpath(
        dropbox_folder.relative_to("/DATA/")
    ).joinpath("all_folders_drop.json")
    excel_json = analysis_folder.joinpath(
        dropbox_folder.relative_to("/DATA/")
    ).joinpath("excel_drop.json")

    # First analysis_json and excel_json will be sought for, these serve as caches for the dropbox data.
    # If they can't be found, the information will be assembled from the indexing files on the Dropbox.
    #
    # if os.path.exists(analysis_json):
    #     all_folders_drop = pd.read_json(analysis_json)
    if os.path.exists(excel_json):
        excel_drop = pd.read_json(excel_json, typ="series")
    if not os.path.exists(analysis_json) or REDO_SCROUNGING:
        print("Redoing the dropbox scrounging, hold on tight.")
        excel_drop, txt_drop = get_dropbox_video_folders_new(
            dropbox_folder,
            date_start=date_start,
            date_end=date_end,
            plate_names=plate_names,
        )
        if not analysis_folder.joinpath(dropbox_folder.relative_to("/DATA/")).exists():
            analysis_folder.joinpath(dropbox_folder.relative_to("/DATA/")).mkdir(
                parents=True
            )

        #         clear_output(wait=False)
        print("Scrounging complete, downloading files...")

        excel_addresses = np.array(
            [
                re.search("^.*Plate.*\/.*Plate\d{2,6}\..*$", entry, re.IGNORECASE)
                for entry in excel_drop
            ]
        )
        print("adresses", excel_addresses)
        if len(excel_addresses) > 0:
            excel_addresses = [i for i in excel_addresses if i is not None]
            print(excel_addresses)
            excel_addresses = [address.group(0) for address in excel_addresses]
        excel_drop = np.concatenate([excel_addresses, txt_drop])
        pd.Series(excel_drop).to_json(excel_json)
    info_addresses = []

    # Once addresses have been found, all relevant index files will be downloaded to their respective position

    for address in tqdm(excel_drop):
        address_local = Path(address).relative_to("/DATA/")
        if REDO_SCROUNGING:
            if not (analysis_folder / address_local.parent).exists():
                (analysis_folder / address_local.parent).mkdir(parents=True)
            # if not (analysis_folder / address_local).exists():
            download(address, (analysis_folder / address_local))
        info_addresses.append(analysis_folder / address_local)
    # clear_output(wait=False)
    print("All files downloaded! Merging files...")

    # The downloaded information is then read and merged into one dataframe containing all relevant information.

    merge_frame = read_video_data_new(info_addresses, analysis_folder)
    merge_frame = merge_frame.rename(
        columns={"tot_path": "folder"}
    )  # This is for the dropbox download functions
    merge_frame = merge_frame.sort_values("unique_id")
    merge_frame = merge_frame.reset_index(drop=True)
    merge_frame = merge_frame.loc[:, ~merge_frame.columns.duplicated()].copy()
    merge_frame["analysis_folder"] = [np.nan for i in range(len(merge_frame))]
    merge_frame["videos_folder"] = [np.nan for i in range(len(merge_frame))]

    for index, row in merge_frame.iterrows():
        # the Img part at the end was taken out, but running bulk analysis prefers them in an Img folder
        target_anals_file = analysis_folder / row["folder"]  # [:-4]
        target_video_file = videos_folder / row["folder"]

        row.loc["analysis_folder"] = target_anals_file.as_posix()
        row.loc["videos_folder"] = target_video_file.as_posix()

        if not target_video_file.exists() and CREATE_VIDEO_HIERARCHY:
            target_video_file.mkdir(parents=True)
        if not target_anals_file.exists() and CREATE_VIDEO_HIERARCHY:
            target_anals_file.mkdir(parents=True)
        row.to_json(target_anals_file / "video_data.json", orient="index")

    return merge_frame


def read_video_data_new(address_array, analysis_folder):
    """
    Takes a list of index files, a frame with basic video information and the analysis folder to merge the
    information altogether into a readable pandas dataframe with consistent columns and values.
    :param address_array:   List of addresses pointing to downloaded .xslx, .csv and .txt files.
    :param folders_frame:   Frame with basic video parameters which will be expanded, created with the dropbox download
                            functions
    :param analysis_folder: Folder where all of this will be found. Highest folder in the hierarchy.
    :return:                Merged frame with all video information that could be found in the index files.
    """

    # Excel and cvs files have different information, and are stored in different hierarchies.

    excel_frame = pd.DataFrame()
    csv_frame = pd.DataFrame()
    txt_frame = pd.DataFrame()
    for address in tqdm(address_array):
        suffix = address.suffix
        if suffix == ".xlsx":
            raw_data = pd.read_excel(address)
            # Earliest .xlsx files did not contain binning information. Assume 1x1 binning
            if "Binned (Y/N)" not in raw_data:
                if "Unnamed: 0" in raw_data:
                    raw_data["Binned (Y/N)"] = ["N" for entry in raw_data["Unnamed: 0"]]
                elif "Binned" in raw_data:
                    raw_data["Binned (Y/N)"] = [
                        "Y" if entry == 2 else "N" for entry in raw_data["Binned"]
                    ]
                else:
                    raw_data["Binned (Y/N)"] = "N"
            raw_data["Binned (Y/N)"] = raw_data["Binned (Y/N)"].astype(str)
            # Filter on excel rows that actually contain data
            raw_data = raw_data[
                raw_data["Treatment"] == raw_data["Treatment"]
            ].reset_index(drop=True)
            # 'Plate[nr] can be both upper case and lower case'
            raw_data["Unnamed: 0"] = [entry.lower() for entry in raw_data["Unnamed: 0"]]
            try:
                raw_data["plate_id"] = [
                    f"{entry.split('_')[-3]}_Plate{entry.split('_')[-2][5:]}"
                    for entry in raw_data["Unnamed: 0"]
                ]
                raw_data["imaging_day"] = [
                    f"{entry.split('_')[-3]}" for entry in raw_data["Unnamed: 0"]
                ]
            except:
                pass
            raw_data["Position mm"] = [
                float(entry) / 1000.0 for entry in raw_data["Position mm"]
            ]
            raw_data["Unnamed: 6"] = [
                float(entry) / 1000.0 for entry in raw_data["Unnamed: 6"]
            ]
            raw_data["tot_path_drop"] = [
                f"/DATA/{address.relative_to(analysis_folder).parent.as_posix()}/DATA/{row['Unnamed: 0']}"
                for index, row in raw_data.iterrows()
            ]
            excel_frame = pd.concat([excel_frame, raw_data])

        elif suffix == ".csv":
            # Different OS languages create different CSV's. Below checks for comma, or semicolon separation
            df_comma = pd.read_csv(address, nrows=1, sep=",")
            df_semi = pd.read_csv(address, nrows=1, sep=";")
            if df_comma.shape[1] > df_semi.shape[1]:
                raw_data = pd.read_csv(address, sep=",")
            else:
                raw_data = pd.read_csv(address, sep=";")
            raw_data["file_name"] = [address.stem] * len(raw_data)
            raw_data["unique_id"] = [
                f"{address.parent.parts[-1]}_{str(row['video']).zfill(3)}"
                for index, row in raw_data.iterrows()
            ]
            raw_data["imaging_day"] = [
                entry.split("_")[0] for entry in raw_data["unique_id"]
            ]
            raw_data["plate_id"] = [
                f"{entry.split('_')[-3]}_Plate{entry.split('_')[-2][5:]}"
                for entry in raw_data["unique_id"]
            ]
            raw_data["tot_path_drop"] = [
                f"/DATA/{address.relative_to(analysis_folder).parent.as_posix()}/{str(row['video']).zfill(3)}/Img"
                for index, row in raw_data.iterrows()
            ]
            raw_data["tot_path"] = [entry[6:] for entry in raw_data["tot_path_drop"]]
            raw_data = raw_data.reset_index()
            csv_frame = pd.concat([csv_frame, raw_data], axis=0, ignore_index=True)

        elif suffix == ".txt":
            # If VideoInfo is missing, likely that the video was not worth saving anyway
            if not address.exists():
                print(f"Could not find {address}, skipping for now")
                continue
            # Read .txt with read_csv? More likely than you think!
            raw_data = pd.read_csv(address, sep=": ", engine="python").T
            # Drop all columns with no data
            raw_data = raw_data.dropna(axis=1, how="all")
            raw_data["unique_id"] = [f"{address.parts[-3]}_{address.parts[-2]}"]
            raw_data["tot_path"] = (
                address.relative_to(analysis_folder).parent / "Img"
            ).as_posix()
            raw_data["tot_path_drop"] = ["DATA/" + raw_data["tot_path"][0]]
            if raw_data["Operation"].to_string().split(" ")[-1] == "Undetermined":
                print(
                    f"Undetermined operation in {raw_data['unique_id'].to_string().split(' ')[-1]}, please amend. Assuming 50x BF."
                )
                raw_data["Operation"] = "  50x Brightfield"
            try:
                txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)
            except:
                print(f"Weird concatenation with {address}, trying to reset index")
                print(raw_data.columns)
                txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)

    # Now that three different kinds of data have been recorded, it's time to merge all of them.
    # A definite final product is not standardized, but below you can see the renaming schemes.
    # These new names will be used in the bulk analysis.

    if len(excel_frame) > 0:
        excel_frame["Binned (Y/N)"] = [
            np.where(entry == "Y", 2, 1) for entry in excel_frame["Binned (Y/N)"]
        ]
        excel_frame["Binned (Y/N)"] = excel_frame["Binned (Y/N)"].astype(int)
        excel_frame["Time after crossing"] = [
            int(entry.split(" ")[-2]) for entry in excel_frame["Time after crossing"]
        ]
        excel_frame = excel_frame.rename(
            columns={
                "Unnamed: 0": "unique_id",
                "Treatment": "treatment",
                "Strain": "strain",
                "Time after crossing": "days_after_crossing",
                "Growing temperature": "grow_temp",
                "Position mm": "xpos",
                "Unnamed: 6": "ypos",
                "dcenter mm": "dcenter",
                "droot mm": "droot",
                "Bright-field (BF)\nor\nFluorescence (F)": "mode",
                "Binned (Y/N)": "binning",
                "Magnification": "magnification",
                "FPS": "fps",
                "Video Length (s)": "time_(s)",
                "Comments": "comments",
            }
        )
    if len(txt_frame) > 0:
        txt_frame = txt_frame.dropna(axis=1, how="all")
        txt_frame = txt_frame.drop(
            [
                "Computer",
                "User",
                "DataRate",
                "DataSize",
                "Frames Recorded",
                "Fluorescence",
                "Four Led Bar",
                "Model",
                "FrameSize",
            ],
            axis=1,
        )
        txt_frame["record_time"] = [
            entry.split(",")[-1] for entry in txt_frame["DateTime"]
        ]
        txt_frame["DateTime"] = [
            f"{entry.split(', ')[1].split(' ')[-1]}{month_to_num(entry.split(', ')[-2].split(' ')[-2])}{entry.split(', ')[1].split(' ')[-3]}"
            for entry in txt_frame["DateTime"]
        ]
        txt_frame["CrossDate"] = [str(int(entry)) for entry in txt_frame["CrossDate"]]
        txt_frame["days_after_crossing"] = [
            (
                datetime.date(
                    int(row["DateTime"][:4]),
                    int(row["DateTime"][4:6]),
                    int(row["DateTime"][6:]),
                )
                - datetime.date(
                    int(row["CrossDate"][:4]),
                    int(row["CrossDate"][4:6]),
                    int(row["CrossDate"][6:]),
                )
            ).days
            for index, row in txt_frame.iterrows()
        ]

        txt_frame["X"] = [
            float(entry.split("  ")[-1].split(" ")[0]) for entry in txt_frame["X"]
        ]
        txt_frame["Y"] = [
            float(entry.split("  ")[-1].split(" ")[0]) for entry in txt_frame["Y"]
        ]
        txt_frame["Z"] = [
            float(entry.split("  ")[-1].split(" ")[0]) for entry in txt_frame["Z"]
        ]

        txt_frame["Binning"] = [int(entry[-1]) for entry in txt_frame["Binning"]]
        txt_frame["FrameRate"] = [
            float(entry.split(" ")[-3]) for entry in txt_frame["FrameRate"]
        ]
        txt_frame["magnification"] = [
            float(entry.split(" ")[-2][:-1]) for entry in txt_frame["Operation"]
        ]
        txt_frame["Operation"] = [
            str(np.where(entry.split(" ")[-1] == "Brightfield", "BF", "F"))
            for entry in txt_frame["Operation"]
        ]
        txt_frame["Time"] = [float(entry.split(" ")[-2]) for entry in txt_frame["Time"]]

        txt_frame["ExposureTime"] = [
            entry.split("  ")[-1].split(" ")[1] for entry in txt_frame["ExposureTime"]
        ]
        txt_frame["ExposureTime"] = pd.to_numeric(
            txt_frame["ExposureTime"], errors="coerce"
        )
        txt_frame["Run"] = [int(entry) for entry in txt_frame["Run"]]
        txt_frame["Gain"] = [float(entry) for entry in txt_frame["Gain"]]
        txt_frame["Gamma"] = [float(entry) for entry in txt_frame["Gamma"]]
        # if there is no root this doesn't work. Needs to be updated
        #         txt_frame["Root"] = [entry.split(" ")[-1] for entry in txt_frame["Root"]]
        txt_frame["Strain"] = [entry.split(" ")[-1] for entry in txt_frame["Strain"]]
        txt_frame["StoragePath"] = [
            entry.split(" ")[-1] for entry in txt_frame["StoragePath"]
        ]
        txt_frame["Treatment"] = [
            entry.split(" ")[-1] for entry in txt_frame["Treatment"]
        ]
        txt_frame = txt_frame.rename(
            columns={
                "DateTime": "imaging_day",
                "StoragePath": "storage_path",
                "Plate": "plate_id",
                "Root": "root",
                "Strain": "strain",
                "Treatment": "treatment",
                "CrossDate": "crossing_day",
                "Run": "video_int",
                "Time": "time_(s)",
                "Operation": "mode",
                "ExposureTime": "exposure_time_(us)",
                "FrameRate": "fps",
                "Binning": "binning",
                "Gain": "gain",
                "Gamma": "gamma",
                "X": "xpos",
                "Y": "ypos",
                "Z": "zpos",
            }
        )

    if len(csv_frame) > 0:
        csv_frame["video_id"] = [
            entry.split("_")[-1] for entry in csv_frame["unique_id"]
        ]
        csv_frame["plate_nr"] = [
            int(entry.split("_")[-2][5:]) for entry in csv_frame["unique_id"]
        ]
        csv_frame["Lens"] = csv_frame["Lens"].astype(float)
        csv_frame["fps"] = csv_frame["fps"].astype(float)
        csv_frame["time"] = csv_frame["time"].astype(float)
        csv_frame = csv_frame.rename(
            columns={
                "video": "video_int",
                "Treatment": "treatment",
                "Strain": "strain",
                "tGermination": "days_after_crossing",
                "tCrossing": "days_after_crossing",
                "Illumination": "mode",
                "Binned": "binning",
                "Lens": "magnification",
                "plate_id_xl": "plate_id",
                "time": "time_(s)",
            }
        )
        csv_frame = csv_frame.drop(columns=["index", "file_name"], axis=1)
    if len(csv_frame) > 0 and len(txt_frame) > 0:
        merge_frame = pd.merge(
            txt_frame, csv_frame, how="outer", on="unique_id", suffixes=("", "_csv")
        )
        merge_frame = merge_frame.drop(columns=["plate_id"], axis=1)
        merge_frame = merge_frame.rename(columns={"plate_id_xl": "plate_id"})
        merge_frame["imaging_day"] = merge_frame["imaging_day"].fillna(
            merge_frame["imaging_day_csv"]
        )
        merge_frame["strain"] = merge_frame["strain"].fillna(merge_frame["strain_csv"])
        merge_frame["treatment"] = merge_frame["treatment"].fillna(
            merge_frame["treatment_csv"]
        )
        merge_frame["video_int"] = merge_frame["video_int"].fillna(
            merge_frame["video_int_csv"]
        )
        merge_frame["time_(s)"] = merge_frame["time_(s)"].fillna(
            merge_frame["time_(s)_csv"]
        )
        merge_frame["mode"] = merge_frame["mode"].fillna(merge_frame["mode_csv"])
        merge_frame["fps"] = merge_frame["fps"].fillna(merge_frame["fps_csv"])
        merge_frame["binning"] = merge_frame["binning"].fillna(
            merge_frame["binning_csv"]
        )
        merge_frame["xpos"] = merge_frame["xpos"].fillna(merge_frame["xpos_csv"])
        merge_frame["ypos"] = merge_frame["ypos"].fillna(merge_frame["ypos_csv"])
        merge_frame["magnification"] = merge_frame["magnification"].fillna(
            merge_frame["magnification_csv"]
        )
        merge_frame["tot_path"] = merge_frame["tot_path"].fillna(
            merge_frame["tot_path_csv"]
        )
        merge_frame["days_after_crossing"] = merge_frame["days_after_crossing"].fillna(
            merge_frame["days_after_crossing_csv"]
        )
        merge_frame = merge_frame.drop(
            columns=[
                "video_int_csv",
                "treatment_csv",
                "strain_csv",
                "days_after_crossing_csv",
                "xpos_csv",
                "ypos_csv",
                "mode_csv",
                "binning_csv",
                "magnification_csv",
                "fps_csv",
                "plate_id_csv",
                "tot_path_csv",
                "time_(s)_csv",
                "imaging_day_csv",
                "tot_path_drop_csv",
            ],
            axis=1,
        )

    elif len(excel_frame) > 0 and len(txt_frame) > 0:
        merge_frame = pd.merge(
            excel_frame, csv_frame, how="left", on="unique_id", suffixes=("", "_csv")
        )
    elif len(txt_frame) > 0:
        merge_frame = txt_frame
        merge_frame["plate_id"] = [
            f"{row['imaging_day']}_Plate{int(row['plate_id'])}"
            for index, row in merge_frame.iterrows()
        ]
    elif len(excel_frame) > 0:
        merge_frame = excel_frame.reset_index(drop=True)
        merge_frame = merge_frame[
            merge_frame["tot_path_drop"] == merge_frame["tot_path_drop"]
        ]
        merge_frame["tot_path"] = [
            (Path(entry[6:]) / "Img").as_posix()
            for entry in merge_frame["tot_path_drop"]
        ]
        merge_frame = merge_frame.rename(
            columns={
                "Plate number": "plate_nr",
                "Date Imaged": "imaging_day",
            }
        )
    elif len(csv_frame) > 0:
        print(csv_frame.columns)
        csv_frame = csv_frame.rename(
            columns={"Date Imaged": "imaging_day", "place_id_csv": "plate_id"}
        )
        merge_frame = csv_frame
    else:
        raise Exception("Could not find enough data!")
    return merge_frame


def analysis_run(
    input_frame,
    analysis_folder,
    videos_folder,
    dropbox_address,
    logging=True,
    kymo_normalize=False,
    kymo_section_width=2.1,
    close_size=None,
    edge_len_min=70,
    thresh_adjust=0,
    frangi_range=None,
    save_edge_extraction_plot=True,
    make_video=True,
    create_snapshot=True,
    create_edge_video=True,
    photobleach_adjust=False,
    speed_ext_window_number=15,
    speed_ext_window_start=3,
    speed_ext_c_thresh=0.95,
    speed_ext_c_falloff=0.005,
    speed_ext_blur_size=5,
    speed_ext_blur=True,
    speed_ext_max_thresh=80,
    dropbox_upload=True,
    fourier_radius=30,
    fourier_prominence=1,
):
    all_edge_objs = []
    for index, row in input_frame.iterrows():
        ### Below code starts the whole address management. ###
        drop_targ = Path(f"/{row['tot_path_drop']}").relative_to(dropbox_address).parent
        db_address = f"{dropbox_address}Analysis/{drop_targ.as_posix()}"
        row["analysis_folder"] = str(Path(f"{analysis_folder}{row['folder'][:-4]}"))
        row["videos_folder"] = str(Path(f"{videos_folder}{row['folder']}"))
        video_analysis = KymoVideoAnalysis(
            input_frame=row,
            logging=logging,
            show_seg=False,
            filter_step=edge_len_min,
            close_size=close_size,
            thresh_adjust=thresh_adjust,
            frangi_range=frangi_range,
        )
        img_seq = np.arange(len(video_analysis.selection_file))
        edge_objs = video_analysis.edge_objects
        all_edge_objs.append(edge_objs)

        ### Use code below to adjust the width of the kymograph area. ###
        target_length = int(kymo_section_width * video_analysis.magnification)

        ### Plot and save the extracted edges to the analysis folder ###
        video_analysis.plot_extraction_img(
            target_length=target_length,
            save_img=save_edge_extraction_plot,
            logging=True,
        )

        ### Save a compressed video of the raw TIFFs ###
        if make_video:
            video_analysis.makeVideo()

        for edge in edge_objs:
            ### Create video snapshot and edge video ###
            if create_snapshot:
                edge.view_edge(img_frame=0, save_im=True, target_length=target_length)
            if create_edge_video:
                edge.view_edge(
                    img_frame=img_seq,
                    save_im=True,
                    quality=6,
                    target_length=target_length,
                )

            ### Create kymograph of edge, do fourier filtering, extract speeds, extract transport ###
            kymos = edge.extract_multi_kymo(
                1,
                target_length=target_length,
                kymo_adj=False,
                kymo_normalize=kymo_normalize,
            )
            edge.fourier_kymo(return_self=False)
            edge.extract_speeds(
                w_size=speed_ext_window_number,
                w_start=speed_ext_window_start,
                C_thresh=speed_ext_c_thresh,
                C_thresh_falloff=speed_ext_c_falloff,
                blur_size=speed_ext_blur_size,
                preblur=speed_ext_blur,
                speed_thresh=speed_ext_max_thresh,
            )
            edge.extract_transport()
            edge.fourier_analysis(
                R_thresh=fourier_radius, prominence=fourier_prominence
            )

        plot_summary(edge_objs)
        save_raw_data(edge_objs, row["analysis_folder"])
        if dropbox_upload:
            upload_folder(row["analysis_folder"], db_address)
    return all_edge_objs


class HighmagDataset(object):
    def __init__(
        self, dataframe: pd.DataFrame, analysis_folder: str, videos_folder: str
    ):
        self.analysis_folder = analysis_folder
        self.videos_folder = videos_folder
        self.video_frame = dataframe
        self.video_objs = np.array(
            [
                VideoDataset(row, analysis_folder, videos_folder)
                for index, row in self.video_frame.iterrows()
            ]
        )
        self.edge_objs = np.concatenate([video.edge_objs for video in self.video_objs])
        self.edges_frame = pd.concat(
            [edg_obj.mean_data for edg_obj in self.edge_objs], axis=1
        ).T.reset_index(drop=True)

    def filter_edges(self, column, compare, constant):
        filter_self = copy.deepcopy(self)
        if compare == ">=":
            filter_self.edges_frame = filter_self.edges_frame[
                filter_self.edges_frame[column].ge(constant)
            ]
        elif compare == "==":
            filter_self.edges_frame = filter_self.edges_frame[
                filter_self.edges_frame[column] == constant
            ]
        elif compare == "<=":
            filter_self.edges_frame = filter_self.edges_frame[
                filter_self.edges_frame[column].le(constant)
            ]
        else:
            print("Comparison symbol not recognised. Please use >=, ==, or <=.")
            raise "Comparison symbol not recognised. Please use >, ==, or <."

        filter_self.edge_objs = filter_self.edge_objs[
            filter_self.edges_frame.index.to_numpy()
        ]
        is_video = filter_self.video_frame["unique_id"].isin(
            filter_self.edges_frame["unique_id"]
        )
        filter_self.video_objs = filter_self.video_objs[is_video[is_video].index.values]
        filter_self.video_frame = filter_self.video_frame[is_video].reset_index(
            drop=True
        )
        filter_self.edges_frame = filter_self.edges_frame.reset_index(drop=True)
        return filter_self

    def context_4x(self, pd_row, FLUO=True):
        space_res = 2 * 1.725 / pd_row["magnification"] * pd_row["binning"]
        pic_4x_res = (
            np.array([[4088, 3000], [2044, 1500]][pd_row["binning"] == 2])
            * space_res
            / 2
        )
        frame_4x_filt = self.filter_edges("plate_id", "==", pd_row["plate_id"])
        frame_4x_filt = frame_4x_filt.filter_edges("magnification", "==", 50.0)
        #         frame_4x_filt = frame_4x_filt.filter_edges('mode', '==', ['BF', 'F'][FLUO])
        frame_4x_filt = frame_4x_filt.filter_edges(
            "xpos", ">=", pd_row["xpos"] - pic_4x_res[0]
        )
        frame_4x_filt = frame_4x_filt.filter_edges(
            "xpos", "<=", pd_row["xpos"] + pic_4x_res[0]
        )
        frame_4x_filt = frame_4x_filt.filter_edges(
            "ypos", ">=", pd_row["ypos"] - pic_4x_res[1]
        )
        frame_4x_filt = frame_4x_filt.filter_edges(
            "ypos", "<=", pd_row["ypos"] + pic_4x_res[1]
        )
        frame_4x_filt.video_frame = pd.concat(
            [pd_row.to_frame().T, frame_4x_filt.video_frame]
        )

        return frame_4x_filt

    def plot_4x_locs(self, analysis_folder, FLUO=True, x_adj=-500, y_adj=400):
        fig, ax = plt.subplots()

        space_res = (
            2
            * 1.725
            / self.video_frame.iloc[0]["magnification"]
            * self.video_frame.iloc[0]["binning"]
        )
        pic_4x_res = (
            np.array(
                [[4088, 3000], [2044, 1500]][self.video_frame.iloc[0]["binning"] == 2]
            )
            * space_res
            / 2
        )

        vidcap = cv2.VideoCapture(
            f"{analysis_folder}{self.video_frame.iloc[0]['folder'][:-4]}{self.video_frame.iloc[0]['unique_id']}_video.mp4"
        )
        success, image = vidcap.read()
        if success:
            ax.imshow(
                image,
                extent=[
                    (self.video_frame.iloc[0]["xpos"] - pic_4x_res[0]) + x_adj,
                    (self.video_frame.iloc[0]["xpos"] + pic_4x_res[0]) + x_adj,
                    (-self.video_frame.iloc[0]["ypos"] - pic_4x_res[1]) + y_adj,
                    (-self.video_frame.iloc[0]["ypos"] + pic_4x_res[1]) + y_adj,
                ],
            )

        for index, row in self.edges_frame.iterrows():
            space_res = 2 * 1.725 / row["magnification"] * row["binning"]
            ax.annotate(
                row["unique_id"].split("_")[-1], (row["xpos"], -row["ypos"]), c="black"
            )

            arrow_start = np.array(
                [
                    (
                        row["xpos"] + space_res * row["edge_ypos_2"],
                        (row["ypos"] + space_res * row["edge_xpos_2"]) * -1,
                    )
                ]
            )[0]
            arrow_end = (
                np.array(
                    [
                        (
                            row["xpos"] + space_res * row["edge_ypos_1"],
                            (row["ypos"] + space_res * row["edge_xpos_1"]) * -1,
                        )
                    ]
                )[0]
                - arrow_start
            )
            ax.quiver(
                arrow_start[0],
                arrow_start[1],
                arrow_end[0],
                arrow_end[1],
                angles="xy",
                color=["tab:green", "gray"][row["magnification"] == 50],
                scale_units="xy",
                scale=1,
            )

        ax.scatter(
            self.video_frame.iloc[1:]["xpos"], -self.video_frame.iloc[1:]["ypos"]
        )
        ax.axis("equal")
        ax.set_title(f"4x overview of {self.video_frame.iloc[0]['unique_id']}")
        fig.tight_layout()
        return None

    def plot_plate_locs(
        self, analysis_folder, spd_thresh=4, annotate_adj=0.1, modes=["scatter"]
    ):
        # There is no way to align to stitched plates for now, so we'll just output a clean graph
        fig, ax = plt.subplots(figsize=(16, 9))
        videos_4x = self.video_frame[self.video_frame["magnification"] == 4.0]
        videos_50x = self.video_frame[self.video_frame["magnification"] == 50.0]
        if "scatter" in modes:
            ax.scatter(
                videos_50x["xpos"], -videos_50x["ypos"], c="black", label="50x mag", s=2
            )
            ax.scatter(
                videos_4x["xpos"], -videos_4x["ypos"], c="tab:green", label="4x mag"
            )
        if "speeds_mean" in modes:
            for video in self.video_objs:
                video.plot_speed_arrows(
                    ax,
                    [video.dataset["xpos"], -video.dataset["ypos"]],
                    plot_text=False,
                    plot_mean=True,
                    significance_thresh=spd_thresh,
                )

        if "flux_mean" in modes:
            for video in self.video_objs:
                video.plot_speed_arrows(
                    ax,
                    [video.dataset["xpos"], -video.dataset["ypos"]],
                    plot_text=False,
                    plot_flux=True,
                    significance_thresh=spd_thresh,
                )

        if "speeds_both" in modes:
            for video in self.video_objs:
                video.plot_speed_arrows(
                    ax,
                    [video.dataset["xpos"], -video.dataset["ypos"]],
                    plot_text=False,
                    plot_both=True,
                    significance_thresh=spd_thresh,
                )

        if "vid_labels" in modes:
            label_frame = self.video_frame.copy()
            label_frame["coords"] = [
                f"{row['xpos']}, {row['ypos']}" for index, row in label_frame.iterrows()
            ]
            for label in label_frame["coords"].unique():
                coordx = label_frame[label_frame["coords"] == label]["xpos"].iloc[0]
                coordy = label_frame[label_frame["coords"] == label]["ypos"].iloc[0]
                vid_list = [
                    int(row["video_int"])
                    for index, row in label_frame[
                        label_frame["coords"] == label
                    ].iterrows()
                ]
                ax.annotate(
                    vid_list,
                    (coordx + annotate_adj, -coordy + annotate_adj),
                    c="black",
                    fontsize=2,
                )
        if "speeds_text" in modes:
            for video in self.video_objs:
                video.plot_speed_arrows(
                    ax, [video.dataset["xpos"], -video.dataset["ypos"]]
                )

        ax.axis("equal")
        if np.mean(self.video_frame["ypos"]) > 100:
            ax.set_xlim((-5000, 70000))
            ax.set_ylim((-50000, -10000))
        else:
            x_mean = videos_50x["xpos"].mean()
            y_mean = -videos_50x["ypos"].mean()
            ax.set_xlim((x_mean - 45, x_mean + 45))
            ax.set_ylim((y_mean - 25, y_mean + 25))
        ax.set_title(f"Plate {self.video_frame['plate_id'][0]}")
        ax.legend()
        fig.tight_layout()

        if not os.path.exists(f"{analysis_folder}plot_outs/{'_'.join(modes)}/"):
            os.makedirs(f"{analysis_folder}plot_outs/{'_'.join(modes)}/")

        fig.savefig(
            f"{analysis_folder}plot_outs/{'_'.join(modes)}/{self.video_frame['plate_id'][0]}_{'_'.join(modes)}.png",
            transparent=True,
        )

        return None

    def filter_videos(self, column, compare, constant):
        return None

    def bin_values(self, column, bins, new_column_name="edge_bin_values"):
        bin_series = pd.cut(self.edges_frame[column], bins, labels=False)
        self.edges_frame[new_column_name] = bin_series
        return bin_series

    def plot_histo(self, bins, column, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.edges_frame[column], bins=bins, edgecolor="black")
        ax.set_xlabel(column)
        ax.set_ylabel("counts")
        if ax is None:
            fig.tight_layout()
        return ax

    def plot_violins(
        self,
        column,
        bins=None,
        bin_separator="edge_bin_values",
        ax=None,
        fig=None,
        c="tab:blue",
        labels=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        if bins is not None:
            violin_data = [
                self.edges_frame[column][self.edges_frame[bin_separator] == i].astype(
                    float
                )
                for i in range(len(bins))
            ]
        else:
            violin_data = [
                self.edges_frame[column][self.edges_frame[bin_separator] == i].astype(
                    float
                )
                for i in self.edges_frame[bin_separator].unique()
            ]
            x_labels = self.edges_frame[bin_separator].unique()
            bins = range(self.edges_frame[bin_separator].nunique())
            ax.set_xticks(bins, labels=x_labels, rotation=45)
        violin_data_d = []
        for data in violin_data:
            if data.empty:
                violin_data_d.append([np.nan, np.nan])
            else:
                violin_data_d.append(data)
        violin_parts = ax.violinplot(
            dataset=violin_data_d,
            positions=bins,
            widths=(bins[1] - bins[0]) * 0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        violin_means = [np.nanmean(data, axis=0) for data in violin_data_d]
        ax.scatter(bins, violin_means, s=4, c="black")
        for vp in violin_parts["bodies"]:
            vp.set_edgecolor("black")
            vp.set_alpha(1)
            vp.set_facecolor(c)
        if labels is not None:
            labels.append((mpatches.Patch(color=c), column))

        return fig, ax

    def return_video_frame(self):
        return self.video_frame

    def return_edge_frame(self):
        return self.edges_frame

    def return_edge_objs(self):
        return self.edge_objs

    def return_vid_objs(self):
        return self.video_objs


class VideoDataset(object):
    def __init__(self, series, analysis_folder, videos_folder):
        self.dataset = series
        self.img_dim = [[2048, 1500], [4096, 3000]][self.dataset["binning"] == 1]
        self.space_res = (
            2 * 1.725 / self.dataset["magnification"] * self.dataset["binning"]
        )
        self.time_res = 1 / self.dataset["fps"]
        self.vid_analysis_folder = Path(f"{analysis_folder}{self.dataset['folder']}")
        edge_adr = Path(f"{analysis_folder}{self.dataset['folder']}") / "edges_data.csv"
        self.edge_adr = edge_adr
        self.imshow_extent = [
            0,
            self.space_res * self.img_dim[0],
            self.space_res * self.img_dim[1],
            0,
        ]
        if edge_adr.exists():
            self.edges_frame = pd.read_csv(edge_adr)
            self.edge_objs = [
                EdgeDataset(
                    pd.concat([row, self.dataset]), analysis_folder, videos_folder
                )
                for index, row in self.edges_frame.iterrows()
            ]
            self.dataset["nr_of_edges"] = len(self.edges_frame)
        else:
            print(
                f"Couldn't find the edges data file at {edge_adr}. Check analysis for {self.dataset['unique_id']}"
            )
            self.dataset["nr_of_edges"] = 0
            self.edge_objs = []

    def show_summary(self):
        summ_path = Path(self.dataset["analysis_folder"]) / "Detected edges.png"
        if summ_path.exists():
            extraction_img = imageio.imread(summ_path)
            fig, ax = plt.subplots()
            ax.imshow(extraction_img)
            ax.set_axis_off()
            ax.set_title(f"{self.dataset['unique_id']}")
            fig.tight_layout()

    def show_segmentation(self):
        seg_path = Path(self.dataset["analysis_folder"]) / "Video segmentation.png"
        if seg_path.exists():
            extraction_img = imageio.imread(seg_path)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
            ax.imshow(extraction_img)
            fig.suptitle(self.dataset["unique_id"])
            ax.set_axis_off()
            fig.tight_layout()

    def scatter_speeds_video(self):
        fig, ax = plt.subplots()
        # ax.grid(True)
        ax.axhline(c="black", linestyle="--", alpha=0.5)
        ax.scatter(
            self.edges_frame.index,
            self.edges_frame["speed_mean"],
            c="black",
            label="effMean",
        )
        ax.errorbar(
            self.edges_frame.index,
            self.edges_frame["speed_right"],
            self.edges_frame["speed_right_std"],
            c="tab:orange",
            label="to tip",
            marker="o",
            linestyle="none",
            capsize=10,
        )
        ax.errorbar(
            self.edges_frame.index,
            self.edges_frame["speed_left"],
            self.edges_frame["speed_left_std"],
            c="tab:blue",
            label="to root",
            marker="o",
            linestyle="none",
            capsize=10,
        )
        ax.legend()
        ax.set_xticks(self.edges_frame.index)
        ax.set_xticklabels(self.edges_frame["edge_name"], rotation=45)
        ax.set_title(f"Edge speed overview for {self.dataset['unique_id']}")
        ax.set_ylabel("Velocity $(\mu m /s)$")
        fig.tight_layout()

    def get_first_frame(self):
        vid_address = (
            self.vid_analysis_folder / f"{self.dataset['unique_id']}_video.mp4"
        )
        print(vid_address)
        vidcap = cv2.VideoCapture(str(vid_address))
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, self.img_dim)
            return image  # save frame as JPEG file

    def plot_speed_arrows(
        self,
        ax=None,
        vid_pos=None,
        video_txt_size=10,
        plot_text=True,
        plot_flux=False,
        plot_both=False,
        plot_mean=False,
        significance_thresh=0.2,
        save_im: str = "",
    ):

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title(f"{self.dataset['unique_id']}")
            plate_offset = [0, 0]
            space_factor = self.space_res
            arr_width = 3
            txt_size = video_txt_size
            line_size = 1
        else:
            plate_offset = np.add(vid_pos, [-0.30, -0.42])
            space_factor = 0.0005
            arr_width = 0.03
            txt_size = 1.75
            line_size = 0.2

        for edge_obj in self.edge_objs:
            speed_l, speed_r = edge_obj.return_speed_medians()
            speed_m = edge_obj.time_data["speed_weight_mean"].mean()
            speed_f = edge_obj.time_data["flux_mean"].mean()
            edge_coords = [
                [edge_obj.mean_data["edge_ypos_1"], edge_obj.mean_data["edge_xpos_1"]],
                [edge_obj.mean_data["edge_ypos_2"], edge_obj.mean_data["edge_xpos_2"]],
            ]

            edge_mids = (
                np.mean(
                    [
                        [edge_coords[0][0], edge_coords[0][1]],
                        [edge_coords[1][0], edge_coords[1][1]],
                    ],
                    axis=0,
                )
                * space_factor
            )

            edge_tangent = [
                edge_coords[1][0] - edge_coords[0][0],
                edge_coords[1][1] - edge_coords[0][1],
            ] / np.linalg.norm(
                [
                    edge_coords[1][0] - edge_coords[0][0],
                    edge_coords[1][1] - edge_coords[0][1],
                ]
            )
            if vid_pos is not None:
                edge_tangent[0] *= -1
                edge_tangent *= -1
                edge_rot = np.arctan2(edge_tangent[1], edge_tangent[0]) * 180 / np.pi
                if edge_rot < -90:
                    edge_rot += 180
            else:
                edge_rot = (
                    np.arctan2(edge_tangent[0], edge_tangent[1]) * 180 / np.pi + 90
                )
                if edge_rot < -90:
                    edge_rot += 180
                if edge_rot > 90:
                    edge_rot -= 180
            edge_dir = edge_tangent * 70 * space_factor
            edge_starts = edge_mids - 0.5 * edge_dir
            edge_ends = edge_starts + edge_dir

            edge_normal = np.array([[0, -1], [1, 0]]).dot(np.array(edge_tangent))
            edge_offset = edge_normal * 80 * space_factor

            if plot_both:
                if abs(speed_r) > significance_thresh:
                    ax.arrow(
                        edge_starts[0] + edge_offset[0] + plate_offset[0],
                        edge_starts[1] + edge_offset[1] + plate_offset[1],
                        edge_dir[0] * speed_r,
                        edge_dir[1] * speed_r,
                        width=arr_width,
                        linewidth=line_size,
                        facecolor="tab:orange",
                        edgecolor="black",
                    )
                else:
                    ax.scatter(
                        [edge_starts[0] + edge_offset[0] + plate_offset[0]],
                        [edge_starts[1] + edge_offset[1] + plate_offset[1]],
                        c="tab:orange",
                        s=0.3,
                    )
                if abs(speed_l) > significance_thresh:
                    ax.arrow(
                        edge_ends[0] - edge_offset[0] + plate_offset[0],
                        edge_ends[1] - edge_offset[1] + plate_offset[1],
                        edge_dir[0] * speed_l,
                        edge_dir[1] * speed_l,
                        width=arr_width,
                        linewidth=line_size,
                        facecolor="tab:blue",
                        edgecolor="black",
                    )
                else:
                    ax.scatter(
                        [edge_starts[0] + edge_offset[0] + plate_offset[0]],
                        [edge_starts[1] + edge_offset[1] + plate_offset[1]],
                        c="tab:blue",
                        s=0.3,
                    )
            if plot_mean:
                if abs(speed_m) > significance_thresh:
                    ax.arrow(
                        edge_ends[0] + plate_offset[0],
                        edge_ends[1] + plate_offset[1],
                        edge_dir[0] * speed_m,
                        edge_dir[1] * speed_m,
                        width=arr_width,
                        linewidth=line_size,
                        facecolor="black",
                        edgecolor="black",
                    )
                else:
                    ax.scatter(
                        [edge_ends[0] + plate_offset[0]],
                        [edge_ends[1] + plate_offset[1]],
                        color="black",
                        s=0.5,
                    )
            if plot_flux:
                if abs(speed_f / 100) > significance_thresh:
                    ax.arrow(
                        edge_ends[0] + plate_offset[0],
                        edge_ends[1] + plate_offset[1],
                        # edge_dir[0] * speed_f / 100,
                        # edge_dir[1] * speed_f / 100,
                        edge_dir[0] * np.sign(speed_f),
                        edge_dir[1] * np.sign(speed_f),
                        width=arr_width,
                        linewidth=line_size,
                        facecolor="black",
                        edgecolor="black",
                    )
                else:
                    ax.scatter(
                        [edge_ends[0] + plate_offset[0]],
                        [edge_ends[1] + plate_offset[1]],
                        color="black",
                        s=0.5,
                    )
            if plot_text:
                if plot_both:
                    ax.annotate(
                        f"{speed_r:.2}",
                        edge_starts + edge_offset,
                        xytext=edge_starts
                        + edge_offset
                        + plate_offset
                        - 10 * edge_normal * space_factor
                        + 20 * edge_tangent * space_factor,
                        rotation=edge_rot,
                        color="white",
                        fontsize=txt_size,
                        path_effects=[
                            pe.withStroke(
                                linewidth=[0.2, 2][vid_pos is None], foreground="black"
                            )
                        ],
                    )
                    ax.annotate(
                        f"{-speed_l:.2}",
                        edge_ends - edge_offset,
                        xytext=edge_ends
                        - edge_offset
                        + plate_offset
                        - 100 * edge_normal * space_factor
                        - 80 * edge_tangent * space_factor,
                        rotation=edge_rot,
                        color="white",
                        fontsize=txt_size,
                        path_effects=[
                            pe.withStroke(
                                linewidth=[0.2, 2][vid_pos is None], foreground="black"
                            )
                        ],
                    )
                if plot_mean:
                    ax.annotate(
                        f"{speed_m:.2}",
                        edge_starts + edge_offset,
                        xytext=edge_starts
                        + edge_offset
                        + plate_offset
                        - 10 * edge_normal * space_factor
                        + 20 * edge_tangent * space_factor,
                        rotation=edge_rot,
                        color="white",
                        fontsize=txt_size,
                        path_effects=[
                            pe.withStroke(
                                linewidth=[0.2, 2][vid_pos is None], foreground="black"
                            )
                        ],
                    )
                if plot_flux:
                    ax.annotate(
                        f"{speed_f / 100:.2}",
                        edge_starts + edge_offset,
                        xytext=edge_starts
                        + edge_offset
                        + plate_offset
                        - 10 * edge_normal * space_factor
                        + 20 * edge_tangent * space_factor,
                        rotation=edge_rot,
                        color="white",
                        fontsize=txt_size,
                        path_effects=[
                            pe.withStroke(
                                linewidth=[0.2, 2][vid_pos is None], foreground="black"
                            )
                        ],
                    )
        if vid_pos is None:
            vid_frame = self.get_first_frame()
            ax.imshow(vid_frame, extent=self.imshow_extent)
            fig.tight_layout()
        if save_im:
            # print(Path(f"{edge_obj.mean_data['analysis_folder']}{edge_obj.mean_data['folder']}")/ "speed_arrows.png")
            fig.savefig(
                Path(
                    f"{edge_obj.mean_data['analysis_folder']}{edge_obj.mean_data['folder']}"
                )
                / f"{save_im}.png"
            )


class EdgeDataset(object):
    def __init__(self, dataframe, analysis_folder, videos_folder):
        self.mean_data = dataframe
        self.mean_data["analysis_folder"] = analysis_folder

        self.edge_name = self.mean_data["edge_name"]
        edge_dat_adr = (
            Path(f"{analysis_folder}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_data.csv"
        )
        self.edge_dat_adr = edge_dat_adr
        self.time_data = pd.read_csv(edge_dat_adr)
        edge_dat_adr2 = (
            Path(f"{analysis_folder}{self.mean_data['folder']}") / f"edges_data.csv"
        )
        # self.edges_dat_adr = edge_dat_adr2
        edges_data = pd.read_csv(edge_dat_adr2)
        self.edge_infos = edges_data[edges_data["edge_name"] == self.edge_name].iloc[0]
        self.xpos1 = edges_data["edge_xpos_1"]
        self.xpos2 = edges_data["edge_xpos_2"]
        self.ypos1 = edges_data["edge_ypos_1"]
        self.ypos2 = edges_data["edge_ypos_2"]
        self.space_res = (
            2 * 1.725 / self.mean_data["magnification"] * self.mean_data["binning"]
        )
        self.time_res = 1 / self.mean_data["fps"]
        self.img_dim = imread(
            Path(f"{analysis_folder}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_speeds_flux_array.tiff"
        )[0].shape
        self.imshow_extent = [
            0,
            self.space_res * self.img_dim[1],
            self.time_res * self.img_dim[0],
            0,
        ]

    def show_summary(self):
        summ_path = (
            Path(f"{self.mean_data['analysis_folder']}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_summary.png"
        )
        summ_img = imageio.imread(summ_path)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.imshow(summ_img)
        ax.set_title(self.mean_data["unique_id"])
        ax.set_axis_off()

    def show_kymos(self):
        summ_path = (
            Path(f"{self.mean_data['analysis_folder']}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_kymos.png"
        )
        summ_img = imageio.imread(summ_path)
        fig = plt.figure()
        fig.suptitle(f"{self.mean_data['unique_id']}")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.imshow(summ_img)
        ax.set_title(self.mean_data["unique_id"])
        ax.set_axis_off()
        fig.tight_layout()

    def plot_flux(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_data["times"], self.time_data["flux_mean"])
        fig.tight_layout()

    def return_speed_medians(self, spd_tiff_lowbound=0.5):
        spd_array_path = (
            Path(f"{self.mean_data['analysis_folder']}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_speeds_flux_array.tiff"
        )
        spd_tiff = imread(spd_array_path)
        spd_tiff[0] = np.where(spd_tiff[0] < -spd_tiff_lowbound, spd_tiff[0], np.nan)
        spd_tiff[1] = np.where(spd_tiff[1] > spd_tiff_lowbound, spd_tiff[1], np.nan)

        spd_max_l = np.nanmedian(spd_tiff[0].flatten())
        spd_max_r = np.nanmedian(spd_tiff[1].flatten())
        return spd_max_l, spd_max_r

    def direction_filter(self):
        kymo_tiff = imread(
            Path(f"{self.mean_data['analysis_folder']}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_kymos_array.tiff"
        )[0]
        flux_tiff = imread(
            Path(f"{self.mean_data['analysis_folder']}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_speeds_flux_array.tiff"
        )[2]
        kymo_filts = [[], []]
        fig, ax = plt.subplots(4, figsize=(5, 10))
        fig.suptitle(self.mean_data["unique_id"])

        for j, kymo in enumerate([kymo_tiff, np.flip(kymo_tiff, axis=1)]):
            img = kymo.copy()
            for i in range(1):
                A = img[:, :]
                B = np.flip(A, axis=0)
                C = np.flip(A, axis=1)
                D = np.flip(B, axis=1)
                tiles = [[D, B, D], [C, A, C], [D, B, D]]
                tiles = [cv2.hconcat(imgs) for imgs in tiles]
                fourrier = cv2.vconcat(tiles)
                img = fourrier
            tiling_for_fourrier = img
            shape_v, shape_h = tiling_for_fourrier.shape

            dark_image_grey_fourier = np.fft.fftshift(
                np.fft.fft2(tiling_for_fourrier, norm="ortho")
            )
            coordinates_middle = np.array(dark_image_grey_fourier.shape) // 2

            # Within the fourier transform, we have four quadrants, where we remove two of them. This in addition to the DC term
            LT_quadrant = np.s_[: coordinates_middle[0] + 0, : coordinates_middle[1]]
            RB_quadrant = np.s_[
                coordinates_middle[0] + 1 :, coordinates_middle[1] + 1 :
            ]

            filtered_fourrier = dark_image_grey_fourier.copy()

            # Calculation of the static part of the fourier spectrum. Looks like a horizontal band with vertically flared ends.
            v_axis = np.arange(0, shape_v) - (shape_v // 2)
            h_axis = np.arange(0, shape_h) - (shape_h // 2)
            v_array = np.array([v_axis for i in range(shape_h)]).transpose()
            h_array = np.array([h_axis for i in range(shape_v)])
            stat_array = 1 * (abs(v_array) <= (1 + abs(h_array) / 1000))

            filtered_fourrier[LT_quadrant] = 0
            filtered_fourrier[RB_quadrant] = 0
            filtered_fourrier *= 1 - stat_array

            stat_filt = np.fft.ifft2(
                np.fft.ifftshift(dark_image_grey_fourier * stat_array), norm="ortho"
            )
            filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fourrier), norm="ortho")

            shape_v, shape_h = shape_v // 3, shape_h // 3
            middle_slice = np.s_[shape_v : 2 * shape_v, shape_h : 2 * shape_h]
            middle = filtered[middle_slice]

            kymo_filts[j] = np.array(middle.real)
        ax[0].imshow(
            kymo_filts[0],
            aspect=4,
            cmap="coolwarm",
            vmin=np.array(kymo_filts).flatten().min(),
            vmax=np.array(kymo_filts).flatten().max(),
        )
        ax[1].imshow(
            kymo_filts[1],
            aspect=4,
            cmap="coolwarm",
            vmin=np.array(kymo_filts).flatten().min(),
            vmax=np.array(kymo_filts).flatten().max(),
        )
        # ax[2].plot(np.sum(abs(kymo_filts[0]), axis=1))
        # ax[2].plot(np.sum(abs(kymo_filts[1]), axis=1))
        ax[2].plot(np.mean(kymo_filts[0], axis=1))
        ax[2].plot(np.mean(kymo_filts[1], axis=1))
        ax[3].plot(np.mean(kymo_filts[0], axis=1) - np.mean(kymo_filts[1], axis=1))
        ax[3].plot(np.nanmean(flux_tiff, axis=1) / 5)

    def plot_speed_histo(
        self,
        spd_extent=15,
        spd_tiff_lowbound=0.5,
        spd_cutoff=0.2,
        bin_res=100,
        plot_fig=True,
    ):
        spd_array_path = (
            Path(f"{self.mean_data['analysis_folder']}{self.mean_data['folder']}")
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_speeds_flux_array.tiff"
        )
        spd_tiff = imread(spd_array_path)
        kymo_tiff = imread(
            Path(self.mean_data["analysis_folder"])
            / f"edge {self.mean_data['edge_name']}"
            / f"{self.mean_data['edge_name']}_kymos_array.tiff"
        )[1]

        spd_tiff[0] = np.where(spd_tiff[0] < -spd_tiff_lowbound, spd_tiff[0], np.nan)
        spd_tiff[1] = np.where(spd_tiff[1] > spd_tiff_lowbound, spd_tiff[1], np.nan)

        spd_bins = np.linspace(-spd_extent, spd_extent, bin_res)
        spd_cut = spd_cutoff / spd_extent / 2
        spd_cut_l = bin_res // 2 - int(bin_res * spd_cut)
        spd_cut_r = bin_res // 2 + int(bin_res * spd_cut)
        spd_hist_l = np.histogram(spd_tiff[0], spd_bins, density=True)
        spd_hist_r = np.histogram(spd_tiff[1], spd_bins, density=True)
        spd_hist = [spd_hist_l, spd_hist_r]
        spd_max_l = np.nanmedian(spd_tiff[0].flatten())
        spd_max_r = np.nanmedian(spd_tiff[1].flatten())

        if plot_fig:
            fig, ax = plt.subplot_mosaic(
                [
                    ["kymo", "spd_left", "spd_right"],
                    ["spd_histo", "spd_histo", "spd_histo"],
                ]
            )
            fig.suptitle(f"{self.mean_data['unique_id']} {self.mean_data['edge_name']}")
            ax["kymo"].imshow(kymo_tiff, extent=self.imshow_extent, aspect="auto")
            ax["kymo"].set_title("Kymograph")
            ax["kymo"].set_xlabel("x $(\mu m)$")
            ax["kymo"].set_ylabel("t $(s)$")
            ax["spd_left"].imshow(
                spd_tiff[0],
                cmap="coolwarm",
                vmin=-20,
                vmax=20,
                extent=self.imshow_extent,
                aspect="auto",
            )
            ax["spd_left"].set_title("Speeds_left")
            ax["spd_left"].set_xlabel("x $(\mu m)$")
            ax["spd_left"].set_ylabel("t $(s)$")
            ax["spd_right"].imshow(
                spd_tiff[1],
                cmap="coolwarm",
                vmin=-20,
                vmax=20,
                extent=self.imshow_extent,
                aspect="auto",
            )
            ax["spd_right"].set_title("Speeds_right")
            ax["spd_right"].set_xlabel("x $(\mu m)$")
            ax["spd_right"].set_ylabel("t $(s)$")
            ax["spd_histo"].plot(
                spd_hist[0][1][:-1],
                spd_hist[0][0],
                label=f"to_root, max={spd_max_l:.3}",
            )
            ax["spd_histo"].plot(
                spd_hist[1][1][:-1], spd_hist[1][0], label=f"to_tip, max={spd_max_r:.3}"
            )
            ax["spd_histo"].set_title("Histogram of speeds")
            ax["spd_histo"].set_xlabel("Velocity $(\mu m /s)$")
            ax["spd_histo"].set_ylabel("Frequency")
            ax["spd_histo"].grid(True)
            ax["spd_histo"].legend()
            fig.tight_layout()

        spd_hist_l_f = spd_hist_l[0][:]

        return spd_max_l, spd_max_r
