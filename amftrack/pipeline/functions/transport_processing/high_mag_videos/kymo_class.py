from pathlib import Path
import imageio.v2 as imageio
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import re
import matplotlib as mpl
from scipy.signal import find_peaks
import tifffile

from amftrack.pipeline.functions.transport_processing.high_mag_videos.high_mag_videos_fun import *

mpl.rcParams["figure.dpi"] = 200


# TODO: Go through these functions to streamline them, and remove deprecated values. Outputting plots is also no longer necessary, as all outputs are created when called for.


class KymoVideoAnalysis(object):
    """
    Master class for video processing. Will use nearby video parameters (from csv's, xslx's or the input frame)
    to segment video, extract edges, and set up edge analysis.
    Use address as input if you want to rely on nearby data sheets,
    use input frame as input to specify video parameters with a pandas frame.
    """

    def __init__(
        self,
        imgs_address=None,
        fps=None,
        binning=2,
        magnification=50,
        vid_type="BRIGHT",
        im_range=(0, -1),
        thresh=5e-07,
        filter_step=70,
        thresh_adjust=0,
        frangi_range=None,
        seg_thresh=20,
        logging=False,
        show_seg=False,
        input_frame=None,
        close_size=None,
        samepos_frame=None,
    ):
        """
        Master class for video processing. Will use nearby video parameters (from csv's, xslx's or the input frame)
        to segment video, extract edges, and set up edge analysis.
        :param imgs_address:    Address for the video files
        :param fps:             Frames per second of the video if unspecified
        :param binning:         Binning (1 or 2) of the video if unspecified
        :param magnification:   Magnification (4, 50, 60) of the video if unspecified
        :param vid_type:        Imaging mode (F or BF) of the video if unspecified
        :param im_range:        Largely obsolete function that says which frames to take
        :param thresh:          Threshold value for something
        :param filter_step:     Length threshold for extracted edges. Measured in pixels
        :param seg_thresh:      Obsolete value, used to specify what percentage of a frame
                                had to be segmented for proper hypha segmentation.
        :param logging:         Boolean on how verbose the class needs to be about its progress
        :param show_seg:        Whether to save the image segmentation
        :param input_frame:     If None, the program will try to find nearby .csv's or .xlsx's to extract video parameters,
                                otherwise it will pull the information from the input_frame.
                                Frames need to be created with specific column names,
                                just use the master notebook for guidance
        """

        # First we're assigning values, and extracting data about the video if there is a .csv (or .xlsx) available.
        # self.imgs_address = os.path.join(imgs_address, 'Img')
        self.im_range = im_range
        self.logging = logging

        if input_frame is None:
            self.imgs_address = Path(imgs_address) / "Img"
            self.kymos_path = f"{imgs_address}Analysis/"
            self.video_nr = int(re.split("/|_", imgs_address[:-1])[-1])
            parent_files = self.imgs_address.parents[1]
            self.binning = binning
            self.magnification = magnification
            self.vid_type = vid_type
            self.fps = fps
            self.im_range = im_range
            self.back_fit = [0, 0, 0]
            self.back_offset = 0

            ### Extracts the video parameters from the nearby csv.
            self.csv_path = next(Path(parent_files).glob("*.csv"), None)
            if self.csv_path is None:
                self.xlsx_path = next(Path(parent_files).glob("*.xlsx"), None)
            #             [a for a in Path(parent_files).glob(f'*.xlsx')]
            else:
                self.xlsx_path = []
            if logging:
                print(f"Using the datasheet at: {self.xlsx_path}")

            if self.csv_path is not None or self.xlsx_path is not None:
                if self.csv_path is not None and self.xlsx_path is not None:
                    print("Found both a csv and xlsx file, will use csv data.")
                if self.csv_path is not None:
                    if logging:
                        print("Found a csv file, using that data")
                    df_comma = pd.read_csv(self.csv_path, nrows=1, sep=",")
                    df_semi = pd.read_csv(self.csv_path, nrows=1, sep=";")
                    if df_comma.shape[1] > df_semi.shape[1]:
                        videos_data = pd.read_csv(self.csv_path, sep=",")
                    else:
                        videos_data = pd.read_csv(self.csv_path, sep=";")

                    if videos_data.isnull().values.any():
                        print(
                            "Found NaNs in the excel files! Blame the experimentalists."
                        )
                        videos_data = videos_data.interpolate(
                            method="pad", limit_direction="forward"
                        )
                    self.video_data = videos_data.loc[
                        videos_data["video"] == self.video_nr
                    ]
                    self.vid_type = ["FLUO", "BRIGHT"][
                        self.video_data["Illumination"].iloc[0] == "BF"
                    ]
                    self.fps = float(self.video_data["fps"].iloc[0])
                    self.binning = (self.video_data["Binned"].iloc[0] * 2) + 1
                    self.magnification = self.video_data["Lens"].iloc[0]
                #                     self.id = self.video_data["video"].iloc[0]
                elif self.xlsx_path is not None:
                    if logging:
                        print("Found an xlsx file, using that data")
                    videos_data = pd.read_excel(self.xlsx_path)
                    print(str(self.imgs_address))
                    excel_addr = f"{imgs_address.split(os.sep)[-3]}_{imgs_address.split(os.sep)[-2]}"
                    self.video_data = videos_data[
                        videos_data["Unnamed: 0"].str.contains(
                            excel_addr, case=False, na=False
                        )
                    ]
                    print(self.video_data.to_string())
                    self.vid_type = ["FLUO", "BRIGHT"][
                        self.video_data.iloc[0, 9] == "BF"
                    ]
                    self.fps = float(self.video_data["FPS"].iloc[0])
                    if "Binned (Y/N)" in self.video_data:
                        self.binning = [1, 2][
                            self.video_data["Binned (Y/N)"].iloc[0] == "Y"
                        ]
                    else:
                        self.binning = 1
                    self.magnification = self.video_data["Magnification"].iloc[0]

                if logging:
                    print(
                        f"Analysing {self.vid_type} video of {self.magnification}X zoom, with {self.fps} fps"
                    )
        else:
            self.fps = float(input_frame["fps"])
            self.magnification = input_frame["magnification"]
            self.binning = input_frame["binning"]
            self.kymos_path = input_frame["analysis_folder"]
            self.imgs_address = Path(input_frame["videos_folder"])
            vid_dic = {"F": "FLUO", "BF": "BRIGHT"}
            self.vid_type = vid_dic[input_frame["mode"]]
            self.video_nr = input_frame["unique_id"]

        self.time_pixel_size = 1 / self.fps
        self.space_pixel_size = (
            2 * 1.725 / (self.magnification) * self.binning
        )  # um.pixel
        self.pos = []
        if not os.path.exists(self.kymos_path):
            os.makedirs(self.kymos_path)
            if self.logging:
                print("Kymos file created, address is at {}".format(self.kymos_path))
        self.images_total_path = [str(adr) for adr in self.imgs_address.glob("*_*.ti*")]
        self.images_total_path.sort()
        self.selection_file = self.images_total_path
        self.selection_file.sort()
        # the line below was commented out in order to get all frames for std seg
        # self.selection_file = self.selection_file[self.im_range[0] : self.im_range[1]]

        if self.logging:
            print("Data input succesful! Starting edge extraction...")

        ### Skeleton creation, we segment the image using either brightfield or fluo segmentation methods.
        if self.vid_type == "BRIGHT":
            print("segmenting")
            (
                self.segmented,
                self.nx_graph_pruned,
                self.pos,
            ) = segment_brightfield_ultimate(self.selection_file[:300])
            print("lenvideo", len(self.selection_file))
        elif self.vid_type == "FLUO":
            if not samepos_frame.empty:
                if isinstance(samepos_frame["videos_folder"], str):
                    bfimage_address = Path(samepos_frame["videos_folder"])
                else:
                    bfimage_address = Path(samepos_frame["videos_folder"].iloc[0])
                #             print(bfimage_address)
                alltiffs = [str(adr) for adr in bfimage_address.glob("*_*.ti*")]
                #             print("the amount of tif(f) images in videos_folder: ", len(self.images_total_path))
                alltiffs.sort()
                # sortedtiffs = alltiffs[self.im_range[0] : self.im_range[1]]
                #             print("this is one tiff: ", onetiff)
                # print(len(self.selection_file))4
                # DOING THE std segmentation

                (
                    self.segmented,
                    self.nx_graph_pruned,
                    self.pos,
                ) = segment_brightfield_ultimate(alltiffs[:300])
                # self.segmented, self.nx_graph_pruned, self.pos = segment_brightfield(
                #     imageio.imread(sortedtiffs[self.im_range[0]]), frangi_range=frangi_range, thresh=thresh,
                #     seg_thresh=seg_thresh,thresh_adjust=thresh_adjust, binning=self.binning, close_size=close_size)
            #             for i, pos in enumerate(input_frame['xpos']):
            #                 if self.pos[0]==pos and self.pos[1]==input_frame['ypos'][i] and self.pos[2]==input_frame['zpos'][i] and input_frame['mode'][i]=='BF':
            #                     if frangi_range is None:
            #                         frangi_range = [np.arange(5, 20, 3), np.arange(20, 160, 20)][self.magnification == 50]
            #                     #print(len(self.selection_file))
            #                     self.segmented, self.nx_graph_pruned, self.pos = segment_brightfield(
            #                         imageio.imread(self.selection_file[i]), frangi_range=frangi_range, thresh=thresh,
            #                         seg_thresh=seg_thresh,thresh_adjust=thresh_adjust, binning=self.binning, close_size=close_size)
            else:
                self.segmented, self.nx_graph_pruned, self.pos = segment_fluo_new(
                    [
                        imageio.imread(addresses)
                        for addresses in self.selection_file[:30]
                    ],
                    threshtype="hist_edge",
                )
        else:
            print(
                "I don't have a valid flow_processing type!!! Using fluo thresholding."
            )
            self.segmented, self.nx_graph_pruned, self.pos = segment_fluo_new(
                [imageio.imread(addresses) for addresses in self.selection_file[:30]]
            )
        self.edges = list(self.nx_graph_pruned.edges)
        for i, edge in enumerate(self.edges):
            if self.pos[edge[0]][0] > self.pos[edge[1]][0]:
                self.edges[i] = edge
            else:
                self.edges[i] = (edge[1], edge[0])
        self.edges = self.filter_edges(filter_step * 2 // self.binning)
        self.edge_objects = [self.createEdge(edge) for edge in self.edges]

        if self.logging:
            print(
                "Successfully extracted the skeleton. Did you know there is a skeleton inside inside you right now?"
            )

        if show_seg:
            fig, ax = plt.subplots()
            ax.imshow(self.segmented)
            seg_shape = self.segmented.shape
            ax.set_title(
                f"Segmentation ({100 * np.sum(1 * self.segmented.flatten()) / (seg_shape[0] * seg_shape[1]):.4} $\%$ coverage)"
            )
            fig.tight_layout()

    def filter_edges(self, step):
        """
        Filter edges by length, return filtered edge object
        :param step:    Length threshold
        :return:        Edge objects which have come through the filter
        """
        filt_edge_objects = []
        for edge in self.edges:
            offset = int(np.linalg.norm(self.pos[edge[0]] - self.pos[edge[1]])) // 4
            if offset >= step:
                filt_edge_objects.append(edge)
        return filt_edge_objects

    def plot_extraction_img(
        self,
        weight=0.05,
        bounds=(0.0, 1.0),
        target_length=130,
        resolution=1,
        step=30,
        save_img=True,
        logging=False,
    ):
        """
        Further organises the graph structure of the video, and plots the edge maps.
        :param weight:          Padding for the text placement
        :param bounds:          Tuple of what fraction of the edge is used.
        :param target_length:   Width of the kymograph box
        :param resolution:      Spacing between kymograph perpendicular lines.
        :param step:            Extent of skeleton to use to smooth the kymograph box
        :param save_img:        Bool on whether to save the edge plot image
        :param logging:         Bool on whether to announce progress in console
        :return:                None
        """

        self.target_length = target_length * 2 // self.binning
        self.x_length = self.target_length * self.space_pixel_size
        image = imageio.imread(self.selection_file[self.im_range[0]])
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(
            self.segmented,
            extent=[
                0,
                self.space_pixel_size * image.shape[1],
                self.space_pixel_size * image.shape[0],
                0,
            ],
        )

        ax2.set_title("Segmentation and skeleton")
        fig2.tight_layout()
        if save_img:
            save_path_seg = os.path.join(self.kymos_path, f"Video segmentation.png")
            fig2.savefig(save_path_seg)
        plt.close(fig2)
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(
            image,
            extent=[
                0,
                self.space_pixel_size * image.shape[1],
                self.space_pixel_size * image.shape[0],
                0,
            ],
        )
        for edge in self.edge_objects:
            if logging:
                print("Working on edge {}, sir!".format(edge.edge_name))
            offset = (
                int(
                    np.linalg.norm(
                        self.pos[edge.edge_name[0]] - self.pos[edge.edge_name[1]]
                    )
                )
                // 4
            )
            segments = edge.create_segments(
                self.pos,
                image,
                self.nx_graph_pruned,
                resolution,
                offset,
                self.target_length,
                bounds,
                step=step,
            )
            # the segments are stored in a csv file
            #             print(type(segments))
            #             print(edge.edge_name)
            segment_frame = pd.DataFrame(segments)
            output_path_seg_csv = os.path.join(
                self.kymos_path, f"segment{edge.edge_name}.csv"
            )

            segment_frame.to_csv(output_path_seg_csv, index=False)

            # this plots the white regions onto the snapshot
            plot_segments_on_image(
                segments,
                ax1,
                bounds=bounds,
                color="white",
                alpha=0.1,
                adj=self.space_pixel_size,
            )
            # this creates colored lines along the edges
            ax1.plot(
                [
                    self.pos[edge.edge_name[0]][1] * self.space_pixel_size,
                    self.pos[edge.edge_name[1]][1] * self.space_pixel_size,
                ],
                [
                    self.pos[edge.edge_name[0]][0] * self.space_pixel_size,
                    self.pos[edge.edge_name[1]][0] * self.space_pixel_size,
                ],
            )
            # this puts white text along the strat and end
            ax1.text(
                *np.flip(
                    (1 - weight) * self.pos[edge.edge_name[0]]
                    + weight * self.pos[edge.edge_name[1]]
                )
                * self.space_pixel_size,
                str(edge.edge_name[0]),
                color="white",
            )
            ax1.text(
                *np.flip(
                    (1 - weight) * self.pos[edge.edge_name[1]]
                    + weight * self.pos[edge.edge_name[0]]
                )
                * self.space_pixel_size,
                str(edge.edge_name[1]),
                color="white",
            )
            ax1.set_title("Extracted Edges")
        print(
            f"To work with individual edges of {self.video_nr}, here is a list of their indices:"
        )
        for i, edge in enumerate(self.edge_objects):
            print("edge {}, {}".format(i, edge.edge_name))
        ax1.set_aspect("equal")
        fig1.tight_layout()
        plt.show()
        if save_img:
            save_path_temp = os.path.join(self.kymos_path, f"Detected edges.png")
            fig1.savefig(save_path_temp)
            print("Saved the extracted edges")

        plt.close(fig1)
        # this didn't work because edge.kymo is still empty, but moving it down in flux_extract solved that
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.imshow(
            image,
            extent=[
                0,
                self.space_pixel_size * image.shape[1],
                self.space_pixel_size * image.shape[0],
                0,
            ],
        )

        for edge in self.edge_objects:
            #
            ax3.text(
                *np.flip(
                    (1 - weight) * self.pos[edge.edge_name[0]]
                    + weight * self.pos[edge.edge_name[1]]
                )
                * self.space_pixel_size,
                f"{np.mean(edge.kymo)/100:.3}",
                color="white",
                fontsize="large",
            )
            ax3.text(
                *np.flip(
                    (1 - weight) * self.pos[edge.edge_name[1]]
                    + weight * self.pos[edge.edge_name[0]]
                )
                * self.space_pixel_size,
                f"{np.mean(edge.flux_tot)/100:.3}",
                color="white",
                fontsize="large",
            )
        ax3.set_aspect("equal")
        fig3.tight_layout()
        plt.show()

        if save_img:
            save_path_temp = os.path.join(self.kymos_path, f"Flux.png")
            fig3.savefig(save_path_temp)
        plt.close(fig3)
        return None

    def makeVideo(self, resize_ratio=4):
        """
        Creates a video from the raw TIFFs
        :param resize_ratio:    Resize the images to reduce video size
        :return:                None
        """
        vid_out = os.path.join(self.kymos_path, f"{self.video_nr}_video.mp4")
        print(vid_out)
        if os.path.exists(vid_out):
            os.remove(vid_out)
        with imageio.get_writer(vid_out, mode="I", fps=self.fps, quality=6) as writer:
            for img_path in self.selection_file:
                print(img_path)
                img = imageio.imread(img_path)
                image_resize = img[::resize_ratio, ::resize_ratio]
                assert img.dtype == np.uint8, "Must be uint8 array!"
                writer.append_data(image_resize)
        return None

    def createEdge(self, edge):
        """
        Creates an edge analysis object from the specified edge
        """
        return KymoEdgeAnalysis(self, edge_name=edge)

    def fit_backgr(self, plots=False):
        """
        Use a matrix problem to calculate the exponential falloff of a fluorescence video.
        If the exponential function with offset fitting does not work, will use a linear fit.
        :param plots:   Boolean on whether to plot the brightness of a video over time
        :return:        Fit variables
        """
        backgr_segm = self.segmented
        video_array = [imageio.imread(img) for img in self.selection_file]
        backgr_ints = [np.mean(img[backgr_segm].flatten()) for img in video_array]

        """
        Fit function for photobleaching. First assumes an exponential function with offset. 
        If the fitted function returns a positive exponent, will do a linear fit instead. 
        """
        s_fit = np.zeros((len(backgr_ints)))
        for i in range(len(backgr_ints)):
            if i > 0:
                s_fit[i] = s_fit[i - 1] + 0.5 * (backgr_ints[i] + backgr_ints[i - 1])
        t_fit = np.arange(len(backgr_ints))
        matr_1 = np.array(
            [
                [np.sum(s_fit**2), np.sum(s_fit * t_fit), np.sum(s_fit)],
                [np.sum(s_fit * t_fit), np.sum(t_fit**2), np.sum(t_fit)],
                [np.sum(s_fit), np.sum(t_fit), len(backgr_ints)],
            ]
        )
        vect_1 = np.array(
            [
                np.sum(s_fit * backgr_ints),
                np.sum(t_fit * backgr_ints),
                np.sum(backgr_ints),
            ]
        )
        answ_1 = np.linalg.inv(matr_1).dot(vect_1)
        if answ_1[0] > 0:
            print("Using linear fit")
            (m, b) = np.polyfit(t_fit, backgr_ints, 1)
            self.back_fit = m * t_fit + b
        else:
            matr_2 = np.array(
                [
                    [len(backgr_ints), np.sum(np.exp(answ_1[0] * t_fit))],
                    [
                        np.sum(np.exp(answ_1[0] * t_fit)),
                        np.sum(np.exp(2 * answ_1[0] * t_fit)),
                    ],
                ]
            )
            vect_2 = np.array(
                [np.sum(backgr_ints), np.sum(np.exp(answ_1[0] * t_fit) * backgr_ints)]
            )
            answ_2 = np.linalg.inv(matr_2).dot(vect_2)

            self.back_fit = [answ_2[0], answ_2[1], answ_1[0]]
            self.back_fit = answ_2[0] + answ_2[1] * np.exp(answ_1[0] * t_fit)
            self.back_offset = answ_2[0]
            print(f"{answ_2[0]:.4} + {answ_2[1]:.4} * exp({answ_1[0]:.4}x)")
        if plots:
            fig, ax = plt.subplots()
            ax.plot(backgr_ints, label="Average backgr intens")
            ax.plot(self.back_fit, label="Fitted exp w/ offset")
            ax.set_title("Intensity of background over time")
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Time (frames)")
            ax.legend()
            fig.tight_layout()
        return self.back_fit


class KymoEdgeAnalysis(object):
    def __init__(self, video_analysis=None, edge_name=None, kymo=None, address=None):
        """
        Create an edge analysis object through multiple means.
        :param video_analysis:  Added if created from KymoVideoAnalysis
        :param edge_name:       Added if created from KymoVideoAnalysis
        :param kymo:            Add kymograph image to create edge analysis object without attached video.
        """
        if kymo is None:
            self.video_analysis = video_analysis
            self.edge_name = edge_name
            self.offset = (
                int(
                    np.linalg.norm(
                        self.video_analysis.pos[self.edge_name[0]]
                        - self.video_analysis.pos[self.edge_name[1]]
                    )
                )
                // 4
            )
            self.kymo = []
            self.kymos = []
            self.edge_path = os.path.join(
                self.video_analysis.kymos_path, f"edge {self.edge_name}"
            )
            self.space_pixel_size = self.video_analysis.space_pixel_size
            self.time_pixel_size = self.video_analysis.time_pixel_size
        else:
            if len(kymo.shape) == 2:
                self.kymo = kymo
            self.kymos = [kymo]
            self.edge_name = edge_name
            self.space_pixel_size = 1.0
            self.time_pixel_size = 1.0
            self.edge_path = Path(address) / f"edge {self.edge_name}"

        if not os.path.exists(self.edge_path):
            os.makedirs(self.edge_path)
        self.filtered_left = []
        self.filtered_right = []
        self.slices = []
        self.segments = []
        self.bounds = (0, 1)
        self.edge_array = []
        self.speeds_tot = []
        self.flux_tot = []
        self.times = []
        self.imshow_extent = []
        self.fourier_speeds = []
        self.angle_plot = []
        self.ftabsimage = []
        self.ftpolarimage = []

    def view_edge(
        self,
        resolution=1,
        step=30,
        target_length=130,
        save_im=True,
        bounds=(0, 1),
        img_frame=0,
        quality=6,
    ):
        """
        Function to either output a snaphot of an edge or a video of an edge. Image will output to a plt plot.
        Video will be saved in the analysis folder.
        This function recalculates the segments,
        so make sure the target length is equal to that of the edge extraction plot.

        resolution:     Determines offset between perpendicular lines
        step:           Used to calculate orientation of profiles, smaller steps will produce more wobbly images.
        target_length:  Width of eventual image or video
        save_im:        Boolean whether the image or video should be saved
        bounds:         Fraction-based cutoff for where to start imaging. Probably interferes with target-length.
        img_frame:      Which frame index to use for imaging. Will produce a video if this is an array.
        quality:        Imageio parameter on video quality. 6 is a good balance of compression and file size.
        """
        self.edge_array = get_edge_image(
            self.edge_name,
            self.video_analysis.pos,
            self.video_analysis.selection_file,
            self.video_analysis.nx_graph_pruned,
            resolution,
            self.offset,
            step,
            target_length,
            img_frame,
            bounds,
            logging=self.video_analysis.logging,
        )

        if save_im:
            if np.ndim(img_frame) == 0:
                fig, ax = plt.subplots()
                ax.imshow(
                    self.edge_array,
                    aspect=1,
                    extent=[
                        0,
                        self.video_analysis.space_pixel_size * self.edge_array.shape[1],
                        0,
                        self.video_analysis.space_pixel_size * self.edge_array.shape[0],
                    ],
                )
                ax.set_title(f"Edge snapshot {self.edge_name}")
                ax.set_xlabel("space ($\mu m$)")
                ax.set_ylabel("space ($\mu m$)")

                # im = Image.fromarray(self.edge_array.astype(np.uint8))
                save_path_temp = os.path.join(
                    self.edge_path, f"{self.edge_name}_snapshot.png"
                )
                if os.path.exists(save_path_temp):
                    os.remove(save_path_temp)
                fig.savefig(save_path_temp)
                if self.video_analysis.logging:
                    print(f"Saved an image of edge {self.edge_name}")
                plt.close(fig)

                return [self.edge_array]
            elif np.ndim(img_frame) == 1:
                vid_out = os.path.join(
                    self.edge_path, f"{self.edge_name}_edge_video.mp4"
                )
                if os.path.exists(vid_out):
                    os.remove(vid_out)
                imageio.mimwrite(
                    vid_out,
                    self.edge_array,
                    quality=quality,
                    fps=self.video_analysis.fps,
                )
            else:
                print(
                    "Input image sequence has a weird amount of dimensions. This will probably crash"
                )
        return self.edge_array

    def create_segments(
        self,
        pos,
        image,
        nx_graph_pruned,
        resolution,
        offset,
        target_length,
        bounds,
        step=30,
    ):
        """
        Internal function which generates coordinates for kymograph and edge videos
        """
        self.slices, self.segments = extract_section_profiles_for_edge(
            self.edge_name,
            pos,
            image,
            nx_graph_pruned,
            resolution=resolution,
            offset=offset,
            target_length=target_length,
            bounds=bounds,
            step=step,
        )

        return self.segments

    def get_widths(
        self,
        resolution=1,
        step=30,
        target_length=200,
        save_im=True,
        bounds=(0, 1),
        img_frame=0,
        quality=6,
        model_path=Path(".").absolute().parents[1]
        / "ml/models/default_CNN_GT_model.h5",
    ):
        """
        Returns the width of a selected edge
        :param resolution:      Distance between kymograph perpendicular lines
        :param step:            Smoothing factor of kymograph
        :param target_length:   Keep at 200, determines width of kymo perpendicular lines. Network model only uses 200 pixels.
        :param save_im:
        :param bounds:          Fraction tuple on what part of the kymograph to use
        :param img_frame:
        :param quality:
        :param model_path:      Folder address where the model to be used is located.
        :return:
        """
        if self.video_analysis.vid_type == "BRIGHT":
            self.create_segments(
                self.video_analysis.pos,
                imageio.imread(
                    self.video_analysis.selection_file[self.video_analysis.im_range[0]]
                ),
                self.video_analysis.nx_graph_pruned,
                resolution,
                4,
                target_length,
                bounds,
                step=step,
            )
            width_model = tf.keras.models.load_model(model_path)
            self.widths = width_model.predict(self.slices, verbose=0)
        else:
            self.create_segments(
                self.video_analysis.pos,
                self.video_analysis.segmented,
                self.video_analysis.nx_graph_pruned,
                resolution,
                4,
                target_length,
                bounds,
                step=step,
            )
            self.widths = [
                max(
                    (
                        sum(1 for _ in group)
                        for value, group in itertools.groupby(pixel_row)
                        if value == 0
                    ),
                    default=0,
                )
                for pixel_row in self.slices
            ]
            self.widths = np.array(self.widths) * self.video_analysis.space_pixel_size
        return self.widths

    def extract_multi_kymo(
        self,
        bin_nr,
        resolution=1,
        step=30,
        target_length=130,
        kymo_adj=False,
        save_array=False,
        save_im=False,
        kymo_normalize=False,
        bounds=(0, 1),
    ):
        """
        Creates kymograph for the edge. Uses bin_nr to divide the width into evenly distributed strips.
        Kymographs are stored in the object, and images will be stored in the analysis folder.

        :param bin_nr:         Number of evenly distributed width strips to make kymogrpahs from
        :param resolution:     Something related to section creation
        :param step:           Quality parameter in calculating the orientation of the edges
        :param target_length:  Width of edge to make kymographs from, really should be just the internal length.
        :param save_array:     Boolean on whether to make a .npy file of the array
        :param save_im:        Boolean on whether to save the kymographs as images
        :param bounds:         Fraction-based limit on edge length
        :return:               Extracted kymograph arrays
        """
        self.bin_edges = np.linspace(bounds[0], bounds[1], bin_nr + 1)
        if self.space_pixel_size is None:
            space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
        else:
            space_pixel_size = self.space_pixel_size
        self.kymos = np.array(
            [
                self.extract_kymo(
                    bounds=(self.bin_edges[i], self.bin_edges[i + 1]),
                    resolution=resolution,
                    step=step,
                    target_length=target_length * 2 // self.video_analysis.binning,
                    save_array=save_array,
                    save_im=save_im,
                    img_suffix=str(bin_nr) + " " + str(i + 1),
                    kymo_adjust=kymo_adj,
                    kymo_normalize=kymo_normalize,
                    x_len=space_pixel_size,
                )
                for i in tqdm(range(bin_nr))
            ]
        )

        return self.kymos

    def extract_kymo(
        self,
        resolution=1,
        step=30,
        target_length=130,
        save_array=True,
        save_im=True,
        bounds=(0, 1),
        img_suffix="",
        x_len=10,
        kymo_adjust=False,
        kymo_normalize=False,
    ):
        """
        Single version of kymograph extraction, is used as a sub-function in multi_kymo_extract
        :param resolution:      Skipping factor for perpendicular kymo lines
        :param step:            SMoothing factor for kymograph
        :param target_length:   Width of kymogeaph box
        :param save_array:      Whether to save kymograph as np array
        :param save_im:         Whether to save kymograph as png
        :param bounds:          Fraction tuple what fraction of edge width to use.
        :param img_suffix:      fuidsvbsdi
        :param x_len:           Spacial resolution of images
        :param kymo_adjust:     Deprecated
        :return:                Kymograph as array
        """
        self.kymo = get_kymo_new(
            self.edge_name,
            self.video_analysis.pos,
            self.video_analysis.selection_file,
            self.video_analysis.nx_graph_pruned,
            resolution,
            self.offset,
            step,
            target_length,
            bounds,
            x_len,
        )
        if kymo_normalize:
            self.kymo -= np.nanmin(self.kymo)
            self.kymo *= 255.0 / np.nanmax(self.kymo)
        if kymo_adjust:
            norm_exp = self.video_analysis.back_fit / self.video_analysis.back_fit[0]
            self.kymo -= self.video_analysis.back_offset
            self.kymo = np.array(
                [self.kymo[i] / norm_exp[i] for i in range(self.kymo.shape[0])]
            )
        if save_array:
            save_path_temp = os.path.join(
                self.edge_path, f"{self.edge_name} {img_suffix} kymo.npy"
            )
            np.save(save_path_temp, (self.kymo))
        if save_im:
            im = Image.fromarray((255 * self.kymo / np.max(self.kymo)).astype(np.uint8))
            save_path_temp = os.path.join(
                self.edge_path, f"{self.edge_name} {img_suffix} kymo.png"
            )
            im.save(save_path_temp)
        return self.kymo

    def fourier_kymo(self, return_self=True):
        """
        Separates forward and backward moving lines in kymographs
        :param return_self: Whether to return separation as a list of arrays
        :param test_plots:  Whether to plot the separated kymographs, either way they are stored in the object

        :return:            Either list of separated kymographs, or None
        """

        if len(self.kymos) > 0:
            self.filtered_left = np.array(
                [filter_kymo_left(kymo) for kymo in self.kymos]
            )
            self.filtered_right = np.array(
                [
                    np.flip(
                        filter_kymo_left(np.flip(kymo, axis=1)),
                        axis=1,
                    )
                    for kymo in self.kymos
                ]
            )
        if return_self:
            return self.filtered_left, self.filtered_right
        else:
            return None

    def extract_speeds(
        self,
        w_size,
        w_start=3,
        C_thresh=0.95,
        blur_size=7,
        C_thresh_falloff=0.002,
        speed_thresh=90,
        preblur=True,
    ):
        """
        Extract speeds from the kymographs. Will iterate different window sizes to get different particle size velocities
        :param w_size:          Number of window sizes to filter over the kymograph
        :param w_start:         Smallest window size
        :param C_thresh:        Confidence threshold above which to use the extracted orientation
        :param blur_size:       Gaussian window size to use in the blur step of the kymograph
        :param C_thresh_falloff:Value to subtract from C_thresh with each increasing window size
        :param speed_thresh:    Maximum speed to record.
        :param preblur:         Whether to blur the kymograph before extracting speeds
        :param plots:           Deprecated
        :return:                list of array of speeds and time-steps
        """

        """Initialize the speed array and time array"""
        times = []
        speeds_tot = []
        if len(self.filtered_left) == 0:
            self.fourier_kymo(return_self=False)

        """Get the real pixel space and time sizes either from the video, or set them as one if testing"""
        if self.space_pixel_size is None:
            space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
            time_pixel_size = self.video_analysis.time_pixel_size  # s.pixel
            self.imshow_extent = [
                0,
                self.video_analysis.space_pixel_size * self.kymos[0].shape[1],
                self.video_analysis.time_pixel_size * self.kymos[0].shape[0],
                0,
            ]
        else:
            space_pixel_size = self.space_pixel_size
            time_pixel_size = self.time_pixel_size
            self.imshow_extent = [
                0,
                self.space_pixel_size * self.kymos[0].shape[1],
                self.time_pixel_size * self.kymos[0].shape[0],
                0,
            ]

        """Iterate over each kymograph, over the forward and backward directions, and over the window sizes"""
        for j, kymo in enumerate(self.kymos):

            """Set up the times axis, should be identical every iteration, also the forward/backward speed arrays"""
            times.append(np.array(range(kymo.shape[0])) * time_pixel_size)
            spds_both = [[], []]
            for i in [0, 1]:

                """Set up speed array for this iteration, also select either forward or backward filtered kymograph"""
                spd_stack = []
                kymo_interest = [self.filtered_right[j], self.filtered_left[j]][i]

                """Preblur gets rid of some gaussian noise in the kymograph"""
                if preblur:
                    kymo_interest = cv2.GaussianBlur(
                        kymo_interest, (blur_size, blur_size), 0
                    )

                """
                imgCoherency is an array with all the coherency and orientations of any given window size,
                imgCoherencySum is meant to document at each pixel which window size provides the highest image coherency
                imgGSTMax is supposed to contain the orientations of the pixels with the highest intensity,
                In the setting up phase, we put nans where the threshold is lower than desired.
                
                Then in the window iterations, we check for what the max image coherency is, 
                and change the image orientation to that.
                """

                imgCoherency = np.array(
                    [
                        calcGST(kymo_interest, w)
                        for w in range(w_start, w_size * 2 + w_start, 2)
                    ]
                )
                imgCoherencySum = 1 * np.greater(imgCoherency[0][0], C_thresh)
                imgCoherencySum = np.where(imgCoherencySum == 1, 0, np.nan)
                imgGSTMax = np.where(imgCoherencySum == 0, imgCoherency[0][1], np.nan)

                for w in range(1, w_size):
                    C_thresh_current = C_thresh - C_thresh_falloff * w
                    coherency_interest = np.where(
                        imgCoherency[w][0] > C_thresh_current,
                        imgCoherency[w][0],
                        np.nan,
                    )
                    imgCoherencySum = np.where(
                        coherency_interest > imgCoherency[w - 1][0], w, imgCoherencySum
                    )
                    newValues = np.isnan(imgCoherencySum) * (
                        np.invert(np.isnan(coherency_interest))
                    )
                    imgCoherencySum = np.where(newValues, w, imgCoherencySum)
                    imgGSTMax = np.where(
                        imgCoherencySum == w, imgCoherency[w][1], imgGSTMax
                    )

                """
                The speed is thresholded based on the max speed threshold, and that we don't expect forward speeds in the 
                backwards filtered image and vice versa.
                """

                speed_unthr = (
                    np.tan((imgGSTMax - 90) / 180 * np.pi)
                    * space_pixel_size
                    / time_pixel_size
                )
                speed = np.where(speed_unthr < speed_thresh, speed_unthr, np.nan)
                speed = np.where(speed > -1 * speed_thresh, speed, np.nan)
                spd_interest = np.where([speed < 0, speed > 0][i], speed, np.nan)
                spds_both[i] = spd_interest
                spd_stack.append(spd_interest)

            speeds_tot.append(spds_both)
            self.speeds_tot = np.array(speeds_tot)
            self.times = times
        return np.array(speeds_tot), times

    def extract_transport(self):
        """
        Function that extracts the flux of internal kymographs.
        """
        flux_arrays = []
        for k, kymo in enumerate(self.kymos):
            back_thresh, forw_thresh = (self.filtered_right[k], self.filtered_left[k])
            speeds = self.speeds_tot
            spds_back = speeds[k][0]
            spds_forw = speeds[k][1]
            speed_right = np.nanmean(speeds[k][0])
            speed_left = np.nanmean(speeds[k][1])
            flux_non_nan = ~(np.isnan(spds_back) * np.isnan(spds_forw))
            flux_right = np.nanmean(back_thresh) * speed_right
            flux_left = np.nanmean(forw_thresh) * speed_left

            flux_tot = np.nansum(
                (
                    np.prod((spds_forw, forw_thresh), 0),
                    np.prod((spds_back, back_thresh), 0),
                ),
                0,
            )
            flux_tot = np.where(flux_non_nan, flux_tot, np.nan)
            self.flux_tot = flux_tot
            self.flux_left = np.nanmean(flux_left)
            self.flux_right = np.nanmean(flux_right)

            flux_arrays.append(flux_tot)
        if len(kymo) == 1:
            return flux_tot
        else:
            return flux_arrays

    def fourier_analysis(self, R_thresh=50, prominence=0.5):
        for k, img in enumerate(self.kymos):
            img_dims = np.array(img.shape)
            ftimage = np.fft.fft2(img)
            ftimage = np.fft.fftshift(ftimage)
            self.ftabsimage = np.log(abs(ftimage.real))
            self.ftpolarimage = cv2.warpPolar(
                self.ftabsimage,
                (-1, -1),
                (img_dims[1] / 2, img_dims[0] / 2),
                img_dims[1],
                cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
            )
            polardims = self.ftpolarimage.shape
            #             print(polardims)
            self.ftpolarimage = self.ftpolarimage[
                0 : polardims[0] // 2, R_thresh : polardims[1] // 4
            ]
            self.ftpolarimage = np.where(
                self.ftpolarimage == 0, np.nan, self.ftpolarimage
            )
            angle_plot = [
                np.linspace(0, np.pi, len(self.ftpolarimage)),
                np.nanmean(self.ftpolarimage, axis=1),
            ]
            #             print(angle_plot[1])

            self.angle_plot = [
                np.tan((angle_plot[0] - 0.5 * np.pi))
                * self.space_pixel_size
                / self.time_pixel_size,
                angle_plot[1],
            ]
            self.fourier_peak_data = find_peaks(angle_plot[1], prominence=prominence)
            peaks, _ = self.fourier_peak_data
            speeds = (
                np.tan((angle_plot[0][peaks] - 0.5 * np.pi))
                * self.space_pixel_size
                / self.time_pixel_size
            )
            print(speeds)
            self.fourier_speeds = speeds
        return self.fourier_peak_data
