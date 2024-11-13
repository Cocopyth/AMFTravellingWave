from amftrack.util.dbx import upload
from scipy import spatial

import cv2

from amftrack.util.video_util import (
    make_hull_images,
    make_video_tile,
    make_images_track,
    make_images_track2,
    make_images_track3,
    make_images_anas,
    make_images_track_hypha,
    make_images_spores,
    make_images_width,
    make_images_betweenness_random,
    make_images_betweenness,
)
import networkx as nx

from amftrack.util.sys import temp_path
import numpy as np
from amftrack.pipeline.functions.post_processing.area_hulls import is_in_study_zone
import os

is_circle = False

dir_drop = "DATA/PRINCE_ANALYSIS"


def delete_files(paths_list):
    for paths in paths_list:
        for path in paths:
            os.remove(path)


def plot_hulls(exp, args=None):
    paths_list = make_hull_images(exp, range(exp.ts))
    id_unique = exp.unique_id
    texts = [[folder] for folder in list(exp.folders["folder"])]
    folder_analysis = exp.save_location.split("/")[-1]
    upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/{id_unique}_hulls.mp4"
    print(upload_path)
    make_video_tile(
        paths_list, texts, None, save_path=None, upload_path=upload_path, fontScale=3
    )
    delete_files(paths_list)


def plot_get_hull_nodes(exp, args=None):
    dir_save = os.path.join(exp.save_location, "time_hull_info")
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)
    for t in range(exp.ts):
        full_hull_nodes = []
        nx_graph = exp.nx_graph[t]
        threshold = 0.1
        S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
        selected = [
            g
            for g in S
            if g.size(weight="weight") * len(g.nodes) / 10**6 >= threshold
        ]
        for g in selected:
            nodes_pos = np.array(
                [
                    node.pos(t)
                    for node in exp.nodes
                    if node.is_in(t)
                    and np.all(is_in_study_zone(node, t, 1000, 150))
                    and (node.label in g.nodes)
                ]
            )
            nodes_label = np.array(
                [
                    node.label
                    for node in exp.nodes
                    if node.is_in(t)
                    and np.all(is_in_study_zone(node, t, 1000, 150))
                    and (node.label in g.nodes)
                ]
            )
            if len(nodes_pos) > 3:
                hull = spatial.ConvexHull(nodes_pos)
                hull_nodes = [nodes_label[vertice] for vertice in hull.vertices]
                full_hull_nodes += hull_nodes
        np.save(
            os.path.join(dir_save, f"hull_nodes_{t}.npy"),
            full_hull_nodes,
        )


def plot_tracking(exp, args=None):
    paths_list = make_images_track(exp, is_circle)
    id_unique = exp.unique_id
    folder_analysis = exp.save_location.split("/")[-1]
    upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/full_track/"
    texts = [(folder) for folder in list(exp.folders["folder"])]
    fontScale = 3
    color = (0, 255, 255)
    for i, paths in enumerate(paths_list):
        path = paths[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
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
        cv2.imwrite(path, img)
        upload_path_im = os.path.join(upload_path, f"frame_{i}.png")
        upload(path, upload_path_im)
    delete_files(paths_list)


def plot_blobs(exp, args=None):
    paths_list = make_images_spores(exp)
    id_unique = exp.unique_id
    folder_analysis = exp.save_location.split("/")[-1]
    upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/full_spores/"
    texts = [(folder) for folder in list(exp.folders["folder"])]
    fontScale = 3
    color = (0, 255, 255)
    for i, paths in enumerate(paths_list):
        path = paths[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
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
        cv2.imwrite(path, img)
        upload_path_im = os.path.join(upload_path, f"frame_{i}.png")
        upload(path, upload_path_im)
    delete_files(paths_list)


def plot_tracking2(exp, args=None):
    for i in range(100):
        paths_list = make_images_track2(exp)
        id_unique = exp.unique_id
        folder_analysis = exp.save_location.split("/")[-1]
        upload_path = (
            f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/{i}_tracked.mp4"
        )
        texts = [(folder, "", "", "") for folder in list(exp.folders["folder"])]
        resize = (2048, 2048)
        make_video_tile(
            paths_list,
            texts,
            resize,
            save_path=None,
            upload_path=upload_path,
            fontScale=3,
        )
        delete_files(paths_list)


def plot_hypha(exp, args=None):
    for i in range(100):
        paths_list = make_images_track3(exp)
        id_unique = exp.unique_id
        folder_analysis = exp.save_location.split("/")[-1]
        upload_path = (
            f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/{i}_hyphae.mp4"
        )
        texts = [(folder, "", "", "") for folder in list(exp.folders["folder"])]
        resize = (2048, 2048)
        make_video_tile(
            paths_list,
            texts,
            resize,
            save_path=None,
            upload_path=upload_path,
            fontScale=3,
        )
        delete_files(paths_list)


def plot_hypha_track(exp, args=[]):
    for hypha in args:
        paths_list = make_images_track_hypha(exp, hypha)
        id_unique = exp.unique_id
        folder_analysis = exp.save_location.split("/")[-1]
        upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/tracked_hyphae/hypha_{hypha}.mp4"
        texts = [(folder, "", "", "") for folder in list(exp.folders["folder"])]
        resize = (2048, 2048)
        make_video_tile(
            paths_list,
            texts,
            resize,
            save_path=None,
            upload_path=upload_path,
            fontScale=3,
        )
        delete_files(paths_list)


def plot_anastomosis(exp, args=None):
    paths_list = make_images_anas(exp)
    id_unique = exp.unique_id
    folder_analysis = exp.save_location.split("/")[-1]
    upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/full_anas/"
    texts = [(folder) for folder in list(exp.folders["folder"])]
    fontScale = 3
    color = (0, 255, 255)
    for i, paths in enumerate(paths_list):
        path = paths[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
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
        cv2.imwrite(path, img)
        upload_path_im = os.path.join(upload_path, f"frame_{i}.png")
        # upload(path, upload_path_im)
    delete_files(paths_list)


def plot_width(exp, args=None):
    paths_list = make_images_width(exp)
    id_unique = exp.unique_id
    folder_analysis = exp.save_location.split("/")[-1]
    upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/full_width/"
    texts = [(folder) for folder in list(exp.folders["folder"])]
    fontScale = 3
    color = (0, 255, 255)
    for i, paths in enumerate(paths_list):
        path = paths[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
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
        cv2.imwrite(path, img)
        upload_path_im = os.path.join(upload_path, f"frame_{i}.png")
        # upload(path, upload_path_im)
    delete_files(paths_list)

#
# def plot_betweenness(exp, args=None):
#     paths_list = make_images_betweenness(exp)
#     id_unique = exp.unique_id
#     folder_analysis = exp.save_location.split("/")[-1]
#     upload_path = f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/betweenness/"
#     texts = [(folder) for folder in list(exp.folders["folder"])]
#     fontScale = 3
#     color = (0, 255, 255)
#     for i, paths in enumerate(paths_list):
#         path = paths[0]
#         img = cv2.imread(path, cv2.IMREAD_COLOR)
#         anchor = img.shape[0] // 10, img.shape[1] // 10
#         cv2.putText(
#             img=img,
#             text=texts[i],
#             org=anchor,
#             fontFace=cv2.FONT_HERSHEY_TRIPLEX,
#             fontScale=fontScale,
#             color=color,
#             thickness=3,
#         )
#         cv2.imwrite(path, img)
#         upload_path_im = os.path.join(upload_path, f"frame_{i}.png")
#         upload(path, upload_path_im)
#     delete_files(paths_list)
#
#
# def plot_betweenness_random(exp, args=None):
#     paths_list = make_images_betweenness_random(exp)
#     id_unique = exp.unique_id
#     folder_analysis = exp.save_location.split("/")[-1]
#     upload_path = (
#         f"/{dir_drop}/{id_unique}/{folder_analysis}/validation/betweenness_random/"
#     )
#     texts = [(folder) for folder in list(exp.folders["folder"])]
#     fontScale = 3
#     color = (0, 255, 255)
#     for i, paths in enumerate(paths_list):
#         path = paths[0]
#         img = cv2.imread(path, cv2.IMREAD_COLOR)
#         anchor = img.shape[0] // 10, img.shape[1] // 10
#         cv2.putText(
#             img=img,
#             text=texts[i],
#             org=anchor,
#             fontFace=cv2.FONT_HERSHEY_TRIPLEX,
#             fontScale=fontScale,
#             color=color,
#             thickness=3,
#         )
#         cv2.imwrite(path, img)
#         upload_path_im = os.path.join(upload_path, f"frame_{i}.png")
#         upload(path, upload_path_im)
#     delete_files(paths_list)


# API = str(np.load(os.getenv("HOME") + "/pycode/API_drop.npy"))
# dir_drop = "prince_data"


# def make_movie_raw(directory, folders, plate_num, exp, id_unique, args=None):
#     skels = []
#     ims = []
#     kernel = np.ones((5, 5), np.uint8)
#     itera = 1
#     for folder in folders:
#         directory_name = folder
#         path_snap = directory + directory_name
#         skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned_compressed.mat")
#         skel = skel_info["skeleton"]
#         skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
#         im = read_mat(path_snap + "/Analysis/raw_image.mat")["raw"]
#         ims.append(im)
#     start = 0
#     finish = len(exp.dates)
#     for i in range(start, finish):
#         plt.close("all")
#         clear_output(wait=True)
#         plot_t_tp1(
#             [],
#             [],
#             None,
#             None,
#             skels[i],
#             ims[i],
#             save=f"{temp_path}/{plate_num}_im{i}.png",
#             time=f"t = {int(get_time(exp, 0, i))}h",
#         )
#         plt.close("all")
#
#     img_array = []
#     for t in range(start, finish):
#         img = cv2.imread(f"{temp_path}/{plate_num}_im{t}.png")
#         img_array.append(img)
#
#     path_movie = f"{temp_path}/{plate_num}.gif"
#     imageio.mimsave(path_movie, img_array, duration=1)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_raw.gif",
#         chunk_size=256 * 1024 * 1024,
#     )
#     path_movie = f"{temp_path}/{plate_num}.mp4"
#     imageio.mimsave(path_movie, img_array)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_raw.mp4",
#         chunk_size=256 * 1024 * 1024,
#     )
#
#
# def make_movie_aligned(directory, folders, plate_num, exp, id_unique, args=None):
#     skels = []
#     ims = []
#     kernel = np.ones((5, 5), np.uint8)
#     itera = 2
#     for folder in folders:
#         directory_name = folder
#         path_snap = directory + directory_name
#         skel_info = read_mat(path_snap + "/Analysis/skeleton_realigned_compressed.mat")
#         skel = skel_info["skeleton"]
#         skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
#     start = 0
#     finish = len(exp.dates)
#     for i in range(start, finish):
#         plt.close("all")
#         clear_output(wait=True)
#         plot_t_tp1(
#             [],
#             [],
#             None,
#             None,
#             skels[i],
#             skels[i],
#             save=f"{temp_path}/{plate_num}_im{i}",
#             time=f"t = {int(get_time(exp,0,i))}h",
#         )
#         plt.close("all")
#     img_array = []
#     for t in range(start, finish):
#         img = cv2.imread(f"{temp_path}/{plate_num}_im{t}.png")
#         img_array.append(img)
#     path_movie = f"{temp_path}/{plate_num}.gif"
#     imageio.mimsave(path_movie, img_array, duration=1)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_aligned.gif",
#         chunk_size=256 * 1024 * 1024,
#     )
#     path_movie = f"{temp_path}/{plate_num}.mp4"
#     imageio.mimsave(path_movie, img_array)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_aligned.mp4",
#         chunk_size=256 * 1024 * 1024,
#     )


# def make_movie_RH_BAS(directory, folders, plate_num, exp, id_unique, args=None):
#     time_plate_info, global_hypha_info, time_hypha_info = get_data_tables(
#         redownload=True
#     )
#     table = global_hypha_info.loc[global_hypha_info["Plate"] == plate_num].copy()
#     table["log_length"] = np.log10((table["tot_length_C"] + 1).astype(float))
#     table["is_rh"] = (table["log_length"] >= 3.36).astype(int)
#     table = table.set_index("hypha")
#     for t in range(exp.ts):
#         plt.close("all")
#         clear_output(wait=True)
#         segs = []
#         colors = []
#         hyphaes = table.loc[
#             (table["strop_track"] >= t)
#             & (table["timestep_init_growth"] <= t)
#             & ((table["out_of_ROI"].isnull()) | (table["out_of_ROI"] > t))
#         ].index
#         for hyph in exp.hyphaes:
#             if t in hyph.ts and hyph.end.label in hyphaes:
#                 try:
#                     nodes, edges = hyph.get_nodes_within(t)
#                     color = (
#                         "red"
#                         if np.all(table.loc[table.index == hyph.end.label]["is_rh"])
#                         else "blue"
#                     )
#                     # color = 'green' if np.all(table.loc[table.index == hyph.end.label]['is_small']) else color
#                     for edge in edges:
#                         origin, end = edge.end.get_pseudo_identity(t).pos(
#                             t
#                         ), edge.begin.get_pseudo_identity(t).pos(t)
#                         segs.append((origin, end))
#                         colors.append(color)
#                 except nx.exception.NetworkXNoPath:
#                     pass
#         segs = [(np.flip(origin) // 5, np.flip(end) // 5) for origin, end in segs]
#         skels = []
#         ims = []
#         kernel = np.ones((5, 5), np.uint8)
#         itera = 2
#         folders = list(exp.folders["folder"])
#         folders.sort()
#         for folder in folders[t : t + 1]:
#             directory_name = folder
#             path_snap = directory + directory_name
#             skel_info = read_mat(
#                 path_snap + "/Analysis/skeleton_realigned_compressed.mat"
#             )
#             skel = skel_info["skeleton"]
#             skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
#         i = 0
#         ln_coll = matplotlib.collections.LineCollection(segs, colors=colors)
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.imshow(skels[i])
#         ax.add_collection(ln_coll)
#         plt.draw()
#         right = 0.80
#         top = 0.80
#         fontsize = 20
#         ax.text(
#             right,
#             top,
#             f"t = {int(get_time(exp,0,t))}h",
#             horizontalalignment="right",
#             verticalalignment="bottom",
#             transform=ax.transAxes,
#             color="white",
#             fontsize=fontsize,
#         )
#         plt.savefig(f"{temp_path}/{plate_num}_rhbas_im{t}")
#         plt.close(fig)
#     img_array = []
#     for t in range(exp.ts):
#         img = cv2.imread(f"{temp_path}/{plate_num}_rhbas_im{t}.png")
#         img_array.append(img)
#     path_movie = f"{temp_path}/{plate_num}_rhbas.gif"
#     imageio.mimsave(path_movie, img_array, duration=1)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_rhbas.gif",
#         chunk_size=256 * 1024 * 1024,
#     )
#     path_movie = f"{temp_path}/{plate_num}.mp4"
#     imageio.mimsave(path_movie, img_array)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_rhbas.mp4",
#         chunk_size=256 * 1024 * 1024,
#     )


# def make_movie_dens(directory, folders, plate_num, exp, id_unique, args=None):
#     time_plate_info, global_hypha_info, time_hypha_info = get_data_tables(
#         redownload=True
#     )
#     ts = range(exp.ts)
#     incr = 100
#     regular_hulls, indexes = get_regular_hulls_area_fixed(exp, ts, incr)
#     plate = plate_num
#     table = time_plate_info.loc[time_plate_info["Plate"] == plate]
#     table = table.fillna(-1)
#     my_cmap = cm.Greys
#     my_cmap.set_under("k", alpha=0)
#     table = table.set_index("t")
#     num_hulls = len(regular_hulls) - 4
#     for t in ts:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         skels = []
#         ims = []
#         kernel = np.ones((5, 5), np.uint8)
#         itera = 2
#         folders = list(exp.folders["folder"])
#         folders.sort()
#         for folder in folders[t : t + 1]:
#             directory_name = folder
#             path_snap = directory + directory_name
#             skel_info = read_mat(
#                 path_snap + "/Analysis/skeleton_realigned_compressed.mat"
#             )
#             skel = skel_info["skeleton"]
#             skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
#         for index in range(num_hulls):
#             polygon = regular_hulls[num_hulls - 1 - index]
#             column = f"ring_density_incr-100_index-{num_hulls - 1 - index}"
#             p = affine_transform(polygon, [0.2, 0, 0, -0.2, 0, 0])
#             p = rotate(p, 90, origin=(0, 0))
#             density = table[column][t]
#             if density != -1:
#                 p = gpd.GeoSeries(p)
#                 _ = p.plot(ax=ax, color=cm.cool(density / 5000), alpha=0.9)
#         polygon = regular_hulls[0]
#         p = affine_transform(polygon, [0.2, 0, 0, -0.2, 0, 0])
#         p = rotate(p, 90, origin=(0, 0))
#         p = gpd.GeoSeries(p)
#         _ = p.plot(ax=ax, color="black")
#         norm = mpl.colors.Normalize(vmin=0, vmax=3000)
#         fig.colorbar(
#             cm.ScalarMappable(norm=norm, cmap=cm.cool), ax=ax, orientation="horizontal"
#         )
#         ax.imshow(
#             skels[0], vmin=0.00000001, cmap=my_cmap, zorder=30, interpolation=None
#         )
#         right = 0.90
#         top = 0.90
#         fontsize = 10
#         text = ax.text(
#             right,
#             top,
#             f"time = {int(table['time_since_begin'][t])}h",
#             horizontalalignment="right",
#             verticalalignment="bottom",
#             transform=ax.transAxes,
#             color="white",
#             fontsize=fontsize,
#         )
#         plt.close(fig)
#         plt.savefig(f"{temp_path}/{plate_num}_dens_im{t}")
#     img_array = []
#     for t in range(exp.ts):
#         img = cv2.imread(f"{temp_path}/{plate_num}_dens_im{t}.png")
#         img_array.append(img)
#     path_movie = f"{temp_path}/{plate_num}_dens.gif"
#     imageio.mimsave(path_movie, img_array, duration=1)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_dens.gif",
#         chunk_size=256 * 1024 * 1024,
#     )
#     path_movie = f"{temp_path}/{plate_num}_dens.mp4"
#     imageio.mimsave(path_movie, img_array)
#     upload(
#         path_movie,
#         f"/{dir_drop}/{id_unique}/movie_dens.mp4",
#         chunk_size=256 * 1024 * 1024,
#     )
