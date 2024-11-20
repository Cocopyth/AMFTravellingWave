import numpy as np
import pandas as pd

# import open3d as o3d
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_all_edges,
)

# import probreg as pr
from scipy.optimize import minimize

from pathlib import Path


def get_segments_ends(
    vid_obj, shiftx, shifty, thresh_length=0, R=np.array([[1, 0], [0, 1]]), t=0
):
    segments = []
    for i in range(len(vid_obj.edge_objs)):
        edge = vid_obj.edge_objs[i]
        x_pos_video = edge.mean_data["xpos_network"]
        y_pos_video = edge.mean_data["ypos_network"]

        x_pos1 = (
            edge.edge_infos["edge_xpos_1"] * edge.space_res / 1.725
            + x_pos_video
            - shiftx
        )
        x_pos2 = (
            edge.edge_infos["edge_xpos_2"] * edge.space_res / 1.725
            + x_pos_video
            - shiftx
        )
        y_pos1 = (
            edge.edge_infos["edge_ypos_1"] * edge.space_res / 1.725
            + y_pos_video
            - shifty
        )
        y_pos2 = (
            edge.edge_infos["edge_ypos_2"] * edge.space_res / 1.725
            + y_pos_video
            - shifty
        )
        begin = transform(np.array([x_pos1, y_pos1]), R, t).tolist()
        end = transform(np.array([x_pos2, y_pos2]), R, t).tolist()
        length = np.linalg.norm(np.array([x_pos1, y_pos1]) - np.array([x_pos2, y_pos2]))
        if length > thresh_length:
            # print(length)
            segments.append([begin, end])
    return segments


def register_rot_trans(
    vid_obj, exp, t, dist=100, R=np.array([[1, 0], [0, 1]]), trans=0, thresh=20
):
    """Finds the rotation and translation to better align videos' edges on
    network edges"""
    positions = R @ np.array(vid_obj.dataset[["xpos_network", "ypos_network"]]) + trans
    shiftx, shifty = get_shifts(vid_obj)
    segments = get_segments_ends(vid_obj, shiftx, shifty, thresh, R, trans)
    edges = get_all_edges(exp, t)
    edges = [edge for edge in edges if dist_edge(edge, positions, t) <= dist]
    pixels = [pixel for edge in edges for pixel in edge.pixel_list(t)]
    pixels = [
        pixel for pixel in pixels if np.linalg.norm(pixel - positions) <= 1.5 * dist
    ]
    segment_points = []
    for begin, end in segments:
        # Include the start point, interpolated points, and the end point
        interpolated_points = interpolate_points(begin, end)
        segment_points.extend(interpolated_points)
        segment_points.append(end)  # Ensure the end point is included

    segment_points = np.array(segment_points)
    Y = np.array(pixels)
    X = np.array(segment_points)
    if len(X) > 0 and len(Y) > 0:
        Rfound, tfound = find_optimal_R_and_t(X, Y)
        # Rfound = transformation[0:2, 0:2]
        # tfound = transformation[0:2, 3]
        if np.linalg.det(Rfound) > 0:
            return (Rfound, tfound)
        else:
            # print("negative det")
            Rfound, tfound = find_optimal_R_and_t(X, Y)
            Rfound, tfound = np.linalg.inv(Rfound), np.dot(
                np.linalg.inv(Rfound), -tfound
            )
            return (Rfound, tfound)
    else:
        return initialize_transformation()


# def find_rot_o3d(X, Y):
#     X = np.transpose(X)
#     Y = np.transpose(Y)
#     X = np.insert(X, 2, values=0, axis=0)
#     Y = np.insert(Y, 2, values=0, axis=0)
#     vectorX = o3d.utility.Vector3dVector(np.transpose(X))
#     vectorY = o3d.utility.Vector3dVector(np.transpose(Y))
#     source = o3d.geometry.PointCloud(vectorX)
#     target = o3d.geometry.PointCloud(vectorY)
#     threshold = 200
#     trans_init = np.asarray(
#         [
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0.0, 0.0, 0.0, 1.0],
#         ]
#     )
#     # print('registering')
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source,
#         target,
#         threshold,
#         trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     )
#     return reg_p2p.transformation


# def find_rot_cpd(X, Y):
#     # Assuming X and Y are 2D arrays of shape (n_points, dimensions),
#     # and we're adding a z-axis with zeros to convert them into 3D for compatibility with CPD.
#     X = np.insert(X, 2, values=0, axis=1)  # Inserting a Z-axis with zero values
#     Y = np.insert(Y, 2, values=0, axis=1)  # Inserting a Z-axis with zero values
#
#     # Convert numpy arrays to point clouds.
#     # In probreg, point clouds can be represented directly as numpy arrays.
#     source = X
#     target = Y
#
#     # Perform Coherent Point Drift registration.
#     # cpd returns an object that contains the transformation model among other details.
#     # Using the 'rigid' transformation model for rotation and translation.
#     cpd = pr.cpd.registration_cpd(source, target, tf_type_name='rigid')
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = cpd.transformation.rot  # Rotation
#     transformation_matrix[:3, 3] = cpd.transformation.t  # Translation
#     return transformation_matrix


def dist_edge(edge, pos, t):
    pos = pos.astype(float)

    dists = np.linalg.norm(np.array(edge.pixel_list(t)) - pos, axis=1)

    return np.min(dists)


def interpolate_points(start, end, step=1):
    """Interpolate points between start and end with a given step."""
    # Vector from start to end
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    direction_normalized = direction / length

    # Number of points to generate (excluding the end point)
    num_points = int(length // step)

    # Generate points
    points = [
        np.array(start) + direction_normalized * step * i for i in range(num_points + 1)
    ]

    return points


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def average_min_distance_to_set(A, B_set):
    """Calculate the average minimum distance from each point in A to the closest point in a B set."""
    total_distance = 0
    for a in A:
        min_distance = min([euclidean_distance(a, b) for b in B_set])
        total_distance += min_distance
    return total_distance / len(A)


def average_min_distance_to_set_fast(A, B_set):
    # Expand dimensions of A and B_set to enable broadcasting
    A_expanded = np.expand_dims(A, axis=1)  # Shape becomes (len(A), 1, dim)
    B_set_expanded = np.expand_dims(B_set, axis=0)  # Shape becomes (1, len(B_set), dim)

    # Calculate squared Euclidean distances (avoiding square root for efficiency)
    # as the square root is a monotonic transformation and doesn't affect argmin.
    distances_squared = np.sum((A_expanded - B_set_expanded) ** 2, axis=2)

    # Find the minimum squared distance for each point in A
    min_distances_squared = np.min(distances_squared, axis=1)

    # Compute the average of the square root of these minimum distances
    average_min_distance = np.mean(min_distances_squared)

    return average_min_distance


# Assuming the objective_function is defined elsewhere in your code
def objective_function(params, source, target):
    # Convert params (first 3 are translation, next 4 are quaternion for rotation) to R and t
    t = params[:2]
    theta = params[2]
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Transform the source points
    transformed_source = np.dot(rotation, source.T).T + t

    # Compute distances to the nearest target points
    dist = average_min_distance_to_set_fast(transformed_source, target)

    # Return the average of these distances
    # print(dist)
    return dist


def callback_function(xk, source, target):
    # xk contains the current parameter values
    # Optionally, evaluate the current objective function value (if needed)
    current_value = objective_function(xk, source, target)
    print(f"Current parameters: {xk}, Objective Function Value: {current_value}")


def find_optimal_R_and_t(source, target, theta_divide=16, delta_space=7000):
    initial_guess = np.array([0, 0, 0])
    deltas = np.array([delta_space, delta_space, np.pi / theta_divide])
    init_params = [-delta_space / 2, -delta_space / 2, -np.pi / (2 * theta_divide)]

    simplex = [init_params]
    for i in range(len(init_params)):
        new_point = np.copy(init_params)
        new_point[i] += deltas[i]
        simplex.append(new_point)

    # Define a wrapper for the callback to include additional arguments
    def callback_with_args(xk):
        callback_function(xk, source, target)

    # Run the optimization
    result = minimize(
        objective_function,
        initial_guess,
        args=(source, target),
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "fatol": 0.1},
        # callback = callback_with_args
    )

    # Extract optimized parameters
    optimized_params = result.x
    t_optimized = optimized_params[
        :2
    ]  # Assuming this was meant to be [:2] for x and y translation
    theta = optimized_params[2]
    R_optimized = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    return R_optimized, t_optimized


def transform(pos, R, t):
    return R @ pos + t


def find_index_min(A, B):
    avg_distances = [average_min_distance_to_set_fast(A, B_set) for B_set in B]
    return (np.argmin(avg_distances), np.min(avg_distances))


def find_mapping(transport_edge_segment, network_edge_names, network_edge_segments):
    index, avg_distances = find_index_min(transport_edge_segment, network_edge_segments)
    return (network_edge_names[index], avg_distances)


def get_shifts(vid_obj):
    shiftx = vid_obj.img_dim[0] * vid_obj.space_res / 1.725 / 2
    shifty = vid_obj.img_dim[1] * vid_obj.space_res / 1.725 / 2
    return (shiftx, shifty)


def get_position(vid_obj):
    return np.array(vid_obj.dataset[["xpos_network", "ypos_network"]])


def get_close_edges(vid_obj, exp, t, R, trans):
    positions = get_position(vid_obj)
    edges = get_all_edges(exp, t)
    edges = [
        edge
        for edge in edges
        if dist_edge(edge, transform(positions, R, trans), t) <= 100
    ]
    return edges


def make_whole_mapping(
    vid_obj, exp, t, dist=100, R=np.array([[1, 0], [0, 1]]), trans=0, thresh=20
):
    shiftx, shifty = get_shifts(vid_obj)
    Rfound, tfound = register_rot_trans(
        vid_obj, exp, t, dist=dist, R=R, trans=trans, thresh=thresh
    )
    Rcurrent = Rfound @ R
    tcurrent = Rfound @ trans + tfound
    segments_final = get_segments_ends(
        vid_obj, shiftx, shifty, thresh, Rcurrent, tcurrent
    )
    edge_names = [edge.edge_name for edge in vid_obj.edge_objs]
    segments_final_interp = []
    for begin, end in segments_final:
        # Include the start point, interpolated points, and the end point
        interpolated_points = interpolate_points(begin, end)
        segments_final_interp.append(interpolated_points)
    edges = get_close_edges(vid_obj, exp, t, Rcurrent, tcurrent)
    # print(Rcurrent[0],edges,"in make whole mapping")
    network_edge_segments = [edge.pixel_list(t) for edge in edges]
    network_edge_names = edges
    mapping = {}
    avg_distances = []
    for transport_edge_name, transport_edge_segment in zip(
        edge_names, segments_final_interp
    ):
        mapping[transport_edge_name], avg_distance = find_mapping(
            transport_edge_segment, network_edge_names, network_edge_segments
        )
        avg_distances.append(avg_distance)
        # print(avg_distances)
    return (mapping, np.mean(avg_distance), Rfound, tfound)


def add_attribute(
    edge_data_csv, vid_edge_obj, network_edge_attribute, name_new_col, mapping
):
    # try:
    new_attribute = network_edge_attribute(mapping[vid_edge_obj.edge_name])
    # except KeyError:
    #     print(name_new_col,"edge not in network",mapping[vid_edge_obj.edge_name])
    #     new_attribute = None
    edge_data_csv.loc[
        edge_data_csv["edge_name"] == vid_edge_obj.edge_name, name_new_col
    ] = new_attribute


def check_hasedges(vid_obj):
    shiftx, shifty = get_shifts(vid_obj)
    segments = get_segments_ends(vid_obj, shiftx, shifty)
    return len(segments) > 0


def initialize_transformation():
    return np.array([[1, 0], [0, 1]]), np.array([0, 0])


def update_transformation(Rcurrent, tcurrent, Rfound, tfound):
    Rcurrent = Rfound @ Rcurrent
    tcurrent = Rfound @ tcurrent + tfound
    return Rcurrent, tcurrent


def should_reset(Rfound):
    return np.linalg.det(Rfound) <= 0 or Rfound[0][0] <= 0.99


def attempt_mapping(vid_obj, exp, t, Rcurrent, tcurrent, thresh=20):
    try:
        mapping, dist, Rfound, tfound = make_whole_mapping(
            vid_obj, exp, t, dist=100, R=Rcurrent, trans=tcurrent, thresh=thresh
        )
        reinitialize = False
    except (ValueError, IndexError) as e:
        try:
            Rcurrent, tcurrent = initialize_transformation()
            mapping, dist, Rfound, tfound = make_whole_mapping(
                vid_obj, exp, t, dist=100, R=Rcurrent, trans=tcurrent, thresh=thresh
            )
            reinitialize = True
        except (ValueError, IndexError) as e:
            Rcurrent, tcurrent = initialize_transformation()
            return ({}, -1, Rcurrent, tcurrent, True)
    return mapping, dist, Rfound, tfound, reinitialize


def process_video_object(vid_obj, exp, t, Rcurrent, tcurrent):
    mapping, dist, Rfound, tfound, reinitialize = attempt_mapping(
        vid_obj, exp, t, Rcurrent, tcurrent
    )
    if np.linalg.det(Rfound) < 0 or Rfound[0][0] <= 0.99:
        Rcurrent, tcurrent = initialize_transformation()
        mapping, dist, Rfound, tfound, reinitialize = attempt_mapping(
            vid_obj, exp, t, Rcurrent, tcurrent
        )
    if reinitialize:
        Rcurrent, tcurrent = initialize_transformation()
    Rcurrent, tcurrent = update_transformation(Rcurrent, tcurrent, Rfound, tfound)
    if dist > 20:
        mapping, dist, Rfound, tfound, reinitialize = attempt_mapping(
            vid_obj, exp, t, Rcurrent, tcurrent
        )
        if reinitialize:
            Rcurrent, tcurrent = initialize_transformation()
        Rcurrent, tcurrent = update_transformation(Rcurrent, tcurrent, Rfound, tfound)

    return Rcurrent, tcurrent, mapping, dist


def process_video_object_new(vid_obj, exp, t, Rcurrent, tcurrent, thresh=20):
    mapping, dist, Rfound, tfound, reinitialize = attempt_mapping(
        vid_obj, exp, t, Rcurrent, tcurrent, thresh
    )
    Rcurrent, tcurrent = update_transformation(Rcurrent, tcurrent, Rfound, tfound)
    return Rcurrent, tcurrent, mapping, dist


def register_dataset(data_obj, exp, t):
    Rcurrent, tcurrent = initialize_transformation()

    for index, vid_obj in enumerate(data_obj.video_objs):
        if check_hasedges(vid_obj) and vid_obj.dataset["magnification"] != 4:
            shiftx, shifty = get_shifts(vid_obj)
            positions = get_position(vid_obj)
            Rcurrent, tcurrent = initialize_transformation()

            Rcurrent, tcurrent, mapping, dist = process_video_object(
                vid_obj, exp, t, Rcurrent, tcurrent
            )
            if len(mapping) > 0:
                edges = get_close_edges(vid_obj, exp, t, Rcurrent, tcurrent)
                print(
                    vid_obj.dataset["video_int"],
                    "R=",
                    Rcurrent[0],
                    "\n dist=",
                    dist,
                    edges,
                )

                positions = Rcurrent @ positions + tcurrent

                segments_final = get_segments_ends(
                    vid_obj, shiftx, shifty, 0, Rcurrent, tcurrent
                )
                aligned_bools = []
                for edge, segment_final in zip(vid_obj.edge_objs, segments_final):
                    aligned_bools.append(
                        is_aligned(
                            edge.edge_name, segment_final, mapping, edges, positions, t
                        )
                    )
                update_edge_attributes(vid_obj, mapping, dist, aligned_bools, Rcurrent)


def update_edge_attributes(vid_obj, mapping, dist, aligned_bools, Rcurrent):
    edge_data_csv = pd.read_csv(vid_obj.edge_adr)
    for edge, aligned in zip(vid_obj.edge_objs, aligned_bools):
        if aligned:
            add_attribute(
                edge_data_csv, edge, lambda edge: edge.end.label, "network_end", mapping
            )
            add_attribute(
                edge_data_csv,
                edge,
                lambda edge: edge.begin.label,
                "network_begin",
                mapping,
            )
        else:
            add_attribute(
                edge_data_csv,
                edge,
                lambda edge: edge.begin.label,
                "network_end",
                mapping,
            )
            add_attribute(
                edge_data_csv,
                edge,
                lambda edge: edge.end.label,
                "network_begin",
                mapping,
            )
        add_attribute(
            edge_data_csv, edge, lambda edge: dist, "mapping_quality", mapping
        )
        add_attribute(
            edge_data_csv, edge, lambda edge: Rcurrent[0][0], "rotation", mapping
        )

    # print(edge_data_csv["network_begin"])
    edge_data_csv.to_csv(vid_obj.edge_adr, index=False)


def get_network_edge_segment_straight(edges, positions, t, dist=100):
    network_edge_segments = [edge.pixel_list(t) for edge in edges]
    network_edge_segments = [
        [pixel for pixel in pixels if np.linalg.norm(pixel - positions) <= 1.5 * dist]
        for pixels in network_edge_segments
    ]
    network_edge_segments = [
        [pixels[0], pixels[-1]] for pixels in network_edge_segments
    ]
    return network_edge_segments


def is_aligned(
    transport_edge_name, transport_edge_segment, mapping, edges, positions, t
):
    network_edge_segments = get_network_edge_segment_straight(
        edges, positions, t, dist=100
    )
    index = edges.index(mapping[transport_edge_name])
    segment_network = np.array(network_edge_segments[index])
    vector_network = segment_network[0] - segment_network[1]
    segment_video = np.array(transport_edge_segment)
    vector_video = segment_video[0] - segment_video[1]
    return np.dot(vector_network, vector_video) > 0
