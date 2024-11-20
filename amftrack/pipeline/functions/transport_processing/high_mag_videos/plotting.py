from amftrack.pipeline.functions.image_processing.experiment_util import *
from PIL import Image


def plot_full_video(
    exp: Experiment,
    t: int,
    region=None,
    edges: List[Edge] = [],
    points: List[coord_int] = [],
    video_num: List[int] = [],
    segments: List[List[coord_int]] = [],
    nodes: List[Node] = [],
    downsizing=5,
    dilation=1,
    save_path="",
    prettify=False,
    with_point_label=False,
    figsize=(6, 3),
    dpi=None,
    node_size=5,
    arrows=None,
) -> None:
    """
    This is the general purpose function to plot the full image or a region `region` of the image at
    any given timestep t. The region being specified in the GENERAL coordinates.
    The image can be downsized by a chosen factor `downsized` with additionnal features such as: edges, nodes, points, segments.
    The coordinates for all the objects are provided in the GENERAL referential.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param edges: list of edges to plot, it is the pixel list that is plotted, not a straight line
    :param nodes: list of nodes to plot (only nodes in the `region` will be shown)
    :param points: points such as [123, 234] to plot with a red cross on the image
    :param segments: plot lines between two points that are provided
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param prettify: if True, the image will be enhanced by smoothing the intersections between images
    :param with_point_label: if True, the index of the point is ploted on top of it

    NB: the full region of a full image is typically [[0, 0], [26000, 52000]]
    NB: the interesting region of a full image is typically [[12000, 15000], [26000, 35000]]
    NB: the colors are chosen randomly for edges
    NB: giving a smaller region greatly increase computation time
    """

    # TODO(FK): fetch image size from experiment object here, and use it in reconstruct image
    # TODO(FK): colors for edges are not consistent
    # NB: possible other parameters that could be added: alpha between layers, colors for object, figure_size
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X  # TODO(FK): Shouldn't be hardcoded
        region[1][1] += DIM_Y

    # 1/ Image layer
    im, f = reconstruct_image_from_general(
        exp,
        t,
        downsizing=downsizing,
        region=region,
        prettify=prettify,
        white_background=False,  # TODO(FK): add image dimention here dimx = ..
    )
    f_int = lambda c: f(c).astype(int)
    new_region = [
        f_int(region[0]),
        f_int(region[1]),
    ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]

    # 2/ Edges layer
    skel_im, _ = reconstruct_skeletton(
        [edge.pixel_list(t) for edge in edges],
        region=region,
        color_seeds=[(edge.begin.label + edge.end.label) % 255 for edge in edges],
        downsizing=downsizing,
        dilation=dilation,
    )

    # 3/ Fusing layers
    fig = plt.figure(
        figsize=figsize
    )  # width: 30 cm height: 20 cm # TODO(FK): change dpi
    ax = fig.add_subplot(111)
    ax.imshow(im, cmap="gray", interpolation="none")
    ax.imshow(skel_im, alpha=1, interpolation="none")

    # 3/ Plotting the Nodes
    size = node_size
    for node in nodes:
        c = f(list(node.pos(t)))
        color = make_random_color(node.label)[:3]
        reciprocal_color = 255 - color
        color = tuple(color / 255)
        reciprocal_color = tuple(reciprocal_color / 255)
        bbox_props = dict(boxstyle="circle", fc=color, edgecolor="none")
        if is_in_bounding_box(c, new_region):
            node_text = ax.text(
                c[1],
                c[0],
                str(node.label),
                ha="center",
                va="center",
                bbox=bbox_props,
                font=fpath,
                fontdict={"color": reciprocal_color},
                size=size,
                # alpha = 0.5
            )
    # 4/ Plotting coordinates
    points = [f(c) for c in points]
    for i, c in enumerate(points):
        if is_in_bounding_box(c, new_region):
            color = make_random_color(video_num[i])[:3]
            color = tuple(color / 255)
            plt.text(c[1], c[0], video_num[i], color="black", fontsize=20, alpha=1)
            plt.plot(c[1], c[0], marker="x", color="black", markersize=10, alpha=0.5)

            if with_point_label:
                plt.text(c[1], c[0], f"{i}")

    # 5/ Plotting segments
    segments = [[f(segment[0]), f(segment[1])] for segment in segments]
    for s in segments:
        plt.plot(
            [s[0][1], s[1][1]],  # x1, x2
            [s[0][0], s[1][0]],  # y1, y2
            color="white",
            linewidth=2,
        )

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    return ax, f


def make_full_image(
    exp: Experiment,
    t: int,
    region=None,
    edges: List[Edge] = [],
    points: List[coord_int] = [],
    video_num: List[int] = [],
    segments: List[List[coord_int]] = [],
    nodes: List[Node] = [],
    downsizing=5,
    dilation=1,
    save_path="",
    prettify=False,
    with_point_label=False,
    figsize=(12, 8),
    dpi=None,
    node_size=5,
) -> None:
    """
    This is the general purpose function to plot the full image or a region `region` of the image at
    any given timestep t. The region being specified in the GENERAL coordinates.
    The image can be downsized by a chosen factor `downsized` with additionnal features such as: edges, nodes, points, segments.
    The coordinates for all the objects are provided in the GENERAL referential.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param edges: list of edges to plot, it is the pixel list that is plotted, not a straight line
    :param nodes: list of nodes to plot (only nodes in the `region` will be shown)
    :param points: points such as [123, 234] to plot with a red cross on the image
    :param segments: plot lines between two points that are provided
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param prettify: if True, the image will be enhanced by smoothing the intersections between images
    :param with_point_label: if True, the index of the point is ploted on top of it

    NB: the full region of a full image is typically [[0, 0], [26000, 52000]]
    NB: the interesting region of a full image is typically [[12000, 15000], [26000, 35000]]
    NB: the colors are chosen randomly for edges
    NB: giving a smaller region greatly increase computation time
    """

    # TODO(FK): fetch image size from experiment object here, and use it in reconstruct image
    # TODO(FK): colors for edges are not consistent
    # NB: possible other parameters that could be added: alpha between layers, colors for object, figure_size
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X  # TODO(FK): Shouldn't be hardcoded
        region[1][1] += DIM_Y

    # 1/ Image layer
    im, f = reconstruct_image_from_general(
        exp,
        t,
        downsizing=downsizing,
        region=region,
        prettify=prettify,
        white_background=False,  # TODO(FK): add image dimention here dimx = ..
    )
    f_int = lambda c: f(c).astype(int)
    new_region = [
        f_int(region[0]),
        f_int(region[1]),
    ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]

    # 2/ Edges layer
    skel_im, _ = reconstruct_skeletton(
        [edge.pixel_list(t) for edge in edges],
        region=region,
        color_seeds=[(edge.begin.label + edge.end.label) % 255 for edge in edges],
        downsizing=downsizing,
        dilation=dilation,
    )
    return (im, skel_im)


def plot_edge_color_value_2(
    exp: Experiment,
    t: int,
    color_fun: Callable,
    region=None,
    intervals=[[1, 4], [4, 6], [6, 10], [10, 20]],
    cmap=cm.get_cmap("Reds", 100),
    plot_cmap=False,
    v_max=10,
    v_min=0,
    nodes: List[Node] = [],
    downsizing=5,
    dilation=5,
    save_path="",
    color_seed=12,
    dpi=None,
    show_background=True,
    label_colorbar="Width ($\mu m)$",
    figsize=(36, 24),
    figax=None,
    alpha=0.5,
) -> None:
    """
    Plot the width for all the edges at a given timestep.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param color_fun: edge -> float a function of edges that needs to be color plotted

    :param nodes: list of nodes to plot
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param intervals: different width intervals that will be given different colors
    :param cmap: a colormap to map width to color
    :param plot_cmap: a boolean, whether or not to plot with cmap
    :param v_max: the max width for the colorbar/colormap
    """
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X
        region[1][1] += DIM_Y

    edges = get_all_edges(exp, t)
    if figax is None:
        fig = plt.figure(
            figsize=figsize
        )  # width: 30 cm height: 20 cm # TODO(FK): change dpi
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    # Give colors to edges
    default_color = 1000
    colors = []
    widths = []
    for edge in edges:
        width = color_fun(edge)
        widths.append(width)
        if not plot_cmap:
            color = default_color
            for i, interval in enumerate(intervals):
                if interval[0] <= width and width < interval[1]:
                    color = i + color_seed
            colors.append(color)
    if plot_cmap:
        colors = [cmap((width - v_min) / (v_max - v_min)) for width in widths]

    # 0/ Make color legend
    def convert(c):
        c_ = c / 255
        return (c_[0], c_[1], c_[2])

    if not plot_cmap:
        handles = []
        for i, interval in enumerate(intervals):
            handles.append(
                mpatches.Patch(
                    color=convert(make_random_color(i + color_seed)),
                    label=str(interval),
                )
            )
        handles.append(
            mpatches.Patch(color=convert(make_random_color(1000)), label="out of bound")
        )
        # fig.legend(handles=handles)
    else:
        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        N = 5
        # plt.colorbar(sm, ticks=np.linspace(v_min, v_max, N), label=label_colorbar,ax=ax)
    # 1/ Image layer
    im, f = reconstruct_image_from_general(
        exp,
        t,
        downsizing=downsizing,
        region=region,
        prettify=False,
        white_background=False,
    )
    f_int = lambda c: f(c).astype(int)

    # 2/ Edges layer
    color_list = (
        [(np.array(color) * 255).astype(int) for color in colors] if plot_cmap else None
    )
    from_edges = reconstruct_skeletton_from_edges(
        exp,
        t,
        edges=edges,
        region=region,
        color_seeds=colors,
        color_list=color_list,
        downsizing=downsizing,
        dilation=dilation,
        timestep=False,
    )
    skel_im, _ = from_edges
    if show_background:
        new_region = [
            f_int(region[0]),
            f_int(region[1]),
        ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]

    # 3/ Fusing layers
    if show_background:
        ax.imshow(im, cmap="gray", interpolation="none")
    ax.imshow(skel_im, alpha=alpha, interpolation="none")

    # 3/ Plotting the Nodes
    size = 5
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nodes:
        c = node.pos(t)
        if is_in_bounding_box(c, region):
            c = f(node.pos(t))
            node_text = ax.text(
                c[1],
                c[0],
                str(node.label),
                ha="center",
                va="center",
                size=size,
                bbox=bbox_props,
            )

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    return fig, ax, f


def plot_edge_color_value_3(
    exp: Experiment,
    t: int,
    color_fun: Callable,
    region=None,
    intervals=[[1, 4], [4, 6], [6, 10], [10, 20]],
    cmap=cm.get_cmap("Reds", 100),
    plot_cmap=False,
    v_max=10,
    v_min=0,
    nodes: List[Node] = [],
    downsizing=5,
    dilation=5,
    save_path="",
    color_seed=12,
    dpi=None,
    show_background=True,
    label_colorbar="Width ($\mu m)$",
    figsize=(36, 24),
    figax=None,
    alpha=0.5,
) -> None:
    """
    Plot the width for all the edges at a given timestep.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param color_fun: edge -> float a function of edges that needs to be color plotted

    :param nodes: list of nodes to plot
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param intervals: different width intervals that will be given different colors
    :param cmap: a colormap to map width to color
    :param plot_cmap: a boolean, whether or not to plot with cmap
    :param v_max: the max width for the colorbar/colormap
    """
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X
        region[1][1] += DIM_Y

    edges = get_all_edges(exp, t)
    if figax is None:
        fig = plt.figure(
            figsize=figsize
        )  # width: 30 cm height: 20 cm # TODO(FK): change dpi
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    # Give colors to edges
    default_color = 1000
    colors = []
    widths = []
    for edge in edges:
        width = color_fun(edge)
        widths.append(width)
        if not plot_cmap:
            color = default_color
            for i, interval in enumerate(intervals):
                if interval[0] <= width and width < interval[1]:
                    color = i + color_seed
            colors.append(color)
    if plot_cmap:
        colors = [cmap((width - v_min) / (v_max - v_min)) for width in widths]

    # 0/ Make color legend
    def convert(c):
        c_ = c / 255
        return (c_[0], c_[1], c_[2])

    if not plot_cmap:
        handles = []
        for i, interval in enumerate(intervals):
            handles.append(
                mpatches.Patch(
                    color=convert(make_random_color(i + color_seed)),
                    label=str(interval),
                )
            )
        handles.append(
            mpatches.Patch(color=convert(make_random_color(1000)), label="out of bound")
        )
        # fig.legend(handles=handles)
    else:
        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        N = 5
        # plt.colorbar(sm, ticks=np.linspace(v_min, v_max, N), label=label_colorbar,ax=ax)
    # 1/ Image layer
    f = lambda x: (
        (x[0] - region[0][0]) / downsizing,
        (x[1] - region[0][1]) / downsizing,
    )
    f_int = lambda c: f(c).astype(int)

    # 2/ Edges layer
    color_list = (
        [(np.array(color) * 255).astype(int) for color in colors] if plot_cmap else None
    )
    from_edges = reconstruct_skeletton_from_edges(
        exp,
        t,
        edges=edges,
        region=region,
        color_seeds=colors,
        color_list=color_list,
        downsizing=downsizing,
        dilation=dilation,
        timestep=False,
    )
    skel_im, _ = from_edges
    if show_background:
        new_region = [
            f_int(region[0]),
            f_int(region[1]),
        ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]
    ax.imshow(skel_im, alpha=alpha, interpolation="none")

    # 3/ Plotting the Nodes
    size = 5
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nodes:
        c = node.pos(t)
        if is_in_bounding_box(c, region):
            c = f(node.pos(t))
            node_text = ax.text(
                c[1],
                c[0],
                str(node.label),
                ha="center",
                va="center",
                size=size,
                bbox=bbox_props,
            )

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    return fig, ax, f
