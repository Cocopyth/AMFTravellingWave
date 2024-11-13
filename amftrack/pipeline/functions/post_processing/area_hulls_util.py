import pickle

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial, stats
from shapely.geometry import Polygon, Point

from amftrack.pipeline.functions.post_processing.util import is_in_study_zone
from amftrack.util.sys import temp_path
import os
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split


def get_time(exp, t, tp1):
    seconds = (exp.dates[tp1] - exp.dates[t]).total_seconds()
    return seconds / 3600

def splitPolygon(polygon, dx, dy):
    minx, miny, maxx, maxy = polygon.bounds
    nx = int((maxx - minx) / dx) + 1
    ny = int((maxy - miny) / dy) + 1
    horizontal_splitters = [
        LineString([(minx, miny + i * dy), (maxx, miny + i * dy)]) for i in range(ny)
    ]
    vertical_splitters = [
        LineString([(minx + i * dx, miny), (minx + i * dx, maxy)]) for i in range(nx)
    ]
    splitters = horizontal_splitters + vertical_splitters
    result = polygon

    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))

    return result

def get_length_shape(exp, shape, t):
    intersect = exp.multipoints[t].loc[exp.multipoints[t].within(shape)]
    tot_length = len(intersect) * 1.725
    return tot_length


def get_hulls(exp, ts):
    hulls = []
    for t in ts:
        nx_graph = exp.nx_graph[t]
        threshold = 0.1
        S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
        selected = [
            g
            for g in S
            if g.size(weight="weight") * len(g.nodes) / 10**6 >= threshold
        ]
        polys = Polygon()
        if len(selected) >= 0:
            area_max = 0
            for g in selected:
                nodes = np.array(
                    [
                        node.pos(t)
                        for node in exp.nodes
                        if node.is_in(t)
                        and np.all(is_in_study_zone(node, t, 1000, 150))
                        and (node.label in g.nodes)
                    ]
                )

                if len(nodes) > 3:
                    hull = spatial.ConvexHull(nodes)
                    poly = Polygon([nodes[vertice] for vertice in hull.vertices])
                    area_hull = poly.area * 1.725**2 / (1000**2)
                    polys = polys.union(poly)
        print(t, len(selected), polys.area)
        hulls.append(polys)
    return hulls


def ring_area(hull1, hull2):
    return np.sum(hull2.difference(hull1).area) * 1.725**2 / (1000**2)


def get_nodes_in_shape(shape, t, exp):
    nodes = [
        node
        for node in exp.nodes
        if node.is_in(t)
        and shape.contains(Point(node.pos(t)))
        and np.all(is_in_study_zone(node, t, 1000, 200))
    ]
    return nodes


def get_nodes_in_ring(hull1, hull2, t, exp):
    nodes = [
        node
        for node in exp.nodes
        if node.is_in(t)
        and hull2.contains(Point(node.pos(t)))
        and not hull1.contains(Point(node.pos(t)))
        and np.all(is_in_study_zone(node, t, 1000, 200))
    ]
    return nodes


def get_hyphae_in_ring(hull1, hull2, t, exp):
    hyphae = [
        hyph
        for hyph in exp.hyphaes
        if hyph.end.is_in(t)
        and hull2.contains(Point(hyph.end.pos(t)))
        and not hull1.contains(Point(hyph.end.pos(t)))
        and np.all(is_in_study_zone(hyph.end, t, 1000, 200))
    ]
    return hyphae


def get_length_in_ring(hull1, hull2, t, exp):
    nodes = get_nodes_in_ring(hull1, hull2, t, exp)
    edges = {edge for node in nodes for edge in node.edges(t)}
    tot_length = np.sum(
        [
            np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t)) * 1.725 / 2
            for edge in edges
        ]
    )
    return tot_length


def get_length_in_ring_new(hull1, hull2, t, exp):
    shape = hull2.difference(hull1)
    tot_length = get_length_shape(exp, shape, t)
    print(tot_length)
    return tot_length

def get_length_fast(edge, t):
    return np.linalg.norm(edge.begin.pos(t) - edge.end.pos(t))

f = lambda edge: np.log(
    edge.width(edge.ts()[-1])
    * get_length_fast(edge, edge.ts()[-1])
    * edge.end.degree(edge.ts()[-1])
    * edge.begin.degree(edge.ts()[-1])
)


f = lambda edge: np.log(
    edge.width(edge.ts()[-1])
    * get_length_fast(edge, edge.ts()[-1])
    * edge.end.degree(edge.ts()[-1])
    * edge.begin.degree(edge.ts()[-1])
)


def get_BAS_length_in_ring(hull1, hull2, t, exp):
    nodes = get_nodes_in_ring(hull1, hull2, t, exp)
    edges = {edge for node in nodes for edge in node.edges(t) if f(edge) <= 10}
    tot_length = np.sum(
        [
            np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t)) * 1.725 / 2
            for edge in edges
        ]
    )
    return tot_length

def get_density_in_ring_bootstrap(hull1, hull2, t, exp, n_resamples=100):
    shape = hull2.difference(hull1)
    geoms = splitPolygon(shape, 100, 100).geoms if shape.area > 0 else []
    densities = [
        get_length_shape(exp, geom, t) / (geom.area * 1.725**2 / (1000**2))
        for geom in geoms
    ]
    if len(densities) > 0:
        res = stats.bootstrap(
            (np.array(densities),),
            np.mean,
            vectorized=True,
            method="basic",
            n_resamples=n_resamples,
        )
        return res

    else:
        return None


def get_tip_in_ring_bootstrap(hull1, hull2, t, exp, rh_only, max_t, n_resamples=100):
    """Returns the bootsrap of the mean density of hyphae in the ring"""
    shape = hull2.difference(hull1)
    geoms = splitPolygon(shape, 100, 100).geoms if shape.area > 0 else []
    densities = [
        len(get_growing_tips_shape(geom, t, exp, rh_only, max_t))
        / (geom.area * 1.725**2 / (1000**2))
        for geom in geoms
    ]
    if len(densities) > 0:
        res = stats.bootstrap(
            (np.array(densities),),
            np.mean,
            vectorized=True,
            method="basic",
            n_resamples=n_resamples,
        )
        return res

    else:
        return None


def get_biovolume_in_ring(hull1, hull2, t, exp):
    """Returns the total biovolume of hyphae in the ring"""
    nodes = get_nodes_in_ring(hull1, hull2, t, exp)
    edges = {edge for node in nodes for edge in node.edges(t)}
    tot_biovolume = np.sum(
        [
            np.pi
            * (edge.width(t) / 2) ** 2
            * np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t))
            * 1.725
            for edge in edges
        ]
    )
    return tot_biovolume


def get_growing_tips_shape(shape, t, exp, rh_only, max_t=np.inf):
    nodes = get_nodes_in_shape(shape, t, exp)
    tips = [
        node
        for node in nodes
        if node.degree(t) == 1 and node.is_in(t + 1) and len(node.ts()) > 2
    ]
    tips = [tip for tip in tips if np.all(is_in_study_zone(tip, t, 1000, 150, False))]
    growing_tips = []
    for tip in tips:
        timesteps = [tim for tim in tip.ts() if tim <= max_t]
        tim = timesteps[-1] if len(timesteps) > 0 else tip.ts()[-1]
        if np.linalg.norm(tip.pos(tim) - tip.pos(t)) >= 40:
            growing_tips.append(tip)
    if rh_only:
        growing_rhs = [
            node
            for node in growing_tips
            if np.linalg.norm(node.pos(node.ts()[0]) - node.pos(node.ts()[-1])) >= 1500
        ]
        return growing_rhs
    else:
        return growing_tips


def get_growing_tips(hull1, hull2, t, exp, rh_only, max_t=np.inf):
    nodes = get_nodes_in_ring(hull1, hull2, t, exp)
    tips = [
        node
        for node in nodes
        if node.degree(t) == 1 and node.is_in(t + 1) and len(node.ts()) > 2
    ]
    tips = [tip for tip in tips if np.all(is_in_study_zone(tip, t, 1000, 150, False))]
    growing_tips = []
    for tip in tips:
        timesteps = [tim for tim in tip.ts() if tim <= max_t]
        tim = timesteps[-1] if len(timesteps) > 0 else tip.ts()[-1]
        if np.linalg.norm(tip.pos(tim) - tip.pos(t)) >= 40:
            growing_tips.append(tip)
    if rh_only:
        growing_rhs = [
            node
            for node in growing_tips
            if np.linalg.norm(node.pos(node.ts()[0]) - node.pos(node.ts()[-1])) >= 1500
        ]
        return growing_rhs
    else:
        return growing_tips


def get_rate_anas_in_ring(hull1, hull2, t, exp, rh_only):
    growing_tips = get_growing_tips(hull1, hull2, t, exp, rh_only)
    anas_tips = [
        tip
        for tip in growing_tips
        if tip.degree(t) == 1
        and tip.degree(t + 1) >= 3
        and 1 not in [tip.degree(t) for t in [tau for tau in tip.ts() if tau > t]]
    ]
    timedelta = get_time(exp, t, t + 1)
    return len(anas_tips) / timedelta


def get_rate_branch_in_ring(hull1, hull2, t, exp, rh_only, max_t=np.inf):
    growing_tips = get_growing_tips(hull1, hull2, t, exp, rh_only, max_t)
    new_tips = [tip for tip in growing_tips if tip.ts()[0] == t]
    timedelta = get_time(exp, t, t + 1)
    return len(new_tips) / timedelta


def get_rate_stop_in_ring(hull1, hull2, t, exp, rh_only, max_t=np.inf):
    growing_tips = get_growing_tips(hull1, hull2, t, exp, rh_only, max_t)
    interest_tips = [
        tip for tip in growing_tips if tip.ts()[-1] != t + 1 and tip.degree(t + 1) == 1
    ]
    stop_tips = []
    for tip in interest_tips:
        timesteps = [tim for tim in tip.ts() if tim <= max_t]
        tim = timesteps[-1] if len(timesteps) > 0 else tip.ts()[-1]
        if np.linalg.norm(tip.pos(tim) - tip.pos(t + 1)) <= 40:
            stop_tips.append(tip)
    timedelta = get_time(exp, t, t + 1)
    return len(stop_tips) / timedelta

def get_rate_lost_track_in_ring(hull1, hull2, t, exp, rh_only, max_t=np.inf):
    growing_tips = get_growing_tips(hull1, hull2, t, exp, rh_only, max_t)
    interest_tips = [
        tip for tip in growing_tips if tip.ts()[-1] != t + 1 and tip.degree(t + 1) == 1
    ]
    stop_tips = []
    for tip in interest_tips:
        timesteps = [tim for tim in tip.ts() if tim <= max_t]
        tim = timesteps[-1] if len(timesteps) > 0 else tip.ts()[-1]
        if np.linalg.norm(tip.pos(tim) - tip.pos(t + 1)) <= 40:
            stop_tips.append(tip)
    timedelta = get_time(exp, t, t + 1)
    return len(stop_tips) / timedelta


def get_num_active_tips_in_ring(hull1, hull2, t, exp, rh_only, max_t=np.inf):
    growing_tips = get_growing_tips(hull1, hull2, t, exp, rh_only, max_t)
    return len(growing_tips)


def get_regular_hulls(exp, ts, incrL):
    path = os.path.join(
        temp_path,
        f"hullsreg_{exp.unique_id}_{incrL}_"
        f"{np.sum(pd.util.hash_pandas_object(exp.folders.iloc[ts]))}.pick",
    )
    if os.path.isfile(path):
        (regular_hulls, indexes) = pickle.load(open(path, "rb"))
    else:
        hulls = get_hulls(exp, ts)
        areas = [hull.area * 1.725**2 / (1000**2) for hull in hulls]
        area_incr = areas[-1] - areas[0]
        regular_hulls = [hulls[0]]
        init_area = areas[0]
        indexes = [0]
        current_area = init_area
        while current_area <= areas[-1]:
            index = min([i for i in range(len(areas)) if areas[i] >= current_area])
            indexes.append(index)
            current_area = (np.sqrt(current_area) + incrL) ** 2
            regular_hulls.append(hulls[index])
        pickle.dump((regular_hulls, indexes), open(path, "wb"))

    return (regular_hulls, indexes)


def get_regular_hulls_area_fixed(exp, ts, incr):
    """Gets the hulls of an experiment with a fixed area increase incr"""
    path = os.path.join(
        temp_path,
        f"hulls_{exp.unique_id}_{incr}_"
        f"{np.sum(pd.util.hash_pandas_object(exp.folders.iloc[ts]))}.pick",
    )
    if os.path.isfile(path):
        (regular_hulls, indexes) = pickle.load(open(path, "rb"))
    else:
        hulls = get_hulls(exp, ts)
        areas = [np.sum(hull.area) * 1.725**2 / (1000**2) for hull in hulls]
        area_incr = np.max(areas) - areas[0]
        num = int(area_incr / incr)
        regular_hulls = [hulls[0]]
        init_area = areas[0]
        indexes = [0]
        current_area = init_area
        for i in range(num - 1):
            current_area += incr
            index = min([i for i in range(len(areas)) if areas[i] >= current_area])
            indexes.append(index)
            regular_hulls.append(hulls[index])

        pickle.dump((regular_hulls, indexes), open(path, "wb"))
    return (regular_hulls, indexes)
