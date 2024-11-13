import os
import unittest
from test.util import helper
import matplotlib.pyplot as plt
import amftrack.pipeline.functions.post_processing.exp_plot as exp_plot
import amftrack.pipeline.functions.post_processing.time_plate as time_plate
import amftrack.pipeline.functions.post_processing.time_hypha as time_hypha
import amftrack.pipeline.functions.post_processing.time_edge as time_edge
import amftrack.pipeline.functions.post_processing.area_hulls as area_hulls
import geopandas as gpd
from shapely.geometry import Point
from random import choice
import matplotlib as mpl
import json
from amftrack.util.sys import test_path
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    load_graphs,
    load_skel,
)
import numpy as np

mpl.use("AGG")

path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/')

class TestExperiment(unittest.TestCase):
    """Tests that need only a static plate with one timestep"""

    @classmethod
    def setUpClass(cls):
        cls.exp = helper.make_experiment_object_analysis(path_data)

    def tearDown(self):
        "Runs after each test"
        plt.close("all")

    # def test_exp_plot_f(self):
    #     fs = dir(exp_plot)
    #     plot_fs = [f for f in fs if f.split("_")[0] == "plot"]
    #     for f in plot_fs:
    #         getattr(exp_plot, f)(self.exp)
    #This will only run properly if dropbox is setup

    def test_time_plate_f(self):
        fs = dir(time_plate)
        plot_fs = [f for f in fs if f.split("_")[0] == "get"]
        for f in plot_fs:
            print(f, getattr(time_plate, f)(self.exp, 0))

    def test_area_hulls_f(self):
        t = 2
        load_skel(self.exp, [t])

        skeletons = []
        for skeleton in self.exp.skeletons:
            if skeleton is None:
                skeletons.append({})
            else:
                skeletons.append(skeleton)
        self.exp.multipoints = [
            gpd.GeoSeries([Point(pixel) for pixel in skeleton.keys()])
            for skeleton in skeletons
        ]

        fs = [
            area_hulls.get_std_density_in_ring_new_bootstrap,
            area_hulls.get_density_in_ring_new,
            area_hulls.get_density_in_ring_new_bootstrap,
        ]
        args = {"incr": 10, "i": 0}
        for f in fs:
            print(f, f(self.exp, t, args))

    def test_branching_hulls(self):
        """Tests the function get_density_branch_rate_in_ring"""
        t = 2
        load_skel(self.exp, [t])
        skeletons = []
        for skeleton in self.exp.skeletons:
            if skeleton is None:
                skeletons.append({})
            else:
                skeletons.append(skeleton)
        self.exp.multipoints = [
            gpd.GeoSeries([Point(pixel) for pixel in skeleton.keys()])
            for skeleton in skeletons
        ]
        fs = [
            # area_hulls.get_density_branch_rate_in_ring,
            area_hulls.get_std_tip_in_ring_new_bootstrap
        ]
        args = {"incr": 10, "i": 0, "rh_only": False, "max_t": 99}
        for f in fs:
            print(f, f(self.exp, t, args))

    def test_time_hypha_f(self):
        fs = dir(time_hypha)
        plot_fs = [f for f in fs if f.split("_")[0] == "get"]
        hypha = choice(self.exp.hyphaes)
        data_hypha = {}
        for f in plot_fs:
            column, result = getattr(time_hypha, f)(hypha, 0, 1)
        data_hypha[column] = result
        path_hyph_info = os.path.join(test_path, "time_hypha.json")
        with open(path_hyph_info, "w") as jsonf:
            json.dump(data_hypha, jsonf, indent=4)

    def test_time_edge_f(self):
        fs = dir(time_edge)
        plot_fs = [f for f in fs if f.split("_")[0] == "get"]
        edge = choice(list(self.exp.nx_graph[0].edges))
        edge = Edge(
            Node(np.min(edge), self.exp), Node(np.max(edge), self.exp), self.exp
        )
        data_hypha = {}
        for f in plot_fs:
            column, result = getattr(time_edge, f)(edge, 0)
        data_hypha[column] = result
        path_hyph_info = os.path.join(test_path, "time_edge.json")
        with open(path_hyph_info, "w") as jsonf:
            json.dump(data_hypha, jsonf, indent=4)
