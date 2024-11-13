import os
import unittest
from test.util import helper

from amftrack.pipeline.launching.run import run_function


from amftrack.pipeline.scripts.image_processing import (
    mask_skel,
    extract_skel_2,
    detect_blob,
    prune_skel,
    extract_nx_graph,
    track_nodes,
    make_labeled_graphs,
    extract_skel_no_external,
    final_alignment_new,
    realign_new,
)

path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/')

class TestImageAnalysis(unittest.TestCase):
    """Tests that need only a static plate with one timestep"""

    @classmethod
    def setUpClass(cls):
        cls.folders = helper.make_folders(path_data)
        cls.directory = path_data

    def test_create_script(self):
        helper.create_script_function("extract_skel_2.py")

    def test_skeletonize(self):
        hyph_width = 30
        perc_low = 85
        perc_high = 99.5
        minlow = 10
        minhigh = 70

        args = [None, hyph_width, perc_low, perc_high, minlow, minhigh, self.directory]
        run_function(extract_skel_2.process, args, self.folders)

    def test_skeletonize_no_exeternal(self):
        hyph_width = 30
        perc_low = 85
        perc_high = 99.5
        minlow = 10
        minhigh = 70

        args = [None, hyph_width, perc_low, perc_high, minlow, minhigh, self.directory]
        run_function(extract_skel_no_external.process, args, self.folders)

    def test_spore(self):
        args = [None, self.directory]
        run_function(detect_blob.process, args, self.folders)

    def test_mask(self):
        thresh = 40
        args = [None, thresh, self.directory]
        run_function(mask_skel.process, args, self.folders)

    def test_prune(self):
        threshold = 0.01 / 1.725
        skip = False
        args = [None, threshold, skip, self.directory]
        run_function(prune_skel.process, args, self.folders[:1])

    def test_realign(self):
        args = [None, self.directory]
        run_function(
            final_alignment_new.process, args, self.folders[:4], sequential_process=True
        )

    def test_create_realign(self):
        args = [None, self.directory]
        run_function(realign_new.process, args, self.folders[:4], per_unique_id=True)

    def test_graph_extract(self):
        args = [None, self.directory]
        run_function(extract_nx_graph.process, args, self.folders[:1])

    def test_node_id(self):
        args = [None, self.directory]
        run_function(track_nodes.process, args, self.folders)

    def test_make_labeled(self):
        args = [None, self.directory]
        run_function(
            make_labeled_graphs.process, args, self.folders, per_unique_id=True
        )
