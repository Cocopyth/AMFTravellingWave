from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,

)
from amftrack.pipeline.functions.post_processing.area_hulls_util import *

import numpy as np
import shapely.speedups


shapely.speedups.enable()


def get_density_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_length_in_ring(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_density_incr-{incr}_index-{i}", length / area)
    else:
        return (f"ring_density_incr-{incr}_index-{i}", None)


def get_density_in_ring_new(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_length_in_ring_new(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_density_incr-{incr}_index-{i}-new", length / area)
    else:
        return (f"ring_density_incr-{incr}_index-{i}-new", None)


def get_density_in_ring_new_fixed(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_length_in_ring_new(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_density_incr_fixex-{incr}_index-{i}-new", length / area)
    else:
        return (f"ring_density_incr_fixex-{incr}_index-{i}-new", None)


def get_density_in_ring_new_bootstrap(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        res = get_density_in_ring_bootstrap(hull1, hull2, t, exp, n_resamples=1)
        if res is None:
            return (f"ring_density_incr-{incr}_index-{i}-boot", None)
        else:
            return (
                f"ring_density_incr-{incr}_index-{i}-boot",
                np.median(res.bootstrap_distribution),
            )
    else:
        return (f"ring_density_incr-{incr}_index-{i}_boot", None)


def get_std_density_in_ring_new_bootstrap(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        res = get_density_in_ring_bootstrap(hull1, hull2, t, exp, n_resamples=100)
        if res is None:
            return (f"ring_density_incr-{incr}_index-{i}-bootstd", None)
        else:
            return (f"ring_density_incr-{incr}_index-{i}-bootstd", res.standard_error)
    else:
        return (f"ring_density_incr-{incr}_index-{i}-bootstd", None)


def get_std_tip_in_ring_new_bootstrap(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        res = get_tip_in_ring_bootstrap(
            hull1, hull2, t, exp, rh_only, max_t, n_resamples=100
        )
        if res is None:
            return (f"ring_active_tips_density_incr-{incr}_index-{i}-bootstd", None)
        else:
            return (
                f"ring_active_tips_density_incr-{incr}_index-{i}-bootstd",
                res.standard_error,
            )
    else:
        return (f"ring_active_tips_density_incr-{incr}_index-{i}-bootstd", None)


def get_biovolume_density_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        biomass = get_biovolume_in_ring(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_biovolume_density_incr-{incr}_index-{i}", biomass / area)
    else:
        return (f"ring_biovolume_density_incr-{incr}_index-{i}", None)


def get_density_BAS_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_BAS_length_in_ring(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_bas_density_incr-{incr}_index-{i}", length / area)
    else:
        return (f"ring_bas_density_incr-{incr}_index-{i}", None)


# def get_mean_speed_in_ring(exp, t, args):
#     incr = args["incr"]
#     i = args["i"]
#     rh_only = args["rh_only"]
#     op_id = args["op_id"]
#
#     regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
#     if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
#         hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
#         speed = get_speed_in_ring(hull1, hull2, t, exp, rh_only, op_id)
#         return (f"mean_speed_incr-{incr}_index-{i}", speed)
#     else:
#         return (f"mean_speed_incr-{incr}_index-{i}", None)


def get_density_anastomose_in_ring(exp: Experiment, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_anas_in_ring(hull1, hull2, t, exp, rh_only)
        area = ring_area(hull1, hull2)
        return (f"ring_anas_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_anas_density_incr-{incr}_index-{i}", None)


def get_density_branch_rate_in_ring(exp, t, args):
    """A function to get the density of branch rate in a ring.
    :param exp: an experiment object
    :param t: the time point
    :param args: a dictionary of arguments
    """
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_branch_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_branch_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_branch_density_incr-{incr}_index-{i}", None)


def get_density_stop_rate_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_stop_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_stop_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_stop_density_incr-{incr}_index-{i}", None)


def get_density_lost_track_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_stop_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_stop_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_stop_density_incr-{incr}_index-{i}", None)


def get_density_active_tips_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_num_active_tips_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_active_tips_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_active_tips_density_incr-{incr}_index-{i}", None)


def get_density_active_tips_in_ring_fixed(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    regular_hulls, indexes = get_regular_hulls(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_num_active_tips_in_ring(hull1, hull2, t, exp, rh_only)
        area = ring_area(hull1, hull2)
        return (f"ring_active_tips_density_reg_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_active_tips_density_reg_incr-{incr}_index-{i}", None)
