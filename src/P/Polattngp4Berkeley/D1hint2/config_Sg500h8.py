import re
from collections import OrderedDict

import numpy as np
from easydict import EasyDict


def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else 0


def getConfigFunc(P, D, S, R, **kwargs):
    config = EasyDict()
    config.P = P
    config.D = D
    config.S = S
    config.R = R

    config.white_back = False
    caseID = get_trailing_number(R)
    config.minBound = np.array([-1, -1, -1], dtype=np.float32)
    config.maxBound = np.array([1, 1, 1], dtype=np.float32)
    config.sceneShiftFirst = np.array([0, 0, 0], dtype=np.float32)
    config.sceneScaleSecond = float(0.8) if ((caseID % 8) not in [3, 7]) else (float(0.7))

    config.datasetConfDict = OrderedDict([])
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"] = {
        "dataset": "renderingNerfBlender58Pointk8L112V100",
        "caseID": get_trailing_number(R),  # [0~7]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        "batchSize": 4096,
        "class": "RenderingNerfBlenderDataset",
        "winWidth": 800,
        "winHeight": 800,
        "ray_near": 2.0,
        "ray_far": 6.0,
        "getRaysMeans": "ELU",
        "minBound": config.minBound,
        "maxBound": config.maxBound,
        "sceneShiftFirst": config.sceneShiftFirst,
        "sceneScaleSecond": config.sceneScaleSecond,
        "hdrScaling": 1,  # This is not used here, but might be useful for total benchmarking
        "white_back": config.white_back,
        "ifLoadImg": False,
        "ifLoadHdr": True,
        "ifMaskDilate": 0,
        "lgtLoadingMode": "QSPL",
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    # net

    config.N_depth = 8
    config.N_depth_appearance = 8
    config.N_width = 256
    config.N_width_bottleneck = 128
    config.N_width_appearance = 256
    config.N_emb_xyz = 10
    config.N_emb_dir = 4
    config.deg_directional_enc = 5
    config.skips = [4]
    config.skips_appearance = [4]
    config.N_importance = 128  # fine level
    config.N_samples = 64  # coarse level
    config.chunk = 32 * 1024  # avoid OOM, effective in the val mode.
    config.use_disp = False
    config.perturb = 0.0
    config.noise_std = 0

    config.density_fix = False

    config.reflection_surface_cutoff = 0.025

    # optimizer
    config.adam_lr = 5.0e-4  # all follows the mlp settings
    config.adam_betas = (0.9, 0.999)
    config.adam_epsilon = 1e-8

    config.rgb_sigmoid_extend_epsilon = 0.0  # try 0.1

    # loss hyper
    # config.masked_in_weight = 3.0
    # config.masked_out_weight = 1.0 / 3

    # wl
    wl = {}
    wl["lossMain"] = 1.0
    wl["lossMask"] = 0.0
    wl["lossTie"] = 0.001 * 1.0
    wl["lossBackface"] = 0.01 * 1.0  # Now we use mean instead of sum
    config.wl = wl

    return config
