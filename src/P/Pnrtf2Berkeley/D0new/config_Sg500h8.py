import re
from collections import OrderedDict

import numpy as np
from easydict import EasyDict

nerfSynthetic_blenderFrameTagList = [
        "0000", "0001", "0136", "0029",
        "0001", "0000", "0186", "0002",
    ]


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

    config.datasetConfDict = OrderedDict([])
    config.datasetConfDict["renderingNerfBlenderExpp500"] = {
        "dataset": "renderingNerfBlenderExpp500",
        "caseID": get_trailing_number(R),  # [0~7]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        "batchSize": 20 * 40000,  # 4096,
        "class": "RenderingNerfBlenderDataset",
        "winWidth": 800,
        "winHeight": 800,
        "ray_near": 2.0,
        "ray_far": 6.0,
        "minBound": np.array([-2, -2, -2], dtype=np.float32),
        "maxBound": np.array([2, 2, 2], dtype=np.float32),
        "white_back": config.white_back,
        "ifLoadImg": False,
        "ifLoadHdr": True,
        "ifMaskDilate": 0,
        "lgtLoadingMode": "NONE",
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    # hash
    config.hash_n_features_per_level = 16
    config.hash_base_resolution = 16
    config.hash_n_levels = 19
    config.hash_finest_resolution = 256
    config.hash_sparse = True

    # nrtf
    config.nrtf_W = 512
    config.nrtf_D = 7
    config.nrtf_skips = [3]
    config.nrtf_activation = "relu"

    # optimizer
    config.sparse_adam_lr = 5.0e-4
    config.adam_lr = 5.0e-4
    # config.adam_epsilon = 1.0e-8
    config.lr_scheduler_step = 30000
    config.lr_scheduler_gamma = 0.33

    # loss hyper
    config.masked_in_weight = 2.0
    config.masked_out_weight = 0.5

    # wl
    wl = {}
    wl["lossHdr"] = 1.0
    config.wl = wl

    return config
