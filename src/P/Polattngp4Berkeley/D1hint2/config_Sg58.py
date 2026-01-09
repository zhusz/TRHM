import re
from collections import OrderedDict

import numpy as np

from .config_Sg500h8 import getConfigFunc as getConfigFuncParent  # noqa


def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else 0


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

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
        "hdrScaling": 4.0 if get_trailing_number(R) in [6, 7] else 1.0,
        "white_back": config.white_back,
        "ifLoadImg": False,
        "ifLoadHdr": True,
        "ifMaskDilate": 0,
        "lgtLoadingMode": "QSPL",
        "ifLoadDepth": False,
        "loadDepthTag": None,

        # "debugReadTwo": True,
        # "debugReadTwoHowMany": 1,
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]
    
    config.scheduler_base = 1.0  # for mlp
    
    config.render_step_size_keys = np.array([0], dtype=np.int32)  # because you start from finetuning, and you already have the mostly good geometry
    config.render_step_size_vals = np.array([1e-3], dtype=np.float32)
    config.empty_cache_stop_iter = 0  # int(config.render_step_size_keys[-1]) + 10
    
    if get_trailing_number(R) % 8 == 7:  # the ship
        config.finetuningPDSRI = {  # only the density part, to avoid initial gpu memory explosion
            "P": "PrefnerfmlpBerkeley",
            "D": "D0large",
            "S": "SgExistingwbs8scalebetter",
            "R": "R%d" % get_trailing_number(R),
            "I": 240960,
        }
    else:
        config.finetuningPDSRI = {  # only the density part, to avoid initial gpu memory explosion
            "P": "Ptngp4Berkeley",
            "D": "D0refmlp",
            "S": "SgExistingwbs0noft",
            "R": "R%d" % get_trailing_number(R),
            "I": 400960,
        }

    return config
