import re
from collections import OrderedDict

import numpy as np

from .config_Sg500h8 import getConfigFunc as getConfigFuncParent  # noqa
# from .config_Sg500h8 import getNerfBlenderSceneMinBound, getNerfBlenderSceneMaxBound
from .config_Sg500h8 import nerfSynthetic_blenderFrameTagList


def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else 0


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict = OrderedDict([])
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"] = {
        "dataset": "renderingNerfBlender58Pointk8L112V100",
        "caseID": get_trailing_number(config.R),  # [0~31]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        "batchSizeFg": 20 * 40000 // 2,
        "batchSizeMg": 2,
        "batchSizeBg": 2,
        "class": "RenderingNerfBlenderDataset",
        # "hdrScaling": 4.0 if get_trailing_number(R) in [6, 7] else 1.0,
        "sceneShiftFirst": np.zeros((3,), dtype=np.float32),
        "sceneScaleSecond": float(1.0),
        "getRaysMeans": "ELU",
        "ifPreloadModeFgbg": True,
        "winWidth": 800,
        "winHeight": 800,
        "ray_near": 2.0,
        "ray_far": 6.0,
        "minBound": np.array([-2, -2, -2], dtype=np.float32),
        "maxBound": np.array([2, 2, 2], dtype=np.float32),
        "white_back": False,
        "ifLoadImg": False,
        "ifLoadHdr": True,
        "ifMaskDilate": 0,
        "lgtLoadingMode": "QSPL",
        #
        "ifLoadDepth": True,
        "loadDepthTag": "R2neusdepth",
        "depthFileListA0Tag": "nameListNeureconwmDepth",
        "blenderFrameTag": "hehe",
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    # assert len(config.datasetConfDict) == 1
    # caseID = list(config.datasetConfDict.values())[0]["caseID"]
    # config.tightMinBound = getNerfBlenderSceneMinBound()[caseID, :]
    # config.tightMaxBound = getNerfBlenderSceneMaxBound()[caseID, :]

    return config
