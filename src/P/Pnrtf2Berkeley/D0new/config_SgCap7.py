import re
from collections import OrderedDict

import numpy as np

from .config_Sg500h8 import getConfigFunc as getConfigFuncParent  # noqa

from Bprelight4.testDataEntry.testDataEntryPool import capture7jf_hdrScalingList

originalBoundTable = np.array([
    [-0.2, -0.25, 1.0, 0.1, 0.05, 1.4],  # 0
    [-0.035, -0.14, 1.05, 0.05, -0.04, 1.28],  # 1
    [-0.08, -0.13, 1.11, 0.08, -0.04, 1.27],  # 2
    [-0.08, -0.13, 1.04, 0.07, -0.06, 1.3],  # 3
    [-0.15, -0.25, 0.9, 0.15, 0.05, 1.45],  # 4
    [-0.045, -0.12, 1.05, 0.05, -0.04, 1.3],  # 5
    [-0.04, -0.12, 1.05, 0.05, -0.06, 1.28],  # 6
    [-0.08, -0.14, 1.05, 0.09, -0.03, 1.32],  # 7
    [-0.04, -0.12, 1.02, 0.05, -0.06, 1.35],  # 8
    [-0.07, -0.12, 1.05, 0.08, -0.06, 1.35],  # 9
    [-0.06, -0.12, 1.04, 0.05, -0.06, 1.30],  # 10
], dtype=np.float32)


def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else 0


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict = OrderedDict([])
    caseID = get_trailing_number(R)
    originalMaxLen = float(originalBoundTable[caseID, 5] - originalBoundTable[caseID, 2])
    # sceneShiftFirst = (originalBoundTable[caseID, :3] + originalBoundTable[caseID, 3:6]) / 2
    # sceneScaleSecond = float(2 / originalMaxLen)
    sceneShiftFirst = np.zeros((3,), dtype=np.float32)
    sceneScaleSecond = float(1)
    maxBound = (originalBoundTable[caseID, 3:6] - sceneShiftFirst) * sceneScaleSecond
    minBound = -maxBound
    config.minBound = minBound
    config.maxBound = maxBound
    bsz_base = 512  # if caseID in [2, 3] else 480
    config.datasetConfDict["capture7jf"] = {
        "dataset": "capture7jf",
        "bucket": "ol3t75",
        "caseID": get_trailing_number(R),  # [0~7]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        "batchSizeFg": 10 * 40000,
        "batchSizeBg": 2,  # bsz_base,
        "batchSizeMg": 2,  # bsz_base * 3,
        "mgPixDilation": 200,
        "class": "RenderingLightStageOriginal3",
        "preloadingForegroundZoneNumberUpperBound": 2,  # use -1 to use betterMask2 as the borderline
        "preloadingBackgroundZoneNumberLowerBound": 4,
        "singleImageForegroundZoneNumberUpperBound": -1,
        "singleImageBackgroundZoneNumberLowerBound": -1,  # Note this has to be the number from one line above "+1"
        "rays_background_pad_width": 0,
        "rays_background_pad_height": 0,
        "minBound": minBound,  # np.array([-1, -1, -1], dtype=np.float32),
        "maxBound": maxBound,  # np.array([1, 1, 1], dtype=np.float32),
        "sceneShiftFirst": sceneShiftFirst,
        "sceneScaleSecond": sceneScaleSecond,
        "hdrScaling": capture7jf_hdrScalingList[caseID],  # This is not used here, but might be useful for total benchmarking
        "white_back": config.white_back,
        "ifLoadImg": True,
        "ifLoadHdr": False,
        "ifLoadDepth": True,
        "loadDepthTag": "NeuS",
        "lgtLoadingMode": "QSPL",

        # "debugReadTwo": True,
        # "debugReadTwoHowMany": 1,
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    return config
