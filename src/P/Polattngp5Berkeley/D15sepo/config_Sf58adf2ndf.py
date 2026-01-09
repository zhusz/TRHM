from .config_SgCap7V4 import getConfigFunc as getConfigFuncParent, get_trailing_number
import numpy as np
from collections import OrderedDict
from Bprelight4.testDataEntry.testDataEntryPool import nerfSynthetic_hdrScalingList


hdr_clipped_clip_table = {
    0: nerfSynthetic_hdrScalingList[0] * 2.0,
    1: nerfSynthetic_hdrScalingList[1] * 4.0 / 2,
    2: nerfSynthetic_hdrScalingList[2] * 4.0 / 2,
    3: nerfSynthetic_hdrScalingList[3] * 2.0,
    4: nerfSynthetic_hdrScalingList[4] * 4.0 / 2,
    5: nerfSynthetic_hdrScalingList[5] * 10.0,
    6: nerfSynthetic_hdrScalingList[6] * 4.0 / 2,
    7: nerfSynthetic_hdrScalingList[7] * 1.0,
}

highlight_case_list = [0, 3, 5, 7]


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.white_back = False
    caseID = get_trailing_number(R)
    config.minBound = np.array([-1, -1, -1], dtype=np.float32)
    config.maxBound = np.array([1, 1, 1], dtype=np.float32)
    config.sceneShiftFirst = np.array([0, 0, 0], dtype=np.float32)
    config.sceneScaleSecond = float(0.8) if ((caseID % 8) not in [3, 7]) else (float(0.7))

    config.datasetConfDict = OrderedDict([])
    bsz_base = 512  # rather than 512
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"] = {
        "dataset": "renderingNerfBlender58Pointk8L112V100",
        "caseID": caseID,  # [0~7]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        "ifPreloadModeFgbg": True,
        "batchSizeFg": 6 * bsz_base,  # 3072,
        "batchSizeBg": 2 * bsz_base,  # 1024,
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
        "hdrScaling": nerfSynthetic_hdrScalingList[caseID],
        "white_back": config.white_back,
        "ifLoadImg": False,
        "ifLoadHdr": True,
        "ifMaskDilate": 0,
        "lgtLoadingMode": "QSPL",

        # "ifPreloadHighlight": caseID in [7],  # although R5 does not need highlight, and normal3 is fixed to be normal2, we still need to predict hdr in the hdr == hdr2 + hdr3 way, that hdr2 is clipped under a particular value (e.g. 10 for R5)
        "ifPreloadHighlight": (caseID % 8) in highlight_case_list,  # we still need to get the highlight batch preloading - for making the highlighted places get more sampling and more weights for training
        "batchSizeHighlight": 128 if ((caseID % 8) in highlight_case_list) else 0,
        "batchSizeSublight": 128 * 7 if ((caseID % 8) in highlight_case_list) else 0,

        # "debugReadTwo": True,
        # "debugReadHowMany": 2,
        # "debugReadEvery": True,
        # "debugReadEveryHowMany": 100,
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    # config.density_fix = density always fixed in D15sepo

    config.needDensityFixer = (caseID % 8 == 7)

    config.if_highlight_case = (caseID in highlight_case_list)

    config.N_width_appearance = 64
    config.N_depth_appearance = 4
    config.skips_appearance = [2]

    config.ifShrinkNdotHloss = True
    config.iterShrinkNdotHloss = 200960
    config.rateShrinkNdotHloss = 0.04

    config.if_normal3_tied_to_normal2 = False
    # config.if_normal3_tied_to_normal2 = False

    config.hdr_clipped_clip = hdr_clipped_clip_table[caseID % 8]
    config.hdrloss_clipped_lp = 2
    config.hdr_highlight_clip = 40.0
    config.hdrloss_highlight_lp = 2

    config.finetuningPDSRI = {
        "P": "Polattngp5Berkeley",
        "D": "D14sepo",
        "S": "Sf58adf2ndf",
        "R": "R%d" % caseID,
        "I": 10960,  # to be updated
    }   
    if caseID % 8 == 7:
        config.finetuningPDSRI_densityTeacher = {
            "P": "PrefnerfmlpBerkeley",
            "D": "D0large",
            "S": "Sg600h8s8scalebetter",
            "R": "R%d" % get_trailing_number(R),
            "I": 480080,
        }
    else:
        config.finetuningPDSRI_densityTeacher = {
            "hehe": "generally, this is not useful for cases fall into this branch of the synthetic data"
        }

    return config
