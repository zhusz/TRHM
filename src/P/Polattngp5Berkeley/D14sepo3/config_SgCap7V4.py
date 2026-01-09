# V4 means V2 shallow

import re
from collections import OrderedDict

import numpy as np
from easydict import EasyDict

from Bprelight4.testDataEntry.testDataEntryPool import capture7jf_hdrScalingList

hdr_clipped_clip_table = {
    0: None,
    1: capture7jf_hdrScalingList[1] * 5,
    2: capture7jf_hdrScalingList[2] * 2.5,
    3: capture7jf_hdrScalingList[3] * 2.0,
    4: None,
    5: capture7jf_hdrScalingList[5] * 5,
    6: capture7jf_hdrScalingList[6] * 5,
    7: capture7jf_hdrScalingList[7] * 5,
    8: capture7jf_hdrScalingList[8] * 5,
    10: capture7jf_hdrScalingList[10] * 5,
}

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
    config = EasyDict()
    config.P = P
    config.D = D
    config.S = S
    config.R = R

    highlight_case_list = [2, 3]

    config.white_back = False

    config.datasetConfDict = OrderedDict([])
    caseID = get_trailing_number(R)
    originalMaxLen = float(originalBoundTable[caseID, 5] - originalBoundTable[caseID, 2])
    sceneShiftFirst = (originalBoundTable[caseID, :3] + originalBoundTable[caseID, 3:6]) / 2
    sceneScaleSecond = float(2 / originalMaxLen)
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
        "batchSizeFg": bsz_base * 4,
        "batchSizeBg": bsz_base,
        "batchSizeMg": bsz_base * 3,
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
        "lgtLoadingMode": "QSPL",

        "ifPreloadHighlight": (get_trailing_number(R) in highlight_case_list),
        "highlightBucket": "highlight331b",
        "batchSizeHighlight": 512 if (get_trailing_number(R) in highlight_case_list) else 0,
        "batchSizeSublight": 512 if (get_trailing_number(R) in highlight_case_list) else 0,

        # "debugReadTwo": True,
        # "debugReadTwoHowMany": 1,
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    config.if_highlight_case = (caseID in highlight_case_list)

    # net

    config.N_depth = 8
    config.N_depth_appearance = 4
    config.N_width = 256
    config.N_width_bottleneck = 128
    config.N_width_appearance = 64
    config.N_emb_xyz = 10
    config.N_emb_dir = 4
    config.deg_directional_enc = 5
    config.N_emb_hint_pointlight_opacities = 10
    config.N_emb_hint_pointlight_ggx = 10
    config.skips = [4]
    config.skips_appearance = [2]
    config.chunk = 32 * 1024  # avoid OOM, effective in the val mode.

    config.density_fix = False

    config.reflection_surface_cutoff = 0.025

    # optimizer
    """
    config.adam_lr = 0.01
    config.adam_betas = (0.9, 0.99)
    config.adam_epsilon = 1e-15
    """
    config.adam_lr = 5.0e-4  # all follows the mlp settings
    config.adam_betas = (0.9, 0.999)
    config.adam_epsilon = 1e-8

    config.scheduler_base = 1.0  # for mlp

    config.hdr_clipped_clip = hdr_clipped_clip_table[caseID]  # 2.0
    config.hdrloss_clipped_lp = 2
    config.hdr_highlight_clip = 9999.0
    config.hdrloss_highlight_lp = 2
    # config.rgb_sigmoid_extend_epsilon = 0.0  # try 0.1

    config.render_step_size_keys = np.array([0], dtype=np.int32) * 2
    config.render_step_size_vals = np.array([1e-3], dtype=np.float32)
    config.empty_cache_stop_iter = 0  # int(config.render_step_size_keys[-1]) + 10

    wl = {}
    wl["lossMain"] = 1.0
    wl["lossMask"] = 0.0
    wl["lossTie"] = 0.001 * 1.0
    wl["lossBackface"] = 0.01 * 1.0  # Now we use mean instead of sum
    wl["lossTie23"] = 0.001 * 1.0
    s = (0.06 / 10 / 10) if (get_trailing_number(R) in highlight_case_list) else 0  # V2
    wl["lossMainHighlight"] = s * wl["lossMain"]
    wl["lossMaskHighlight"] = s * wl["lossMask"]
    wl["lossTieHighlight"] = s * wl["lossTie"]
    wl["lossBackfaceHighlight"] = s * wl["lossBackface"]
    wl["lossTie23Highlight"] = s * wl["lossTie23"]

    wl["lossHighlightNdotH"] = (0.1 * 2) if (get_trailing_number(R) in highlight_case_list) else 0  # V2
    # wl["lossSuppressHdr3"] = 0.01 / 20  # V2
    wl["lossSuppressHdr3"] = 0.001 / 2
    config.wl = wl
    config.sloss = s
    config.ifShrinkNdotHloss = False

    if caseID in [7, 8, 10]:
        config.density_activation = "softplus"
        config.finetuningPDSRI = {
            "P": "PrefnerfmlpBerkeley",
            "D": "D0large",
            "S": "SgCap7nerf",
            "R": "R%d" % caseID,
            "I": 480960,
        }
    else:
        config.density_activation = "exp"
        config.finetuningPDSRI = {  # only the density part, to avoid initial gpu memory explosion
            "P": "Ptngp4Berkeley",
            "D": "D0refmlp",
            "S": "SgCap7",
            "R": "R%d" % get_trailing_number(R),
            "I": 480960,
        }

    return config
