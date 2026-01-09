import numpy as np
import re
from easydict import EasyDict
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

def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else 0

highlight_case_list = [0, 1, 3, 5, 7]


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
        "ifLoadDepth": False,
        "loadDepthTag": None,
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

    config.density_fix = caseID in [0, 1, 2, 4, 5, 6]

    config.if_highlight_case = (caseID in highlight_case_list)

    config.inject_normal3 =  caseID in highlight_case_list

    config.hash_n_features_per_level = 4
    config.sigma_b_dim = 31
    config.sigma_n_hidden_layers = 1  # should get shortened later for the purpose of computing efficiency.
    config.color_net_hidden_layers = 3

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

    # config.rgb_sigmoid_extend_epsilon = 0.0  # try 0.1

    config.render_step_size_keys = np.array([0], dtype=np.int32) * 2
    config.render_step_size_vals = np.array([1e-3], dtype=np.float32)
    config.empty_cache_stop_iter = 0  # int(config.render_step_size_keys[-1]) + 10

    config.gray_envmap_training = True  # Now this must be true
    config.pyramid_ksize_list = np.array([0, 7, 15, 31, 51, 101], dtype=np.int32)
    config.pyramid_len = int(config.pyramid_ksize_list.shape[0])

    config.if_normal3_tied_to_normal2 = False

    config.hdr_clipped_clip = hdr_clipped_clip_table[caseID % 8]
    config.hdrloss_clipped_lp = 2
    config.hdr_highlight_clip = 40.0
    config.hdrloss_highlight_lp = 2

    wl = {}
    wl["lossMain"] = 1.0
    wl["lossMask"] = 0.0
    wl["lossSigmaTie"] = 0.01
    wl["lossNormal3Tie"] = 0.1
    s = (0.06 / 10 / 10 / 5) if ((caseID % 8) in highlight_case_list) else 0  # V2
    wl["lossMainHighlight"] = s * wl["lossMain"]
    wl["lossMaskHighlight"] = s * wl["lossMask"]
    wl["lossSigmaTieHighlight"] = s * wl["lossSigmaTie"]
    wl["lossNormal3TieHighlight"] = s * wl["lossNormal3Tie"]
    wl["lossFreeSigmaTie"] = 0.01
    wl["lossHighlightNdotH"] = (0.1 * 2) if (caseID % 8 in highlight_case_list) else 0  # V2
    wl["lossSuppressHdr3"] = 0.01 / 20 / 4
    config.wl = wl
    config.sloss = s
    config.ifShrinkNdotHloss = True
    config.iterShrinkNdotHloss = 200960
    config.rateShrinkNdotHloss = 0.04

    config.needDensityFixer = False

    if caseID % 8 == 7:
        config.density_activation = "softplus"
    else:
        config.density_activation = "exp"

    config.finetuningPDSRI = None
    config.finetuningPDSRI_slow = {
        "P": "Polattngp5Berkeley",
        "D": "D15sepo" if (caseID % 8 in highlight_case_list) else "D14sepo3",
        "S": "Sf58adf2ndf01357d8V4" if (caseID % 8 in highlight_case_list) else "Sf58adf2ndf01357",
        "R": "R%d" % caseID,
        "I": 160960,
    }

    return config
