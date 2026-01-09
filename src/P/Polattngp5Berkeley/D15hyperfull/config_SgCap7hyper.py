
import re
from collections import OrderedDict

import numpy as np
from easydict import EasyDict
from Bprelight4.testDataEntry.testDataEntryPool import capture7jfHyper_hyperHdrScalingList

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


hdr_clipped_clip_table = {
    1: 5.0,
    2: 2.5,
    3: 2.0,
    5: 5.0,
    6: 5.0,
    7: 5.0,
    8: 5.0,
    10: 5.0,
}


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
    caseID = get_trailing_number(R)
    originalMaxLen = float(originalBoundTable[caseID, 5] - originalBoundTable[caseID, 2])
    sceneShiftFirst = (originalBoundTable[caseID, :3] + originalBoundTable[caseID, 3:6]) / 2
    sceneScaleSecond = float(2 / originalMaxLen)
    maxBound = (originalBoundTable[caseID, 3:6] - sceneShiftFirst) * sceneScaleSecond
    minBound = -maxBound
    config.minBound = minBound
    config.maxBound = maxBound
    config.hyperHdrScaling = capture7jfHyper_hyperHdrScalingList[caseID]
    config.datasetConfDict["capture7jfLfhyper"] = {
        "dataset": "capture7jfLfhyper",
        "caseID": caseID,  # [0~7]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        "batchSizeFg": 512 * 4,
        "batchSizeBg": 512,
        "batchSizeMg": 512 * 3,
        "batchSizeEnvmapLgt": 16,  # it seems setting it to "1" can avoid the initialization problem
        "mgPixDilation": 200,
        "class": "RenderingLightStageLfhyper",
        "rays_background_pad_width": 0,
        "rays_background_pad_height": 0,
        "minBound": minBound,  # np.array([-1, -1, -1], dtype=np.float32),
        "maxBound": maxBound,  # np.array([1, 1, 1], dtype=np.float32),
        "sceneShiftFirst": sceneShiftFirst,
        "sceneScaleSecond": sceneScaleSecond,
        "hdrScaling": None,  # This is not used here, but might be useful for total benchmarking
        "white_back": config.white_back,

        "lgtDataset": "lgtEnvmapLaval1k",
        "lgt_quantile_cut_min": 0.96,
        "lgt_quantile_cut_max": 1.0,
        "lgt_quantile_cut_fixed": 0.98,

        "hyperHdrScaling": config.hyperHdrScaling,
        "if_lgt_load_highres": False,

        "mbtot": int((75 * 162 * 1366 * 2048 * 0.25 / 512) * 3.2),  # 3.2 times of the 75 olat scenario
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    config.if_highlight_case = (caseID in [2, 3])

    # net
    config.memory_bank_read_out_freq = 50

    config.paramInitialScale = 1.0

    config.hash_n_features_per_level = 4
    config.sigma_b_dim = 31
    config.sigma_n_hidden_layers = 1

    config.inject_normal3 = caseID in [2, 3]

    config.N_depth = 8
    config.N_depth_appearance = 8
    config.N_width = 256
    config.N_width_bottleneck = 128
    config.N_width_appearance = 256
    config.N_emb_xyz = 10
    config.N_emb_dir = 4
    config.deg_directional_enc = 5
    config.N_emb_hint_pointlight_opacities = 10
    config.N_emb_hint_pointlight_ggx = 10
    config.skips = [4]
    config.skips_appearance = [4]
    config.chunk = 32 * 1024  # avoid OOM, effective in the val mode.

    config.density_fix = True

    config.reflection_surface_cutoff = 0.025

    config.gray_envmap_training = True  # Now this must be true
    config.pyramid_ksize_list = np.array([0, 7, 15, 31, 51, 101], dtype=np.int32)
    config.pyramid_len = int(config.pyramid_ksize_list.shape[0])

    # optimizer
    config.adam_lr = 5.0e-4  # all follows the mlp settings
    config.adam_betas = (0.9, 0.999)
    config.adam_epsilon = 1e-8

    config.scheduler_base = 1.0  # for mlp

    config.hdr_clipped_clip = hdr_clipped_clip_table[caseID]  # 2.0
    config.hdrloss_clipped_lp = 2
    config.hdr_highlight_clip = 9999.0
    config.hdrloss_highlight_lp = 2

    config.if_add_in_hdr3 = (caseID in [2, 3])
    config.inject_normal3 = (caseID in [2, 3])

    config.render_step_size_keys = np.array([0], dtype=np.int32) * 2
    config.render_step_size_vals = np.array([1e-3], dtype=np.float32)
    config.empty_cache_stop_iter = 0  # int(config.render_step_size_keys[-1]) + 10

    wl = {}
    wl["lossMain"] = 1.0
    config.wl = wl

    if caseID in [7, 8, 10]:
        config.density_activation = "softplus"
    else:
        config.density_activation = "exp"

    config.if_disable_hint_ref_hdr = True
    config.finetuningPDSRI = {
        "P": "Polattngp5Berkeley",
        "D": "D15sepod",
        "S": "SgCap7V44",
        "R": "R%d" % get_trailing_number(R),
        "I": 240960 if caseID == 3 else 160960,  #  if (caseID in [1, 2, 3, 6, 10]) else 200960,
    }

    return config
