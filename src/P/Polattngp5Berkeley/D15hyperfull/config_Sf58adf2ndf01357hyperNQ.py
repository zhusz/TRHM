import numpy as np
import re
from easydict import EasyDict
from collections import OrderedDict
from Bprelight4.testDataEntry.testDataEntryPool import nerfSyntheticHyper_hyperHdrScalingList

highlight_case_list = [0, 1, 3, 5, 7]


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
    config.hyperHdrScaling = nerfSyntheticHyper_hyperHdrScalingList[caseID]

    config.datasetConfDict = OrderedDict([])
    bsz_base = 512  # rather than 512
    config.datasetConfDict["renderingNerfBlender58EnvolatGray16x32k8distillV100"] = {
        "dataset": "renderingNerfBlender58EnvolatGray16x32k8distillV100",
        "caseID": caseID,  # [0~7]
        "singleImageMode": False,  # Because this is for training
        "split": "train",
        # "ifPreloadModeFgbg": True,
        "batchSizeFg": 6 * bsz_base,  # 3072,
        "batchSizeBg": 2 * bsz_base,  # 1024,
        "batchSizeEnvmapLgt": 16,
        "class": "RenderingNerfBlenderDatasetLfhyper",
        "winWidth": 800,
        "winHeight": 800,
        "ray_near": 2.0,
        "ray_far": 6.0,
        "getRaysMeans": "ELU",
        "rays_background_pad_width": 0,
        "rays_background_pad_height": 0,
        "minBound": config.minBound,
        "maxBound": config.maxBound,
        "sceneShiftFirst": config.sceneShiftFirst,
        "sceneScaleSecond": config.sceneScaleSecond,
        "hyperHdrScaling": config.hyperHdrScaling,
        "white_back": config.white_back,
        "ifLoadImg": False,
        "ifLoadHdr": True,
        "ifLoadDepth": False,
        "loadDepthTag": None,
        "ifMaskDilate": 0,
        # "lgtLoadingMode": "QSPL",

        "lgtDataset": "lgtEnvmapLaval1k",
        "lgt_quantile_cut_min": 0.999,
        "lgt_quantile_cut_max": 1.0,
        "lgt_quantile_cut_fixed": 1.0,
        "if_lgt_load_highres": False,

        "debugReadTwo": False,
    }

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    config.density_fix = False  # caseID in [0, 1, 2, 4, 5, 6]

    config.if_highlight_case = (caseID in highlight_case_list)

    config.if_add_in_hdr3 = caseID in highlight_case_list
    config.inject_normal3 =  caseID in highlight_case_list

    config.memory_bank_read_out_freq = 0  # No need to do so under the synthetic data
    config.paramInitialScale = 1.0

    config.hash_n_features_per_level = 4
    config.sigma_b_dim = 31
    config.sigma_n_hidden_layers = 1  # should get shortened later for the purpose of computing efficiency.
    config.color_net_hidden_layers = 3

    # net

    config.N_depth = 8
    config.N_depth_appearance = 8  # 4
    config.N_width = 256
    config.N_width_bottleneck = 128
    config.N_width_appearance = 256  # 64
    config.N_emb_xyz = 10
    config.N_emb_dir = 4
    config.deg_directional_enc = 5
    config.N_emb_hint_pointlight_opacities = 10
    config.N_emb_hint_pointlight_ggx = 10
    config.skips = [4]
    config.skips_appearance = [4]  # [2]
    config.chunk = 32 * 1024  # avoid OOM, effective in the val mode.

    config.reflection_surface_cutoff = 0.025

    # optimizer
    config.adam_lr = 5.0e-4  # all follows the mlp settings
    config.adam_betas = (0.9, 0.999)
    config.adam_epsilon = 1e-8

    config.scheduler_base = 1.0  # for mlp

    config.render_step_size_keys = np.array([0], dtype=np.int32) * 2
    config.render_step_size_vals = np.array([1e-3], dtype=np.float32)
    config.empty_cache_stop_iter = 0  # int(config.render_step_size_keys[-1]) + 10

    config.gray_envmap_training = True  # Now this must be true
    config.pyramid_ksize_list = np.array([0, 7, 15, 31, 51, 101], dtype=np.int32)
    config.pyramid_len = int(config.pyramid_ksize_list.shape[0])

    config.if_normal3_tied_to_normal2 = False

    config.hdr_clipped_clip = float("nan")  # hdr_clipped_clip_table[caseID % 8]
    config.hdrloss_clipped_lp = float("nan")  # 2
    config.hdr_highlight_clip = 10.0  # 40.0
    config.hdrloss_highlight_lp = 2

    wl = {}
    wl["lossMain"] = 1.0
    config.wl = wl

    config.needDensityFixer = False

    if caseID % 8 == 7:
        config.density_activation = "softplus"
    else:
        config.density_activation = "exp"

    config.if_disable_hint_ref_hdr = False

    if caseID in [3]:
        config.finetuningPDSRI = {
            "P": "Polattngp5Berkeley",
            "D": "D15sepod",
            "S": "Sft2",
            "R": "R%d" % caseID,
            "I": 400960,
        }
    else:
        config.finetuningPDSRI = {
            "P": "Polattngp5Berkeley",
            "D": "D15sepod",
            "S": "Sf58adf2ndf01357V44",
            "R": "R%d" % caseID,
            "I": 160960,
        }

    return config
