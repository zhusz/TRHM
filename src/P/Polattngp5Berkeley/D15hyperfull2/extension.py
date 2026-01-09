import torch
import os
import copy
import socket
from collections import OrderedDict
from multiprocessing import shared_memory
from configs_registration import getConfigGlobal
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from Bprelight4.testDataEntry.renderingNerfBlender.renderingLightStageLfhyper import RenderingLightStageLfhyper
from Bprelight4.testDataEntry.renderingNerfBlender.renderingNerfBlenderDatasetLfhyper import RenderingNerfBlenderDatasetLfhyper
from Bprelight4.testDataEntry.testDataEntryPool import getTestDataEntryDict
# from ..trainer import Trainer
from ..D15hyperfull.extension import DTrainer as Trainer
from ..tngp1.nerf.refrelight_large_exp_only_r_hint15hyperfull2 import NeRFNetwork as NeRFNetwork15
from ..tngp1.nerf.renderer_hint15hyperfull2 import NeRFRenderer
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
# import nerfacc
from codes_py.toolbox_nerfacc.estimators.occ_grid import OccGridEstimator
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.rotations_v1 import ELU2cam
from torch.cuda.amp.grad_scaler import GradScaler  # You need to import nerfacc first before this one
from collections import defaultdict
from functools import partial
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb
from codes_py.toolbox_3D.nerf_metrics_v1 import psnr
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc
import numpy as np
import math
import time


bt = lambda s: s[0].upper() + s[1:]


class DTrainer(Trainer):
    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        if not config.double_density_net:
            model15 = NeRFNetwork15(config=config, if_need_normal3_net=True, if_need_hyper_net=True, df=config.density_fix)
            model15 = model15.to(self.cudaDeviceForAll)
        else:
            assert not config.density_fix
            model15 = NeRFNetwork15(config=config, if_need_normal3_net=True, if_need_hyper_net=True, df=config.density_fix)
            model15 = model15.to(self.cudaDeviceForAll)
            model14 = NeRFNetwork15(config=config, if_need_normal3_net=True, if_need_hyper_net=False, df=True)
            model14 = model14.to(self.cudaDeviceForAll)

        renderer = NeRFRenderer(
            config=config,
        )
        renderer = renderer.to(self.cudaDeviceForAll)

        occupancy_grid = OccGridEstimator(
            roi_aabb=torch.FloatTensor([-1, -1, -1, 1, 1, 1]),
            resolution=128,
            levels=4,
        )
        occupancy_grid = occupancy_grid.to(self.cudaDeviceForAll)

        models = {}
        models["model17"] = model15
        if config.double_density_net:
            models["model16"]= model14
        models["renderer"] = renderer
        models["occGrid"] = occupancy_grid
        meta = {
            "nonLearningModelNameList": ["occGrid", "renderer", "model16"],
            "nonLearningButToSave": ["occGrid"],  # , "model16"],
            "if_add_in_hdr3": config.if_add_in_hdr3,
        }
        models["meta"] = meta

        self.models = models

    def _netFinetuningInitialization(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netFinetuningInitialization")

        finetuningPDSRI = self.config.finetuningPDSRI
        if finetuningPDSRI is not None:
            for kNow, kPre in [("model17", "modelFast"), ("model16", "modelFast"), ("occGrid", "occGrid")]:
                print(kNow, kPre)
                if (kNow == "model16") and (not config.double_density_net):
                    continue
                fn = (
                    self.projRoot
                    + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth"
                    % (
                        finetuningPDSRI["P"],
                        finetuningPDSRI["D"],
                        finetuningPDSRI["S"],
                        finetuningPDSRI["R"],
                        kPre,
                        finetuningPDSRI["I"],
                    )
                )
                assert os.path.isfile(fn), fn
                with open(fn, "rb") as f:
                    print("Finetuning from %s" % fn)
                    loaded_state_dict = torch.load(f, map_location=self.cudaDeviceForAll)
                keys = list(loaded_state_dict.keys())
                keys_to_del = []
                if kPre == "modelFast":
                    keys_to_del += [k for k in keys if k in [
                        # "normal3_alpha.params", "normal3_hdr.params", "color_net.params"
                        "color_net.params"
                    ]]
                    if self.models[kNow].if_need_normal3_net:
                        keys_to_del += [k for k in keys if k in ["normal3_alpha.params", "normal3_hdr.params", "normal3_envmap.params"]]
                    if self.models[kNow].if_need_hyper_net:
                        pass  # do nothing
                for _ in keys_to_del:
                    del loaded_state_dict[_]
                self.models[kNow].load_state_dict(  # k == "model"
                    loaded_state_dict,
                    strict=False,
                )


def returnExportedClasses(
    wishedClassNameList,
):

    exportedClasses = {}

    if wishedClassNameList is None or "DTrainer" in wishedClassNameList:
        exportedClasses["DTrainer"] = DTrainer

    return exportedClasses
