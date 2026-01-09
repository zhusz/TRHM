import torch
import os
from ..trainer import Trainer
from ..tngp1.nerf.refrelight_large_exp_only_r_hint14sepo3 import NeRFNetwork as NeRFNetwork14
from ..tngp1.nerf.refrelight_large_exp_only_r_hint15sepo import NeRFNetwork as NeRFNetwork15
from ..tngp1.nerf.renderer_hint15sepo import NeRFRenderer
# import nerfacc
from codes_py.toolbox_nerfacc.estimators.occ_grid import OccGridEstimator
from collections import defaultdict


class DTrainer(Trainer):
    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        if config.get("needDensityFixer", False):
            modelDensityTeacher = NeRFNetwork14(config=config, modelDensityTeacher=None)
            modelDensityTeacher = modelDensityTeacher.to(self.cudaDeviceForAll)
        else:
            modelDensityTeacher = None

        model14 = NeRFNetwork14(config=config, modelDensityTeacher=modelDensityTeacher)
        model14 = model14.to(self.cudaDeviceForAll)
        model15 = NeRFNetwork15(config=config, modelDensityTeacher=modelDensityTeacher)
        model15 = model15.to(self.cudaDeviceForAll)

        renderer = NeRFRenderer(
            bound=1,
            cuda_ray=True,
            enable_refnerf=True,
            density_scale=1,
            min_near=0.2,
            density_thresh=10,
            bg_radius=-1,

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
        models["model14"] = model14
        models["model"] = model15
        if config.get("needDensityFixer", False):
            models["modelDensityTeacher"] = modelDensityTeacher
        models["renderer"] = renderer
        models["occGrid"] = occupancy_grid
        meta = {
            "nonLearningModelNameList": ["occGrid", "renderer", "model14", "modelDensityTeacher"],
            "nonLearningButToSave": ["occGrid"],
            "density_fn": model15.density,  # this makes it easier when you are to use DDP
            "render_fn": renderer.render,
        }
        models["meta"] = meta

        self.models = models

    def _netFinetuningInitialization(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netFinetuningInitialization")
        if hasattr(self.config, "finetuningPDSRI"):
            for k in ["model14", "model"]:
                finetuningPDSRI = self.config.finetuningPDSRI
                fn = self.projRoot + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth" % (
                    finetuningPDSRI["P"],
                    finetuningPDSRI["D"],
                    finetuningPDSRI["S"],
                    finetuningPDSRI["R"],
                    "model",
                    finetuningPDSRI["I"],
                )
                assert os.path.isfile(fn), fn
                with open(fn, "rb") as f:
                    loaded_state_dict = torch.load(f, map_location=self.cudaDeviceForAll)
                # del some params for the trained model
                if k == "model14":
                    pass  # do nothing
                elif k == "model":
                    keys = list(loaded_state_dict.keys())
                    keys_to_del = [
                        "normal3_alpha.params", "normal3_hdr.params"
                    ]
                    for _ in keys_to_del:
                        del loaded_state_dict[_]
                else:               
                    raise ValueError("Unknown model to finetune: k = %s" % k)
                strict = k == "model14"
                self.models[k].load_state_dict(loaded_state_dict, strict=strict)
                print("Finished Finetuning for %s from %s (strict = %s)" % (k, fn, strict))
        else:
            self.logger.info("    No finetuning specs found")
        
        if config.needDensityFixer:
            assert hasattr(config, "finetuningPDSRI_densityTeacher")
            finetuningPDSRI_densityTeacher = config.finetuningPDSRI_densityTeacher
            if finetuningPDSRI_densityTeacher["P"] == "PrefnerfmlpBerkeley":
                fn = (
                    self.projRoot
                    + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth"
                    % (
                        finetuningPDSRI_densityTeacher["P"],
                        finetuningPDSRI_densityTeacher["D"],
                        finetuningPDSRI_densityTeacher["S"],
                        finetuningPDSRI_densityTeacher["R"],
                        "nerfFine",
                        finetuningPDSRI_densityTeacher["I"],
                    )
                )
                assert os.path.isfile(fn), fn
                with open(fn, "rb") as f:
                    loaded_state_dict = torch.load(f, map_location=self.cudaDeviceForAll)
                if (config.P == finetuningPDSRI_densityTeacher["P"]) and (config.D == finetuningPDSRI_densityTeacher["D"]):
                    raise NotImplementedError("I do not think currently we need this branch")
                else:
                    keys = list(loaded_state_dict.keys())
                    keys_to_del = [k for k in keys if (
                        (not k.startswith("xyz_encoding_")) and (not k.startswith("sigma.")) 
                        and (not k.startswith("out_unormalized_normal_pred_spatial."))
                    )]
                    for _ in keys_to_del:
                        del loaded_state_dict[_]
                    self.models["modelDensityTeacher"].load_state_dict(
                        loaded_state_dict,
                        strict=False,
                    )
            else:
                raise NotImplementedError(
                    "You may wish to refer to how D14sepo did this, whose member method is defined in the parent class (trainer.py)"
                )

    @staticmethod
    def forwardRendering(rays, **kwargs):
        models = kwargs["models"]
        meta = models["meta"]
        config = kwargs["config"]
        iterCount = kwargs["iterCount"]
        cpu_ok = kwargs["cpu_ok"]
        force_all_rays = kwargs["force_all_rays"]
        requires_grad = kwargs["requires_grad"]
        if_query_the_expected_depth = kwargs["if_query_the_expected_depth"]
        lgtInput = kwargs["lgtInput"]
        callFlag = kwargs["callFlag"]
        logDir = kwargs["logDir"]
        hyperOutputs = kwargs.get("hyperOutputs", None)
        lgtMode = kwargs.get("lgtMode", "pointLightMode")
        envmap0 = kwargs["envmap0_thgpu"]
        B = rays.shape[0]
        results = defaultdict(list)
        # batch_thgpu = kwargs["batch_thgpu"]  # debug purpose - for inspecting - do not use for computation

        model = models["model"]
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        bg_color = rays[:, 10:13]

        # outputs = model.render(
        outputs = models["meta"]["render_fn"](
            rays_o, rays_d, models=models, staged=False, bg_color=bg_color, perturb=True, force_all_rays=force_all_rays,
            min_near=0.2, density_thresh=10, bg_radius=-1,
            # batch_thgpu=batch_thgpu,  # debug purpose
            iterCount=iterCount,
            requires_grad=False,  # requires_grad,  # now always False
            if_query_the_expected_depth=if_query_the_expected_depth,
            lgtInput=lgtInput,
            callFlag=callFlag,
            logDir=logDir,
            hyperOutputs=hyperOutputs,
            lgtMode=lgtMode,
            envmap0=envmap0,
        )

        for k in outputs.keys():
            if k in outputs.keys():
                v = outputs[k]
                if cpu_ok:
                    v = v.detach().cpu()
                results[k + "_fine"] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results


def returnExportedClasses(
    wishedClassNameList,
):

    exportedClasses = {}

    if wishedClassNameList is None or "DTrainer" in wishedClassNameList:
        exportedClasses["DTrainer"] = DTrainer

    return exportedClasses
