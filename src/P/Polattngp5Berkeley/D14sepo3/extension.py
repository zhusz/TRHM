import torch
from ..trainer import Trainer
from ..tngp1.nerf.refrelight_large_exp_only_r_hint14sepo3 import NeRFNetwork
from ..tngp1.nerf.renderer_hint14sepo3 import NeRFRenderer
# import nerfacc
from codes_py.toolbox_nerfacc.estimators.occ_grid import OccGridEstimator
from collections import defaultdict


class DTrainer(Trainer):
    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        if config.get("needDensityFixer", False):
            modelDensityTeacher = NeRFNetwork(config=config, if_self_is_model_density_teacher=True)
            modelDensityTeacher = modelDensityTeacher.to(self.cudaDeviceForAll)
        else:
            modelDensityTeacher = None

        model = NeRFNetwork(config=config, if_self_is_model_density_teacher=False)
        model = model.to(self.cudaDeviceForAll)

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
        models["model"] = model
        if config.get("needDensityFixer", False):
            models["modelDensityTeacher"] = modelDensityTeacher
        models["renderer"] = renderer
        models["occGrid"] = occupancy_grid
        meta = {
            "nonLearningModelNameList": ["occGrid", "renderer", "modelDensityTeacher"],
            "nonLearningButToSave": ["occGrid"],
            "density_fn": model.density,  # this makes it easier when you are to use DDP
            "render_fn": renderer.render,
        }
        models["meta"] = meta

        self.models = models

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

        batch_head_index = kwargs.get("batch_head_index", None)

        model = models["model"]
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        bg_color = rays[:, 10:13]

        # debug
        # rays_o = rays_o[:, [1, 2, 0]]
        # rays_d = rays_d[:, [1, 2, 0]]

        # outputs = model.render(
        outputs = models["meta"]["render_fn"](
            rays_o, rays_d, models=models, staged=False, bg_color=bg_color, perturb=True, force_all_rays=force_all_rays,
            min_near=0.2, density_thresh=10, bg_radius=-1,
            # batch_thgpu=batch_thgpu,  # debug purpose
            iterCount=iterCount,
            requires_grad=requires_grad,
            if_query_the_expected_depth=if_query_the_expected_depth,
            lgtInput=lgtInput,
            callFlag=callFlag,
            logDir=logDir,
            hyperOutputs=hyperOutputs,
            lgtMode=lgtMode,
            envmap0=envmap0,
            batch_head_index=batch_head_index,
        )

        # for k in ["rgb", "depth", "opacity", "lossTieRays", "lossBackfaceRays", "statNumSampledPoints"]:
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
