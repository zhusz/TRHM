from ..trainer import Trainer
import torch
from ..tngp1.nerf.refrelight_large_exp_only_r_hint2 import NeRFNetwork
# import nerfacc
from codes_py.toolbox_nerfacc.estimators.occ_grid import OccGridEstimator


class DTrainer(Trainer):
    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        model = NeRFNetwork(
            D=config.N_depth,
            D_appearance=config.N_depth_appearance,
            W=config.N_width,
            W_bottleneck=config.N_width_bottleneck,
            W_appearance=config.N_width_appearance,
            deg_emb_xyz=config.N_emb_xyz,
            deg_emb_dir=config.N_emb_dir,
            deg_directional_enc=config.deg_directional_enc,
            skips=config.skips,
            skips_appearance=config.skips_appearance,
            rgb_sigmoid_extend_epsilon=config.rgb_sigmoid_extend_epsilon,
            density_fix=config.density_fix,
            minBound=config.minBound,
            maxBound=config.maxBound,
            cudaDevice=self.cudaDeviceForAll,

            bound=1,
            cuda_ray=True,
            enable_refnerf=True,
            density_scale=1,
            min_near=0.2,
            density_thresh=10,
            bg_radius=-1,
            empty_cache_stop_iter=config.empty_cache_stop_iter,
            render_step_size_keys=config.render_step_size_keys,
            render_step_size_vals=config.render_step_size_vals,

            config=config,
        )
        model = model.to(self.cudaDeviceForAll)

        occupancy_grid = OccGridEstimator(
            roi_aabb=torch.FloatTensor([-1, -1, -1, 1, 1, 1]),
            resolution=128,
            levels=4,
        )
        occupancy_grid = occupancy_grid.to(self.cudaDeviceForAll)

        models = {}
        models["model"] = model
        models["occGrid"] = occupancy_grid
        meta = {
            "nonLearningModelNameList": ["occGrid"],
            "nonLearningButToSave": ["occGrid"],
            "density_fn": model.density,  # this makes it easier when you are to use DDP
            "render_fn": model.render,
        }
        models["meta"] = meta

        self.models = models


def returnExportedClasses(
    wishedClassNameList,
):

    exportedClasses = {}

    if wishedClassNameList is None or "DTrainer" in wishedClassNameList:
        exportedClasses["DTrainer"] = DTrainer

    return exportedClasses
