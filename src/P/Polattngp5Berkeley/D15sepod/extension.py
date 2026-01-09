import torch
import os
import time
from socket import gethostname
from ..trainer import Trainer
from ..tngp1.nerf.refrelight_large_exp_only_r_hint15sepod import NeRFNetwork as NeRFNetworkFast
from ..tngp1.nerf.renderer_hint15sepod import NeRFRenderer, soft_clamp
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb
from codes_py.toolbox_3D.nerf_metrics_v1 import psnr
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc
import torch.optim as optim
import math
import numpy as np
from torch_ema import ExponentialMovingAverage
from torch.cuda.amp.grad_scaler import GradScaler
from functools import partial
# import nerfacc
from codes_py.toolbox_nerfacc.estimators.occ_grid import OccGridEstimator
from collections import defaultdict


bt = lambda s: s[0].upper() + s[1:]


class DTrainer(Trainer):
    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        if config.if_highlight_case:
            from ..tngp1.nerf.refrelight_large_exp_only_r_hint15sepo import NeRFNetwork as NeRFNetworkSlow
        else:
            from ..tngp1.nerf.refrelight_large_exp_only_r_hint14sepo3 import NeRFNetwork as NeRFNetworkSlow
        modelSlow = NeRFNetworkSlow(config=config)
        modelSlow = modelSlow.to(self.cudaDeviceForAll)

        modelFast = NeRFNetworkFast(config=config)
        modelFast = modelFast.to(self.cudaDeviceForAll)

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
        models["modelSlow"] = modelSlow
        models["modelFast"] = modelFast
        models["renderer"] = renderer
        models["occGrid"] = occupancy_grid
        meta = {
            "nonLearningModelNameList": ["occGrid", "renderer", "modelSlow"],
            "nonLearningButToSave": ["occGrid"],
            "density_fn": modelFast.density,  # this makes it easier when you are to use DDP
            "render_fn": renderer.render,
        }
        models["meta"] = meta

        self.models = models

    def _netFinetuningInitialization(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netFinetuningInitialization")
        
        # finetuning only the slow part
        finetuningPDSRI_slow = self.config.finetuningPDSRI_slow
        for kNow, kPre in [
            ("modelSlow", "model"),
            ("modelFast", "model"),
        ]:
            fn = (
                self.projRoot
                + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth"
                % (
                    finetuningPDSRI_slow["P"],
                    finetuningPDSRI_slow["D"],
                    finetuningPDSRI_slow["S"],
                    finetuningPDSRI_slow["R"],
                    kPre,
                    finetuningPDSRI_slow["I"],
                )
            )
            assert os.path.isfile(fn), fn
            with open(fn, "rb") as f:
                loaded_state_dict = torch.load(f, map_location=self.cudaDeviceForAll)
            if kNow == "modelSlow":
                # no need to del keys from the loaded_state_dict
                self.models[kNow].load_state_dict(loaded_state_dict, strict=True)
            elif kNow == "modelFast":
                # the hdr3 part is to be loaded. To del all the others (hdr2 part)
                keys_to_del = [k for k in loaded_state_dict.keys() if not k.startswith("normal3_")]
                for _ in keys_to_del:
                    del loaded_state_dict[_]
                self.models[kNow].load_state_dict(loaded_state_dict, strict=False)
            else:
                raise ValueError("Unknown kNow: %s" % kNow)

    def _optimizerSetups(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _optimizerSetups")

        # unless the two optimizers are different, you should write in this form
        modelKeys = [
            k
            for k in self.models.keys()
            if k != "meta" and k not in self.models["meta"]["nonLearningModelNameList"]
        ]
        params = list(self.models[modelKeys[0]].parameters())
        for j in range(1, len(modelKeys)):
            params += list(self.models[modelKeys[j]].parameters())
        self.optimizerModels = {}
        self.optimizerModels["all"] = optim.Adam(
            params,
            lr=config.adam_lr,
            betas=config.adam_betas,
            eps=config.adam_epsilon,
        )

        # gradScaler
        self.gradScalerModels = {}
        self.gradScalerModels["all"] = GradScaler(
            enabled=True
        )  # when you use tcnn with half precision, it is important to use gard_scaler

        # lr_scheduler
        self.schedulerModels = {}
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iterCount: config.scheduler_base ** min(iterCount / 30000, 1)
        )
        self.schedulerModels["all"] = scheduler(self.optimizerModels["all"])

        # ema
        self.emaModels = {}
        self.emaModels["all"] = ExponentialMovingAverage(self.models["modelFast"].parameters(), decay=0.95)

    def forwardNetNeRF(self, batch_thgpu, **kwargs):
        models = kwargs["models"]
        config = kwargs["config"]
        dataset = kwargs["dataset"]
        iterCount = kwargs["iterCount"]
        wl = config.wl
        self.assertModelsTrainingMode(models)
        """
        if (iterCount == 0) or ((iterCount - self.resumeIter) % 16 == 0):
            models["model"].update_extra_state()
        """
        if self.numMpProcess > 0:
            dist.barrier(device_ids=[self.rank])
        if True:  # all need to be updated
            models["occGrid"].update_every_n_steps(
                step=iterCount - max(self.resumeIter, 0),
                occ_eval_fn=partial(
                    models["modelSlow"].density,
                    if_do_bound_clamp=True,
                    if_output_only_sigma=True,
                    if_compute_normal3=False,
                ),
                occ_thre=10,
                n=16,
            )
        if self.numMpProcess > 0:
            dist.barrier(device_ids=[self.rank])
        if iterCount < config.empty_cache_stop_iter:
            torch.cuda.empty_cache()
        _dataset = "_" + dataset
        # For this mode, you need to modify the rays[:, 10:13]
        # (where valid_mask == False pixels need to be set to 1 or 0)
        lgtInput = {
            "lgtE": batch_thgpu["lgtE" + _dataset],
            "objCentroid": batch_thgpu["objCentroid" + _dataset],
        }
        results = self.forwardRendering(
            batch_thgpu["rays" + _dataset],
            models=models,
            config=config,
            iterCount=iterCount,
            cpu_ok=False,
            batch_thgpu=batch_thgpu,  # debug purpose
            force_all_rays=False,
            requires_grad=True,
            if_query_the_expected_depth=False,
            lgtInput=lgtInput,
            callFlag="forwardNetNeRF",
            logDir=self.logDir,
            lgtMode="pointLightMode",
            envmap0_thgpu=None,
            batch_head_index=None,
        )
        
        if config.sloss > 0:
            results_hs = {}  # hs represents highlight and sublight
            for hs in ["highlight", "sublight"]:
                lgtInput_hs = {
                    "lgtE": batch_thgpu["%sLgtEs_%s" % (hs, dataset)],
                    "objCentroid": batch_thgpu["%sObjCentroid_%s" % (hs, dataset)],
                }
                results_hs[hs] = self.forwardRendering(
                    batch_thgpu["%sRays_%s" % (hs, dataset)],
                    models=models,
                    config=config,
                    iterCount=iterCount,
                    cpu_ok=False,
                    batch_thgpu=batch_thgpu,  # debug purpose
                    force_all_rays=False,
                    requires_grad=True,
                    if_query_the_expected_depth=False,
                    lgtInput=lgtInput_hs,
                    callFlag="forwardNetNeRF",
                    logDir=self.logDir,
                    lgtMode="pointLightMode",
                    envmap0_thgpu=None,
                    batch_head_index=None,
                )
        else:
            results_hs = None
        assert torch.all(
            batch_thgpu["random_background_color_%s" % dataset] ==
            batch_thgpu["rays_%s" % dataset][:, 10:13]
        )
        gt_hdr = torch.where(
            batch_thgpu["valid_mask_%s" % dataset][:, None].repeat(1, 3),
            batch_thgpu["hdrs_%s" % dataset],
            batch_thgpu["random_background_color_%s" % dataset],
        )
        if config.sloss > 0:
            assert torch.all(
                batch_thgpu["highlightRandomBackgroundColor_%s" % dataset] ==
                batch_thgpu["highlightRays_%s" % dataset][:, 10:13]
            )
            # all the highlight pixels' valid_mask are True
            gt_hdr_hs = torch.cat([
                batch_thgpu["highlightHdrs_%s" % dataset],
                batch_thgpu["sublightHdrs_%s" % dataset],
            ], 0)
      
        lossMain = wl["lossMain"] * self.criterion_highlight(results["hdr_fine"], gt_hdr).nanmean()
        if ("lossMainHighlight" in wl.keys()) and (wl["lossMainHighlight"] > 0):
            lossMainHighlight = wl["lossMainHighlight"] * self.criterion_highlight(
                # results_highlight["hdr_fine"],
                torch.cat([
                    results_hs["highlight"]["hdr_fine"],
                    results_hs["sublight"]["hdr_fine"],
                ], 0),
                gt_hdr_hs,
            ).nanmean()
        else:
            lossMainHighlight = 0
        if ("lossMask" in wl.keys()) and (wl["lossMask"] > 0):
            lossMask = wl["lossMask"] * (
                (1.0 - batch_thgpu["valid_mask" + _dataset].float())
                * self.criterion_mask(
                    results["opacity_fine"], torch.zeros_like(results["opacity_fine"])
                )
            ).mean()
        else:
            lossMask = 0
        if ("lossMaskHighlight" in wl.keys()) and (wl["lossMaskHighlight"] > 0):
            lossMaskHighlight = wl["lossMaskHighlight"] * self.criterion_mask(
                # results_highlight["opacity_fine"],
                torch.cat([
                    results_hs["highlight"]["opacity_fine"],
                    results_hs["sublight"]["opacity_fine"],
                ], 0),
                torch.zeros(
                    int(
                        batch_thgpu["highlightRays_%s" % dataset].shape[0] +
                        batch_thgpu["sublightRays_%s" % dataset].shape[0]
                    ),
                    dtype=torch.float32, device=self.cudaDeviceForAll)
            ).mean()
        else:
            lossMaskHighlight = 0
 
        if ("lossSigmaTie" in wl.keys()) and (wl["lossSigmaTie"] > 0):
            lossSigmaTie = wl["lossSigmaTie"] * results["lossSigmaTie_fine"].mean()
        else:
            lossSigmaTie = 0
        if ("lossSigmaTieHighlight" in wl.keys()) and (wl["lossSigmaTieHighlight"] > 0):
            lossSigmaTieHighlight = wl["lossSigmaTieHighlight"] * torch.cat([
                results_hs["highlight"]["lossSigmaTie_fine"],
                results_hs["sublight"]["lossSigmaTie_fine"],
            ], 0).mean()
        else:
            lossSigmaTieHighlight = 0
        if ("lossNormal3Tie" in wl.keys()) and (wl["lossNormal3Tie"] > 0):
            lossNormal3Tie = wl["lossNormal3Tie"] * results["lossNormal3Tie_fine"].mean()
        else:
            lossNormal3Tie = 0
        if ("lossNormal3TieHighlight" in wl.keys()) and (wl["lossNormal3TieHighlight"] > 0):
            lossNormal3TieHighlight = wl["lossNormal3TieHighlight"] * torch.cat([
                results_hs["highlight"]["lossNormal3Tie_fine"],
                results_hs["sublight"]["lossNormal3Tie_fine"],
            ], 0).mean()
        else:
            lossNormal3TieHighlight = 0
        if ("lossFreeSigmaTie" in wl.keys()) and (wl["lossFreeSigmaTie"] > 0):
            assert np.all(config.minBound + config.maxBound == 0)
            freexyz = (2 * torch.rand(
                int(batch_thgpu["indexID_%s" % dataset].shape[0]), 3, dtype=torch.float32,
                device=batch_thgpu["indexID_%s" % dataset].device,
            ) - 1) * torch.from_numpy(config.maxBound).float().to(batch_thgpu["indexID_%s" % dataset].device)[None, :]
            free_sigma_before_activation_fast = models["modelFast"].density(
                freexyz, if_do_bound_clamp=True, if_output_only_sigma=True, if_compute_normal3=False,
                if_output_sigma_before_activation=True,
            ).float()
            free_sigma_before_activation_slow = models["modelSlow"].density(
                freexyz, if_do_bound_clamp=True, if_output_only_sigma=True, if_compute_normal3=False,
                if_output_sigma_before_activation=True,
            ).float()
            assert torch.all(torch.isfinite(free_sigma_before_activation_slow))
            free_sigma_before_activation_fast = torch.where(
                free_sigma_before_activation_fast > -torch.inf, free_sigma_before_activation_fast, float(free_sigma_before_activation_slow.min()) - 1)
            assert torch.all(torch.isfinite(free_sigma_before_activation_fast))
            loss_free_sigma_tie = torch.abs(
                soft_clamp(free_sigma_before_activation_slow, s_left=float(-10), s_right=float(10), shrinkage=float(0.001))
                -
                soft_clamp(free_sigma_before_activation_fast, s_left=float(-10), s_right=float(10), shrinkage=float(0.001))
            )
            lossFreeSigmaTie = wl["lossFreeSigmaTie"] * loss_free_sigma_tie.mean()
        else:
            lossFreeSigmaTie = 0
        if ("lossHighlightNdotH" in wl.keys()) and (wl["lossHighlightNdotH"] > 0):
            assert torch.all(results_hs["highlight"]["normal3DotH_fine"] <= 1.00001)
            lossHighlightNdotH = wl["lossHighlightNdotH"] * torch.abs(
                1 - results_hs["highlight"]["normal3DotH_fine"]
            ).mean()
            if (config.ifShrinkNdotHloss and (iterCount >= config.iterShrinkNdotHloss)):
                lossHighlightNdotH *= float(config.rateShrinkNdotHloss)
        else:
            lossHighlightNdotH = 0
        if ("lossSuppressHdr3" in wl.keys()) and (wl["lossSuppressHdr3"] > 0):
            # This loss does not apply to results_hs - in there, lots of hdr3 should be non-zero
            lossSuppressHdr3 = wl["lossSuppressHdr3"] * torch.abs(results["hdr3_fine"]).mean()
        else:
            lossSuppressHdr3 = 0
 
        assert wl.get("lossBfdirect", 0) == 0
        
        batch_thgpu["lossMain"] = lossMain
        batch_thgpu["lossMask"] = lossMask
        batch_thgpu["lossSigmaTie"] = lossSigmaTie
        batch_thgpu["lossNormal3Tie"] = lossNormal3Tie
        batch_thgpu["lossMainHighlight"] = lossMainHighlight
        batch_thgpu["lossMaskHighlight"] = lossMaskHighlight
        batch_thgpu["lossSigmaTieHighlight"] = lossSigmaTieHighlight
        batch_thgpu["lossNormal3TieHighlight"] = lossNormal3TieHighlight
        batch_thgpu["lossFreeSigmaTie"] = lossFreeSigmaTie
        batch_thgpu["lossHighlightNdotH"] = lossHighlightNdotH
        batch_thgpu["lossSuppressHdr3"] = lossSuppressHdr3
        batch_thgpu["loss"] = (
            lossMain + lossMask + lossSigmaTie + lossNormal3Tie +  # lossTie + lossBackface +
            lossMainHighlight + lossMaskHighlight + lossSigmaTieHighlight + lossNormal3TieHighlight +  # lossTieHighlight + lossBackfaceHighlight +
            lossFreeSigmaTie +
            lossHighlightNdotH + lossSuppressHdr3  # + lossTie23 + lossTie23Highlight
        )
        # stat
        gt_rgb = torch.clamp(tonemap_srgb_to_rgb(gt_hdr), min=0, max=1)
        if results_hs:
            gt_rgb_highlight = torch.clamp(tonemap_srgb_to_rgb(
                batch_thgpu["highlightHdrs_%s" % dataset]), min=0, max=1)
            gt_rgb_sublight = torch.clamp(tonemap_srgb_to_rgb(
                batch_thgpu["sublightHdrs_%s" % dataset]), min=0, max=1)
        for coarseOrFine in ["fine"]:
            batch_thgpu["statPsnr%s" % bt(coarseOrFine)] = psnr(
                torch.clamp(tonemap_srgb_to_rgb(results[("hdr_%s" % coarseOrFine)]), min=0, max=1),
                gt_rgb
            )
            if results_hs:
                batch_thgpu["statPsnrHighlight%s" % bt(coarseOrFine)] = psnr(
                    torch.clamp(tonemap_srgb_to_rgb(results_hs["highlight"]["hdr_%s" % coarseOrFine]), min=0, max=1),
                    gt_rgb_highlight,
                )
                batch_thgpu["statPsnrSublight%s" % bt(coarseOrFine)] = psnr(
                    torch.clamp(tonemap_srgb_to_rgb(results_hs["sublight"]["hdr_%s" % coarseOrFine]), min=0, max=1),
                    gt_rgb_sublight,
                )
            batch_thgpu["statNumSampledPoints%s" % bt(coarseOrFine)] = (
                results["statNumSampledPoints_%s" % coarseOrFine][0]
            )
            batch_thgpu["statP2Rratio%s" % bt(coarseOrFine)] = (
                results["statP2Rratio_%s" % coarseOrFine][0]
            )
        return batch_thgpu

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

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        bg_color = rays[:, 10:13]

        outputs = models["renderer"].render(
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
            batch_head_index=kwargs["batch_head_index"],
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

    @classmethod
    def forwardOrthogonalDensity(cls, models, **kwargs):
        cls.assertModelEvalMode(models)
        chunk = kwargs["chunk"]
        # minBound = kwargs.get("minBound", np.array([-2, -2, -2], dtype=np.float32))
        # maxBound = kwargs.get("maxBound", np.array([2, 2, 2], dtype=np.float32))
        minBound = kwargs["minBound"]
        maxBound = kwargs["maxBound"]
        Lx = kwargs["Lx"]
        Ly = kwargs["Ly"]
        Lz = kwargs["Lz"]
        cudaDevice = kwargs["cudaDevice"]
        marching_cube_thre = 10  # this should the property of this approach.
        epsilon = 1.0e-5
        sCell = (maxBound - minBound) / np.array([Lx, Ly, Lz], dtype=np.float32)
        goxyz = minBound + 0.5 * sCell
        xi = np.linspace(goxyz[0], goxyz[0] + (Lx - 1) * sCell[0], Lx).astype(
            np.float32
        )
        yi = np.linspace(goxyz[1], goxyz[1] + (Ly - 1) * sCell[1], Ly).astype(
            np.float32
        )
        zi = np.linspace(goxyz[2], goxyz[2] + (Lz - 1) * sCell[2], Lz).astype(
            np.float32
        )
        x, y, z = np.meshgrid(xi, yi, zi)  # YXZ volume
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        xyz = np.stack([x, y, z], 1)
        xyz_thgpu = torch.from_numpy(xyz).to(cudaDevice)
        dir_chunk_thgpu = torch.zeros(chunk, 3, dtype=torch.float32, device=cudaDevice)
        batchTot = int(math.ceil(float(xyz.shape[0]) / float(chunk)))
        outFine = np.zeros((xyz.shape[0],), dtype=np.float32)
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=True):
            for b in range(batchTot):
                head = b * chunk
                tail = min((b + 1) * chunk, xyz.shape[0])

                xyz_chunk_thgpu = xyz_thgpu[head:tail]
                # t = models["model"].density(
                t = models["modelFast"].density(
                    xyz_chunk_thgpu, if_do_bound_clamp=True, if_output_only_sigma=True,
                    if_compute_normal3=False)
                outFine[head:tail] = t.detach().cpu().numpy()
        outFine = outFine.reshape((Ly, Lx, Lz))
        # marching cube
        if outFine.max() < marching_cube_thre:
            outFine[0, 0, 0] = marching_cube_thre + epsilon
        elif outFine.min() > marching_cube_thre:
            outFine[-1, -1, -1] = marching_cube_thre - epsilon
        fineVert0, fineFace0 = voxSdfSign2mesh_skmc(
            outFine, goxyz, sCell, level=marching_cube_thre
        )
        return {
            "fineVertWorld": fineVert0,
            "fineFace": fineFace0,
        }

    @classmethod
    def doPred0(cls, bsv0, **kwargs):
        datasetObj = kwargs["datasetObj"]
        datasetConf = datasetObj.datasetConf
        models = kwargs["models"]
        config = kwargs["config"]
        cudaDevice = kwargs["cudaDevice"]
        orthogonalDensityResolution = kwargs["orthogonalDensityResolution"]
        marching_cube_thre = kwargs["marching_cube_thre"]
        callFlag = kwargs["callFlag"]
        logDir = kwargs["logDir"]
        lgtMode = kwargs.get("lgtMode", "pointLightMode")
        cls.assertModelEvalMode(models)
        rays_thgpu = torch.from_numpy(bsv0["rays"]).to(cudaDevice)

        ifRequiresMesh = kwargs["ifRequiresMesh"]
        if_only_predict_hdr2 = kwargs["if_only_predict_hdr2"]

        with torch.no_grad():
            # chunk = 800
            chunk = 10000
            results_list = []
            
            if datasetConf["lgtLoadingMode"] == "ENVMAP":
                # Assumes all the envmaps samples are the same
                # forward the nerf appearance parameters just for once (to accelerate)

                hyperOutputs = None
                envmaps_reference = None
               
            t0 = time.time()
            for i in range(0, rays_thgpu.shape[0], chunk):
                if datasetConf["lgtLoadingMode"] == "QSPL":
                    lgtInput = {
                        k: torch.from_numpy(
                            bsv0[k][i : i + chunk, :]
                        ).to(rays_thgpu.device)
                        for k in ["lgtE", "objCentroid"]
                    }
                    hyperOutputs = None
                elif datasetConf["lgtLoadingMode"] == "ENVMAP":
                    setLgtRadius = 100.0
                    lgtInput = {
                        # "envmaps": torch.from_numpy(bsv0["envmapOriginalHighres"][None, :, :, :]).float().to(cudaDevice),
                        "lgtE": torch.from_numpy(-bsv0["envmapPointlightOmega"][None, :]).float().to(cudaDevice).repeat(chunk, 1) * setLgtRadius,
                        "objCentroid": torch.zeros(1, 3, dtype=torch.float32, device=cudaDevice).repeat(chunk, 1),
                    }
 
                else:
                    raise ValueError("Unknown lgtMode: %s" % lgtMode)
                # print("doPred0: %d / %d" % (i, rays_thgpu.shape[0]))
                results_list.append(cls.forwardRendering(
                    rays_thgpu[i:i + chunk],
                    models=models,
                    config=config,
                    iterCount=int(bsv0["iterCount"]),
                    cpu_ok=True,
                    force_all_rays=True,
                    requires_grad=False,  # debug the normal also for the test case
                    if_query_the_expected_depth=True,
                    lgtInput=lgtInput,
                    callFlag=callFlag,
                    logDir=logDir,
                    hyperOutputs=hyperOutputs,
                    lgtMode=lgtMode,
                    envmap0_thgpu=None if (lgtMode != "envmapMode") else (
                        torch.from_numpy(bsv0["envmapOriginalHighres"]).to(rays_thgpu.device)
                    ),
                    if_only_predict_hdr2=if_only_predict_hdr2,
                    batch_head_index=i,  # test time debugging purpose
                ))
            t1 = time.time()
            bsv0["timeElapsed"] = t1 - t0
            bsv0["timeEvalMachine"] = gethostname()
            results = {k: torch.cat([x[k] for x in results_list], 0) for k in results_list[0].keys()}
        for k in results.keys():
            assert "float32" in str(results[k].dtype), k
        winWidth = int(bsv0["winWidth"])
        winHeight = int(bsv0["winHeight"])
        if "hdr_fine" in results.keys():
            hdrFinePred = results["hdr_fine"].detach().cpu().numpy()
            bsv0["imghdrFinePred"] = hdrFinePred  # .reshape((winHeight, winWidth, 3))
            ldrFinePred = torch.clamp(tonemap_srgb_to_rgb(results["hdr_fine"].detach()), min=0, max=1).cpu().numpy()
            bsv0["imgFinePred"] = ldrFinePred  # .reshape((winHeight, winWidth, 3))
            bsv0["ldrFinePred"] = ldrFinePred  # benchmarking needs this one
        if "hdr2_fine" in results.keys():
            bsv0["imghdr2FinePred"] = results["hdr2_fine"].detach().cpu().numpy()  # .reshape((winHeight, winWidth, 3))
        if "hdr3_fine" in results.keys():
            bsv0["imghdr3FinePred"] = results["hdr3_fine"].detach().cpu().numpy()  # .reshape((winHeight, winWidth, 3))

        if "depth_fine" in results.keys():
            bsv0["depthFinePred"] = (
                results["depth_fine"].detach().cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
            )
        if "opacity_fine" in results.keys():
            bsv0["opacityFinePred"] = (
                results["opacity_fine"].detach().cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
            )
        if "normal2_fine" in results.keys():
            bsv0["normal2"] = results["normal2_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "normal3_fine" in results.keys():
            bsv0["normal3"] = results["normal3_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "normal2DotH_fine" in results.keys():
            bsv0["normal2DotH"] = results["normal2DotH_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "normal3DotH_fine" in results.keys():
            bsv0["normal3DotH"] = results["normal3DotH_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsPointlightOpacities_fine" in results.keys():
            bsv0["hintsPointlightOpacities"] = results["hintsPointlightOpacities_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        for i in range(4):
            if ("hintsPointlightGGX%d_fine" % i) in results.keys():
                bsv0["hintsPointlightGGX%d" % i] = results["hintsPointlightGGX%d_fine" % i].detach(
                    ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefOpacities_fine" in results.keys():
            bsv0["hintsRefOpacities"] = results["hintsRefOpacities_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefSelf_fine" in results.keys():
            bsv0["hintsRefSelf"] = results["hintsRefSelf_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "hintsRefLevels_fine" in results.keys():
            bsv0["hintsRefLevels"] = results["hintsRefLevels_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, -1)).astype(np.float32)
        if "hintsRefLevelsColor_fine" in results.keys():
            bsv0["hintsRefLevelsColor"] = results["hintsRefLevelsColor_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # reshape((winHeight, winWidth, -1, 3)).astype(np.float32)
        if "hintsRefEnv_fine" in results.keys():
            bsv0["hintsRefEnv"] = results["hintsRefEnv_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefDistribute_fine" in results.keys():
            bsv0["hintsRefDistribute"] = results["hintsRefDistribute_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, -1)).astype(np.float32)

        if ifRequiresMesh:
            # raise ValueError("We disable this branch for now.")
            if orthogonalDensityResolution > 0:
                tmp = cls.forwardOrthogonalDensity(
                    models,
                    Lx=orthogonalDensityResolution,
                    Ly=orthogonalDensityResolution,
                    Lz=orthogonalDensityResolution,
                    cudaDevice=cudaDevice,
                    chunk=config.chunk,
                    marching_cube_thre=marching_cube_thre,
                    minBound=datasetObj.datasetConf["minBound"],
                    maxBound=datasetObj.datasetConf["maxBound"],
                )
                for meshName in ["fine"]:
                    for k in ["VertWorld", "Face"]:
                        bsv0[meshName + k] = tmp[meshName + k]
        return bsv0


def returnExportedClasses(
    wishedClassNameList,
):

    exportedClasses = {}

    if wishedClassNameList is None or "DTrainer" in wishedClassNameList:
        exportedClasses["DTrainer"] = DTrainer

    return exportedClasses
