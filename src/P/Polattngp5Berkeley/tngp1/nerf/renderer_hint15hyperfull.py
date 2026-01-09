import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import grid_sample

from functools import partial

from codes_py.toolbox_nerfacc.volrend import render_weight_from_density, accumulate_along_rays

from .. import raymarching
from .utils import custom_meshgrid


def get_render_step_size(iterCount, render_step_size_keys, render_step_size_vals):
    assert render_step_size_keys[0] == 0
    assert iterCount >= 0
    b = np.where(iterCount >= render_step_size_keys)[0][-1]
    return render_step_size_vals[b]


def soft_clamp(x, s_left, s_right, shrinkage): 
    assert type(s_left) is float
    assert type(s_right) is float
    assert type(shrinkage) is float
    assert s_left <= s_right
    val_mid = torch.clamp(x, min=s_left, max=s_right)
    val_left = torch.clamp(x - s_left, max=0) * shrinkage
    val_right = torch.clamp(x - s_right, min=0) * shrinkage
    val = val_mid + val_left + val_right
    return val


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=True,  # original default val is False
                 enable_refnerf=True,  # original default val is False
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=10,  # original default val is 0.01
                 bg_radius=-1,
                 **kwargs,
                 ):
        super().__init__()

        config = kwargs["config"]
        self.config = config

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        self.empty_cache_stop_iter = config.empty_cache_stop_iter
        self.render_step_size_keys = config.render_step_size_keys
        self.render_step_size_vals = config.render_step_size_vals

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
        self.enable_refnerf = enable_refnerf

        # set up the gaussian kernel
        self.setup_gaussian_kernel()

    def setup_gaussian_kernel(self):
        # self.gauKSizeList = np.array([0, 7, 15, 31, 51, 101], dtype=np.int32)
        config = self.config
        self.gauKSizeList = config.pyramid_ksize_list
        self.gauKSigmaList = (self.gauKSizeList.astype(np.float32) - 1) / 6.0
        self.gauL = int(self.gauKSizeList.shape[0])
        self.gauHSum = []
        self.envmapWidth = 1024
        self.envmapHeight = 512
        self.ratioRadiusOverPixel = float(np.pi / self.envmapHeight)  # computed from the height. Width should be the same.
        for l in range(self.gauL):
            # if l == 0:
            if self.gauKSizeList[l] == 0:
                self.gauHSum.append(None)  # (self.gauKSizeList[l] == 0) should always be separately processed
            else:
                gauKSize = self.gauKSizeList[l]
                gauKSigma = self.gauKSigmaList[l]
                xi = np.linspace(-(gauKSize // 2), gauKSize // 2, gauKSize).astype(np.int32)
                H_pred_unnormalized = np.exp(-(xi ** 2) / (2 * (gauKSigma ** 2))) / (2 * np.pi * (gauKSigma ** 2))
                H_pred_sum = H_pred_unnormalized.sum()
                self.gauHSum.append(H_pred_sum)  
    
    @staticmethod
    def do_reflection_surface_cutoff(ref_ray_indices, ref_t_starts, **kwargs):
        reflection_surface_cutoff = kwargs["reflection_surface_cutoff"]
        render_step_size = kwargs["render_step_size"]

        """
        if (
            (ref_ray_indices.shape == (1,)) and (ref_ray_indices[0] == 0) and 
            (ref_t_starts.shape == (1,)) and (ref_t_starts[0] == 0.001)
        ):
            import ipdb
            ipdb.set_trace()
            print(1 + 1)
        """

        assert ref_ray_indices.shape == ref_t_starts.shape
        assert len(ref_ray_indices.shape) == 1

        if (ref_ray_indices.shape[0] == 1):
            flag_keep = ref_t_starts > reflection_surface_cutoff

        else:
            ref_ray_indices_lead_locations_flag = (ref_ray_indices[1:] != ref_ray_indices[:-1])
            ref_ray_indices_lead_locations_flag = torch.cat([
                torch.ones_like(ref_ray_indices_lead_locations_flag[:1]),
                ref_ray_indices_lead_locations_flag,
            ], 0)
            ref_ray_indices_lead_locations = torch.where(ref_ray_indices_lead_locations_flag)[0]
            locations_of_their_lead = ref_ray_indices_lead_locations[
                torch.cumsum(ref_ray_indices_lead_locations_flag.int(), dim=0) - 1]
            locations = torch.arange(ref_t_starts.shape[0], dtype=torch.int32, device=ref_t_starts.device)
            flag_just_surface = (  # these sampled segments are going to be deleted
                (ref_t_starts - ref_t_starts[locations_of_their_lead] ==
                 render_step_size * (locations - locations_of_their_lead))
                &
                (
                    ref_t_starts <= reflection_surface_cutoff
                )
            )
            flag_keep = flag_just_surface == 0

        return flag_keep

    def run_cuda(self, rays_o, rays_d, models, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]
        lgtMode = kwargs["lgtMode"]
        recursive_depth = kwargs["recursive_depth"]
        if_only_predict_hdr2 = kwargs["if_only_predict_hdr2"]
        callFlag = kwargs["callFlag"]

        occGrid = models["occGrid"]
        model17 = models["model17"]
        if_add_in_hdr3 = models["meta"]["if_add_in_hdr3"]

        specRoughnessID = None  # change to kwargs when doing the demo

        config = self.config

        lgtInputRays = kwargs["lgtInputRays"]
        envmap0 = kwargs["envmap0"]
        envmapNormalizingFactor = kwargs["envmapNormalizingFactor"]
        batch_head_ray_index = kwargs.get("batch_head_ray_index", None)

        min_near = kwargs["min_near"]
        if_do_reflection_surface_cutoff = kwargs["if_do_reflection_surface_cutoff"]

        chunk = 1024 * 32  # will be useful if you are to do refnerf (where gpu memories are not sufficient)

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        iterCount = kwargs["iterCount"]

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(
            rays_o,
            rays_d,
            self.aabb_train if self.training else self.aabb_infer,
            min_near,
        )

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = model17.density(
                positions,
                if_do_bound_clamp=True,
                if_output_only_sigma=True,
                if_compute_normal3=False,
            ) * self.density_scale
            return sigmas

        render_step_size = get_render_step_size(
            iterCount, self.render_step_size_keys, self.render_step_size_vals)
        ray_indices, t_starts, t_ends = occGrid.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            near_plane=nears.min(), far_plane=(fars[fars < 1000].max() if torch.any(fars < 1000) else 10 * nears.min()),
            early_stop_eps=1e-4, alpha_thre=1e-5,
            render_step_size=render_step_size,
        )

        if iterCount < self.empty_cache_stop_iter:
            torch.cuda.empty_cache()

        # avoid empty error
        if ray_indices.shape[0] == 0:
            ray_indices = torch.cuda.LongTensor([0], device=rays_o.device)
            t_starts = torch.cuda.FloatTensor([nears[0] if (nears[0] < 1000) else 2], device=rays_o.device)
            t_ends = torch.cuda.FloatTensor([fars[0] if (fars[0] < 1000) else 60], device=rays_o.device)

        # flag_keep (sometimes we need to remove just-surface sampled point for in-scene cameras)
        if if_do_reflection_surface_cutoff:
            flag_keep = self.do_reflection_surface_cutoff(
                ray_indices, t_starts,
                reflection_surface_cutoff=config.reflection_surface_cutoff,
                render_step_size=render_step_size,               
            )
        else:
            flag_keep = np.ones((ray_indices.shape[0],), dtype=bool)
        ray_indices = ray_indices[flag_keep]
        t_starts = t_starts[flag_keep]
        t_ends = t_ends[flag_keep]

        # avoid empty error once again
        if ray_indices.shape[0] == 0:
            ray_indices = torch.cuda.LongTensor([0], device=rays_o.device)
            t_starts = torch.cuda.FloatTensor([nears[0] if (nears[0] < 1000) else 2], device=rays_o.device)
            t_ends = torch.cuda.FloatTensor([fars[0] if (fars[0] < 1000) else 60], device=rays_o.device)

        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        xyzs = positions
        dirs = t_dirs

        # forward hyper
        hyperPredictionDict = kwargs["hyperPredictionDict"]
        # You are acknowledging that fg rays are first, then followed by mg/bg.
        # For mg/bg rays, we randomly assign an envmap predicted hyper params to it.
        insideBatchInd_fg_lgt = lgtInputRays["insideBatchInd_lgt"]
        num_rays_mg_bg = int(rays_o.shape[0]) - int(insideBatchInd_fg_lgt.shape[0])
        insideBatchInd_mgbg_lgt = torch.randint(
            low=0, high=int(lgtInputRays["envmapAlphaed"].shape[0]), size=(num_rays_mg_bg,),
            dtype=insideBatchInd_fg_lgt.dtype, 
            device=insideBatchInd_fg_lgt.device,
        )
        insideBatchInd_fgmgbg_lgt = torch.cat([
            insideBatchInd_fg_lgt,
            insideBatchInd_mgbg_lgt,
        ], 0)
        hyperPredictionDict["insideBatchInd_fgmgbg_point_lgt"] = insideBatchInd_fgmgbg_lgt[ray_indices]

        assert self.enable_refnerf
        assert iterCount >= self.empty_cache_stop_iter

        preAppearancePredictionDict = model17.preAppearance(
            xyzs,
            if_do_bound_clamp=True,
            requires_grad=False,  # requires_grad,
            iterCount=iterCount,
        )

        sigmas = preAppearancePredictionDict["out_tau_spatial"] * self.density_scale

        weights, _, _ = render_weight_from_density(
            t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=int(rays_o.shape[0]),
        )
        opacities = accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=int(rays_o.shape[0]))[:, 0]
        # depths: this is euclidean depth, rather than zbuffer depth
        depths = accumulate_along_rays(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )[:, 0] / opacities.clamp_min(torch.finfo(opacities.dtype).eps) 

        # hdr2
        fullPredictionDict = model17.appearance(
            xyzs, dirs, preAppearancePredictionDict,
            hyperPredictionDict=hyperPredictionDict,
            iterCount=iterCount,
        )
        hdr2 = accumulate_along_rays(
            weights,
            values=fullPredictionDict["out_hdr2"],
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )
        if (if_only_predict_hdr2) and (recursive_depth == 0):
            return {"hdr2": hdr2}

        # Reflection Hint
        with torch.no_grad():
            if (callFlag != "forwardNetNeRF") and (config.if_highlight_case):
                normal3_npp = F.normalize(accumulate_along_rays(
                    weights,
                    values=preAppearancePredictionDict["out_normal3_npp"],
                    ray_indices=ray_indices,
                    n_rays=int(rays_o.shape[0]),
                ), dim=1, p=2)

                valid_mask = opacities >= 0.99
                valid_mask[depths < torch.finfo(torch.float32).eps] = False
                valid_mask[torch.norm(normal3_npp, p=2, dim=1) < 0.9] = False

                valid_mask_3 = valid_mask[:, None].repeat(1, 3)
                valid_ind = torch.where(valid_mask)[0]
                valid_ind_3 = valid_ind[:, None].repeat(1, 3)
                ref_point = torch.where(
                    valid_mask_3,
                    rays_o + depths[:, None] * rays_d,  # depths is euclidean depth
                    0,
                ) 
                
                omega_o = -rays_d
                omega_r = (
                    2.0
                    * (omega_o * normal3_npp).sum(1)[:, None]
                    * normal3_npp
                    - omega_o
                )
                omega_r = torch.where(valid_mask_3, omega_r, 0)
                assert torch.all(torch.abs(torch.norm(omega_r[valid_mask_3].view(-1, 3), p=2, dim=1) - 1) < 1.0e-3)
                ref_rays_o, ref_rays_d = ref_point, omega_r

                if (recursive_depth == 0) and (valid_mask.sum() > 0):
                    ref_out_masked = self.run_cuda(
                        ref_rays_o[valid_mask, :], ref_rays_d[valid_mask, :], models=models, 
                        iterCount=iterCount,
                        # lgtInputRays={k: v[valid_mask, :] for k, v in lgtInputRays.items()},
                        lgtInputRays={
                            "envmapAlphaed": lgtInputRays["envmapAlphaed"],
                            "insideBatchInd_lgt": lgtInputRays["insideBatchInd_lgt"][valid_ind],
                        },
                        lgtMode=lgtMode,
                        envmap0=envmap0,
                        envmapNormalizingFactor=envmapNormalizingFactor,  # this won't be used for the low-freq branch, but for the sake of completness of the code base, we still input this useless value
                        envmapFiltered=kwargs["envmapFiltered"],
                        hyperPredictionDict=kwargs["hyperPredictionDict"],
                        batch_head_ray_index=batch_head_ray_index,
                        recursive_depth=recursive_depth + 1,
                        min_near=0.001,
                        if_do_reflection_surface_cutoff=True,
                        if_only_predict_hdr2=False,
                        callFlag=callFlag,
                    )
                    ref_out = {
                        "opacity": torch.zeros(
                            valid_mask.shape[0], device=valid_mask.device, dtype=torch.float32,
                        ).scatter_(dim=0, index=valid_ind, src=ref_out_masked["opacity"]),
                        "hdr": torch.zeros(
                            valid_mask.shape[0], 3, device=valid_mask.device, dtype=torch.float32,
                        ).scatter_(dim=0, index=valid_ind_3, src=ref_out_masked["hdr"]),
                    }
                else:
                    ref_out = {
                        "opacity": torch.zeros(
                            valid_mask.shape[0], device=valid_mask.device, dtype=torch.float32,
                        ),
                        "hdr": torch.zeros(
                            valid_mask.shape[0], 3, device=valid_mask.device, dtype=torch.float32,
                        ),
                    }

                # hdr3 (no grad)
                hints_reflection_hdrs = ref_out["hdr"]
                hints_reflection_opacities = ref_out["opacity"]
                if False:  # lgtMode == "pointLightMode":
                    cos_omega_local_omega_r = (omega_r * omega_local).sum(1)
                    assert torch.all(torch.abs(cos_omega_local_omega_r) <= 1.001), torch.abs(cos_omega_local_omega_r).max()
                    cos_omega_local_omega_r = torch.clamp(cos_omega_local_omega_r, min=-1, max=1)
                    rad_omega_local_omega_r = torch.acos(cos_omega_local_omega_r)
                    envmap_pix_distance = rad_omega_local_omega_r / self.ratioRadiusOverPixel
                    assert torch.all(envmap_pix_distance >= 0), (
                        envmap_pix_distance.min(),
                        envmap_pix_distance.max(),
                        rad_omega_local_omega_r_rays.min(),
                        rad_omega_local_omega_r_rays.max(),
                        cos_omega_local_omega_r_rays.max(),
                        cos_omega_local_omega_r_rays.min(),
                    )
                    hint_ref_levels = []
                    for l in range(self.gauL):
                        if self.gauKSizeList[l] == 0:
                            hint_l = (envmap_pix_distance < 2).float()
                        else:
                            gauKSize = self.gauKSizeList[l]
                            gauKSigma = self.gauKSigmaList[l]
                            gauHSum = self.gauHSum[l]
                            # normalized with the middle one
                            hint_l = torch.exp(
                                (-(envmap_pix_distance ** 2))
                                / 
                                (2 * (gauKSigma ** 2))
                            )
                            hint_l[envmap_pix_distance * 2 + 1 >= gauKSize] = 0

                        assert len(hint_l.shape) == 1
                        hint_l[valid_mask == 0] = 0
                        hint_ref_levels.append(hint_l)

                    hint_ref_levels = torch.stack(hint_ref_levels, 1)
        
                elif True:  # lgtMode == "envmapMode":
                    envmapFiltered = kwargs["envmapFiltered"]
                    # 2. grid_sample according to omega_r
                    phi = torch.atan2(omega_r[:, 1], omega_r[:, 0])
                    # phi_ = phi / torch.pi  # map from [-pi, pi] to [-1, 1]
                    phi = torch.pi - phi  # blender convention. Now phi's range becomes [0, 2 * np.pi]
                    phi_ = phi / torch.pi - 1  # map from [0, 2 * np.pi] to [-1, 1]

                    assert torch.all(torch.abs(torch.abs(omega_r[:, 2])) < 1.00001)
                    theta = torch.acos(torch.clamp(omega_r[:, 2], min=-1, max=1))
                    theta_ = theta / torch.pi * 2 - 1  # map from [0, pi] to [-1, 1] 
                    tmp = grid_sample(
                        envmapFiltered.view(1, self.gauL * 3, envmap0.shape[0], envmap0.shape[1]),
                        torch.stack([phi_, theta_], 1)[None, :, None, :],
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    sampled_envmap_rgb = tmp[0, :, :, 0].permute(1, 0).view(omega_r.shape[0], self.gauL, 3)
                    hint_ref_levels_color_distribution = sampled_envmap_rgb[:, 1, :]  # use this to re-distribute colors
                    hint_ref_levels = sampled_envmap_rgb.mean(2)
                    hint_ref_levels[valid_mask == 0, :] = 0
                    hint_ref_levels = torch.clamp(hint_ref_levels, min=0, max=1)
                else:
                    raise ValueError("Unknown lgtMode %s" % lgtMode)

                assert torch.all(ref_out["opacity"][valid_mask == 0] == 0)
                assert torch.all(ref_out["hdr"][valid_mask == 0, :] == 0)
                if config.if_disable_hint_ref_hdr:
                    ref_out["hdr"][:] = 0  # a temporary solution
                tmp = model17.normal3_envmap(
                    torch.cat([
                        model17.embeddingHintPointlightOpacities(ref_out["opacity"][:, None]),  # 9
                        ref_out["hdr"],  # 3
                        hint_ref_levels,  # 6
                        model17.embeddingXyz(ref_point),  # 63
                    ], 1)
                )
                hdr3 = self.safe_exp(tmp, 1, 10).float() * opacities[:, None]

                if False:  # lgtMode == "pointLightMode":
                    pass  # do nothing
                elif True:  # lgtMode == "envmapMode":
                    s = hint_ref_levels_color_distribution / (hint_ref_levels_color_distribution.sum(1)[:, None] + 0.00001) * 3
                    hdr3 *= s
                else:
                    raise ValueError("Unknown lgtMode %s" % lgtMode)
            
            else:
                hdr3 = torch.zeros_like(hdr2)

        assert len(opacities.shape) == 1
        assert len(depths.shape) == 1
        results = {}
        results["rawhdr2"] = hdr2.detach().clone()
        if callFlag != "forwardNetNeRF":
            # testing mode, alpha is all 1.
            assert torch.all(torch.abs(lgtInputRays["envmapAlphaed"].view(-1, 16 * 32 * 3).mean(1) - 1.0 / 3.0 < 1e-4))
            hdr = (hdr2 / config.hyperHdrScaling * (envmapNormalizingFactor / 16.0 / 32.0 * 1) + hdr3 / 3) * 3  # hdr3 divided by 3: hdr3 (high-freq) was trained from grayscale point light. When applied to the RGB envmap, it needs to be renormalized to match hdr2 (low-freq)
            results["hdr2"] = hdr2 / config.hyperHdrScaling * (envmapNormalizingFactor / 16.0 / 32.0 * 1)  # demo
        else:
            hdr = hdr2
            results["hdr2"] = hdr2  # training

        results["hdr"] = hdr
        if (callFlag != "forwardNetNeRF") and (config.if_highlight_case):
            results["hdr3"] = hdr3
            results["normal3"] = normal3_npp
        results["depth"] = depths
        results["opacity"] = opacities

        # hints
        if (callFlag != "forwardNetNeRF") and (config.if_highlight_case):
            results["hintsRefOpacities"] = ref_out["opacity"]
            results["hintsRefSelf"] = ref_out["hdr"]
            results["hintsRefLevels"] = hint_ref_levels
            results["hintsRefLevelsColor"] = hint_ref_levels[:, :, None] * s[:, None, :]
            results["hintsRefColor"] = torch.clamp(s * 0.5, min=0, max=1)

        # sampling stats
        results["statNumSampledPoints"] = torch.cuda.FloatTensor([int(xyzs.shape[0])])
        assert results["statNumSampledPoints"].device == xyzs.device
        results["statP2Rratio"] = torch.cuda.FloatTensor([float(xyzs.shape[0]) / float(rays_o.shape[0])])
        assert results["statP2Rratio"].device == xyzs.device

        return results

    @staticmethod
    def safe_exp(x, shift, thre):  # the classic setting: shift is 1, and thre is 10
        x = x - shift
        out = torch.where(
            x <= thre,
            torch.exp(torch.clamp(x, max=thre)),
            float(math.e ** thre + 0.01) + (x - thre),
        )
        return out

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_hdr: [B, N, 3]

        if self.cuda_ray:
            """
            if self.enable_refnerf:
                _run = self.run_cuda_refnerf
            else:
                _run = self.run_cuda
            """
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results
