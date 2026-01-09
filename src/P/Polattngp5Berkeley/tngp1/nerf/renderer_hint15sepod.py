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


def surface_eye(ref_rays_o, ref_rays_d, valid_mask, **kwargs):
    # suggest run under the with torch.no_grad() context
    # All naming conventions have the ref_ prefix, seems to be representing the reflection direction (omega_r)
    # Actually it can be other directions (e.g. omega_local if it is point light)

    # Every ray only shoot for one surface_eye ray

    # ref_rays_o (B, 3)
    # ref_rays_d (B, 3)
    # valid_mask (B, )

    B = int(ref_rays_o.shape[0])
    assert ref_rays_o.shape == (B, 3)
    assert ref_rays_d.shape == (B, 3)
    assert valid_mask.shape == (B,)
    assert ref_rays_o.dtype == torch.float32
    assert ref_rays_d.dtype == torch.float32
    assert valid_mask.dtype == torch.bool

    renderer = kwargs["renderer"]
    modelRef = kwargs["modelRef"]
    occGrid = kwargs["occGrid"]
    config = kwargs["config"]
    requesting_list = kwargs["requesting_list"]
    iterCount = kwargs["iterCount"]
    lgtInputRays = kwargs["lgtInputRays"]

    out = {}

    ref_nears, ref_fars = raymarching.near_far_from_aabb(
        ref_rays_o,
        ref_rays_d,
        renderer.aabb_train,
        0.001,  # model.min_near,
    )
    ref_nears[valid_mask == 0] = 0
    ref_fars[valid_mask == 0] = 0

    if valid_mask.sum() > 0:
        def sigma_fn_ref_points(ref_t_starts, ref_t_ends, ref_ray_indices):
            ref_t_origins = ref_rays_o[valid_mask, :][ref_ray_indices]
            ref_t_dirs = ref_rays_d[valid_mask, :][ref_ray_indices]
            ref_positions = ref_t_origins + ref_t_dirs * (ref_t_starts + ref_t_ends)[:, None] / 2.0
            ref_sigmas = modelRef.density(
                ref_positions, if_do_bound_clamp=True, if_output_only_sigma=True, if_compute_normal3=False,
            ) * renderer.density_scale
            return ref_sigmas

        render_step_size = get_render_step_size(
            iterCount, renderer.render_step_size_keys, renderer.render_step_size_vals)
        ref_ray_indices_, ref_t_starts, ref_t_ends = occGrid.sampling(
            ref_rays_o[valid_mask, :], ref_rays_d[valid_mask, :],
            sigma_fn=sigma_fn_ref_points,
            near_plane=ref_nears[valid_mask].min(),
            far_plane=(
                ref_fars[valid_mask][ref_fars[valid_mask] < 1000].max()
                if torch.any(ref_fars[valid_mask] < 1000) else 10 * ref_nears[valid_mask].min()
            ),
            early_stop_eps=1e-4,
            alpha_thre=1e-5,
            render_step_size=render_step_size,
        )
        ref_ray_indices = torch.where(valid_mask)[0][ref_ray_indices_]
        assert torch.all(valid_mask[torch.unique(ref_ray_indices)])

        # automaticaly remove the just-surface density
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
                ref_t_starts <= 0.025 * 10000  # config.reflection_surface_cutoff
            )
        )
        flag_keep = flag_just_surface == 0
        if (ref_ray_indices.shape[0] > 0) and (flag_keep.sum() > 0):
            ref_ray_indices = ref_ray_indices[flag_keep]
            ref_t_starts = ref_t_starts[flag_keep]
            ref_t_ends = ref_t_ends[flag_keep]

            ref_t_origins = ref_rays_o[ref_ray_indices]
            ref_t_dirs = ref_rays_d[ref_ray_indices]
            ref_positions = ref_t_origins + ref_t_dirs * (ref_t_starts + ref_t_ends)[:, None] / 2.0
            ref_lgtInputPoints = {k: lgtInputRays[k][ref_ray_indices] for k in lgtInputRays}
            tmp = modelRef(
                ref_positions, ref_t_dirs,
                if_do_bound_clamp=True, requires_grad=False, lgtInputPoints=ref_lgtInputPoints,
                hints_pointlight_opacities=torch.zeros(ref_positions.shape[0], 1, dtype=torch.float32, device=ref_positions.device),
                iterCount=iterCount,
            )
            ref_sigmas = tmp["out_tau_spatial"]
            ref_hdr = tmp["out_hdr"]
            ref_weights, _, _ = render_weight_from_density(
                ref_t_starts, ref_t_ends, ref_sigmas, ray_indices=ref_ray_indices, n_rays=(ref_rays_o.shape[0]),
            )
            if "opacities" in requesting_list:
                ref_opacities = accumulate_along_rays(
                    ref_weights,
                    values=None,
                    ray_indices=ref_ray_indices,
                    n_rays=int(ref_rays_o.shape[0]),
                )
                out["opacities"] = ref_opacities
            if "hdr" in requesting_list:
                ref_hdr = accumulate_along_rays(
                    ref_weights,
                    values=ref_hdr,
                    ray_indices=ref_ray_indices,
                    n_rays=int(ref_rays_o.shape[0]),
                )
                out["hdr"] = ref_hdr

        else:
            if "opacities" in requesting_list:
                out["opacities"] = torch.zeros_like(ref_rays_o[:, :1])
            if "hdr" in requesting_list:
                out["hdr"] = torch.zeros_like(ref_rays_o)
    else:
        if "opacities" in requesting_list:
            out["opacities"] = torch.zeros_like(ref_rays_o[:, :1])
        if "hdr" in requesting_list:
            out["hdr"] = torch.zeros_like(ref_rays_o)

    return out


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 enable_refnerf=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
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
    
    def run_cuda(self, rays_o, rays_d, models, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        occGrid = models["occGrid"]
        modelSlow = models["modelSlow"]
        modelFast = models["modelFast"]

        config = self.config

        lgtInput = kwargs["lgtInput"]
        lgtMode = kwargs["lgtMode"]
        envmap0 = kwargs["envmap0"]

        chunk = 1024 * 32  # will be useful if you are to do refnerf (where gpu memories are not sufficient)

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        iterCount = kwargs["iterCount"]
        requires_grad = kwargs["requires_grad"]
        if_query_the_expected_depth = kwargs["if_query_the_expected_depth"]  # generally, true only if at test time (training time does not need this)
        callFlag = kwargs["callFlag"]

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        # setup counter
        counter = self.step_counter[self.local_step % 16]
        counter.zero_() # set to 0
        self.local_step += 1

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = modelFast.density(
                positions,
                if_do_bound_clamp=True,
                if_output_only_sigma=True,
                if_compute_normal3=False,
            ) * self.density_scale
            return sigmas

        ray_indices, t_starts, t_ends = occGrid.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            near_plane=nears.min(), far_plane=(fars[fars < 1000].max() if torch.any(fars < 1000) else 10 * nears.min()),
            early_stop_eps=1e-4, alpha_thre=1e-5,
            render_step_size=get_render_step_size(iterCount, self.render_step_size_keys, self.render_step_size_vals)
        )
        if iterCount < self.empty_cache_stop_iter:
            torch.cuda.empty_cache()
        if ray_indices.shape[0] == 0:  # avoid the empty error
            ray_indices = torch.cuda.LongTensor([0], device=rays_o.device)
            t_starts = torch.cuda.FloatTensor([nears[0] if (nears[0] < 1000) else 2], device=rays_o.device)
            t_ends = torch.cuda.FloatTensor([fars[0] if (fars[0] < 1000) else 60], device=rays_o.device)
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        xyzs = positions
        dirs = t_dirs
        
        lgtInputRays = lgtInput
        lgtInputPoints = {k: lgtInputRays[k][ray_indices] for k in lgtInputRays if k in ["lgtE", "objCentroid"]}
        
        assert self.enable_refnerf
        assert iterCount >= self.empty_cache_stop_iter

        with torch.no_grad():
            if config.finetuningPDSRI_slow["P"] in ["Polattngp5Berkeley"]:
                preAppearancePredictionDict_slow = modelSlow.preAppearance(
                    xyzs,
                    if_do_bound_clamp=True,
                    requires_grad=False,  # requires_grad,
                    lgtInputPoints=lgtInputPoints,
                    iterCount=iterCount,
                )
            else:
                raise NotImplementedError("Unknown config.finetuningPDSRI_slow: " + str(config.finetuningPDSRI_slow))
        preAppearancePredictionDict_fast = modelFast.preAppearance(
            xyzs,
            if_do_bound_clamp=True,
            requires_grad=False,  # requires_grad,
            lgtInputPoints=lgtInputPoints,
            iterCount=iterCount,
        )
        sigma_before_activation_slow = preAppearancePredictionDict_slow["sigma_before_activation"]
        sigma_before_activation_fast = preAppearancePredictionDict_fast["sigma_before_activation"]
        sigmas_fast = preAppearancePredictionDict_fast["out_tau_spatial"] * self.density_scale
        loss_sigma_tie = torch.abs(
            soft_clamp(sigma_before_activation_slow, s_left=float(-10), s_right=float(10), shrinkage=float(0.001))
            -
            soft_clamp(sigma_before_activation_fast, s_left=float(-10), s_right=float(10), shrinkage=float(0.001))
        )
        weights, _, _ = render_weight_from_density(
            t_starts, t_ends, sigmas_fast, ray_indices=ray_indices, n_rays=int(rays_o.shape[0]),
        )
        opacities = accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=int(rays_o.shape[0]))[:, 0]
        depths = accumulate_along_rays(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )[:, 0] / opacities.clamp_min(torch.finfo(opacities.dtype).eps)
        if ("lossNormal3Tie" in config.wl.keys()) and (config.wl["lossNormal3Tie"] > 0):
            loss_normal3_tie_per_point = ((
                preAppearancePredictionDict_fast["out_normal3_npp"] -
                preAppearancePredictionDict_slow["out_normal3_npp"]
            ) ** 2).sum(1)
            loss_normal3_tie = accumulate_along_rays(
                weights, values=loss_normal3_tie_per_point[:, None], ray_indices=ray_indices, n_rays=int(rays_o.shape[0]),
            )[:, 0]

        normal3_npp = F.normalize(accumulate_along_rays(
            weights,
            values=preAppearancePredictionDict_fast["out_normal3_npp"],
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        ), dim=1, p=2)
        with torch.no_grad():
            # appearance

            # First order reflection
            valid_mask = opacities >= 0.99
            valid_mask[depths < torch.finfo(torch.float32).eps] = False
            valid_mask[torch.norm(normal3_npp, p=2, dim=1) < 0.9] = False
            
            valid_mask_3 = valid_mask[:, None].repeat(1, 3)
            ref_point = torch.where(
                valid_mask_3,
                rays_o + depths[:, None] * rays_d,  # depths is euclidean depth
                0,
            )

            assert torch.all(torch.abs(torch.norm(rays_d, p=2, dim=1) - 1) < 1.0e-3)
            if not torch.all(torch.abs(torch.norm(normal3_npp[valid_mask_3].view(-1, 3), p=2, dim=1) - 1) < 1.0e-3):
                import ipdb
                ipdb.set_trace()
                print(1 + 1)
            assert torch.all(torch.abs(torch.norm(normal3_npp[valid_mask_3].view(-1, 3), p=2, dim=1) - 1) < 1.0e-3)
            omega_o = -rays_d
            # Use normal3
            omega_r = (
                2.0
                * (omega_o * normal3_npp).sum(1)[:, None]
                * normal3_npp
                - omega_o
            )

            omega_r = torch.where(valid_mask_3, omega_r, 0)
            assert torch.all(torch.abs(torch.norm(omega_r[valid_mask_3].view(-1, 3), p=2, dim=1) - 1) < 1.0e-3)
            ref_rays_o, ref_rays_d = ref_point, omega_r

            ref_out = surface_eye(
                ref_rays_o, ref_rays_d, valid_mask,
                renderer=self,
                modelRef=modelFast,
                occGrid=occGrid,
                requesting_list=["opacities", "hdr"],
                iterCount=iterCount,
                config=config,
                lgtInputRays=lgtInputRays,
            )

            # prepare for the hints_from_the_ref
            # normal3_npp is already available here
            if lgtMode == "pointLightMode":
                # something to pay attention to on the ref_out["hdr"]
                # under this lgtMode, there should be no nans
                assert torch.all(torch.isfinite(ref_out["hdr"]))

                lgtE_rays = lgtInputRays["lgtE"]
                omega_local_rays = F.normalize(lgtE_rays - ref_point, p=2, dim=1)
                cos_omega_local_omega_r_rays = (omega_r * omega_local_rays).sum(1)
                assert torch.all(torch.abs(cos_omega_local_omega_r_rays) <= 1.0001), cos_omega_local_omega_r_rays.max()
                cos_omega_local_omega_r_rays = torch.clamp(cos_omega_local_omega_r_rays, min=-1, max=1)
                rad_omega_local_omega_r_rays = torch.acos(cos_omega_local_omega_r_rays)
                envmap_pix_distance = rad_omega_local_omega_r_rays / self.ratioRadiusOverPixel
                assert torch.all(envmap_pix_distance >= 0), (
                    envmap_pix_distance.min(),
                    envmap_pix_distance.max(),
                    rad_omega_local_omega_r_rays.min(),
                    rad_omega_local_omega_r_rays.max(),
                    cos_omega_local_omega_r_rays.max(),
                    cos_omega_local_omega_r_rays.min(),
                )
                assert config.gray_envmap_training  # otherwise, think about how to deal with this
                hint_ref_levels = []
                for l in range(self.gauL):
                    # if l == 0:
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
            elif lgtMode == "envmapMode":
                # something to pay attention to on the ref_out["hdr"]
                # under this lgtMode, we simply cast all the elements in ref_out["hdr"] into 0
                ref_out["hdr"][:] = 0

                # Note, class 15 low freq branch cannot do envmap
                # Even if you wish to play with the envmapMode here
                # Do remember to also input some fake point light info here

                # In here, it is assumed that all the rays are under the same envmap (envmap0)

                # 1. smooth the envmap to each level
                envmapFiltered = []
                for l in range(self.gauL):
                    if kwargs.get("specRoughnessID", None) is None:
                        gauKSize = int(self.gauKSizeList[l])
                        gauKSigma = float(self.gauKSigmaList[l])
                    else:
                        specRoughnessID = kwargs["specRoughnessID"]
                        assert type(specRoughnessID) is int
                        assert 0 <= specRoughnessID < len(self.gauKSizeList)
                        gauKSize = int(self.gauKSizeList[specRoughnessID])
                        gauKSigma = float(self.gauKSigmaList[specRoughnessID])
                    if gauKSize == 0:
                        envmapFiltered.append(envmap0.permute(2, 0, 1))
                    else:
                        tmp = gaussian_blur(
                            envmap0.permute(2, 0, 1), 
                            gauKSize,
                            gauKSigma,
                        )  # .permute(1, 2, 0)
                        envmapFiltered.append(tmp)
                envmapFiltered = torch.stack(envmapFiltered, 0)  # (6(L), 3(RGB), 512(H), 1024(W))
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
                hint_ref_levels = tmp[0, :, :, 0].permute(1, 0).reshape(-1, self.gauL, 3)
                hint_ref_levels_color_distribution = hint_ref_levels[:, 1, :] # use this to re-distribute colors
                hint_ref_levels_color = hint_ref_levels  # This will be used to do the visualization
                hint_ref_levels = hint_ref_levels.mean(2)
                hint_ref_levels[valid_mask == 0, :] = 0
                hint_ref_levels = hint_ref_levels.clamp(min=0, max=1)
            else:
                raise ValueError("Unknown lgtMode: %s" % lgtMode)

            hint_ref_opacities = modelFast.embeddingHintPointlightOpacities(ref_out["opacities"])
            hint_ref_self = ref_out["hdr"]
            hint_ref_levels = hint_ref_levels  # no meaning, just make sure you know there are three hints here
            hint_ref_all = torch.cat([hint_ref_opacities, hint_ref_self, hint_ref_levels], 1)
            
        hints_pointlight_opacities = ref_out["opacities"][ray_indices]  # put to points
        fullPredictionDict_fast = modelFast.appearance(
            xyzs, dirs, preAppearancePredictionDict_fast,
            lgtInputPoints=lgtInputPoints,
            hints_pointlight_opacities=hints_pointlight_opacities,
        )

        normal2_toadd = accumulate_along_rays(
            weights,
            values=fullPredictionDict_fast["out_hdr"],
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )

        # hdr3
        tmp = modelFast.normal3_envmap(
            torch.cat([
                hint_ref_all,
                modelFast.embeddingXyz(ref_point),
            ], 1)
        )
        normal3_toadd = modelFast.safe_exp(tmp, 1, 10).float()
        
        if lgtMode == "envmapMode":
            s = hint_ref_levels_color_distribution / (hint_ref_levels_color_distribution.sum(1)[:, None] + 0.00001) * 3
            normal3_toadd *= s

        assert len(opacities.shape) == 1
        assert len(depths.shape) == 1
        if config.if_highlight_case:
            colors = normal2_toadd + normal3_toadd + (1 - opacities[:, None]) * bg_color
        else:
            colors = normal2_toadd + (1 - opacities[:, None]) * bg_color

        results["lossSigmaTie"] = loss_sigma_tie
        if ("lossNormal3Tie" in config.wl.keys()) and (config.wl["lossNormal3Tie"] > 0):
            results["lossNormal3Tie"] = loss_normal3_tie
        results["hdr"] = colors
        results["hdr2"] = normal2_toadd
        results["hdr3"] = normal3_toadd
        # results["normal2"] = normal_2
        results["normal3"] = normal3_npp
        results["depth"] = depths
        results["opacity"] = opacities
        results["hintsRefOpacities"] = ref_out["opacities"][:, 0]
        results["hintsRefSelf"] = ref_out["hdr"]
        results["hintsRefLevels"] = hint_ref_levels
        if lgtMode == "pointLightMode":
            if valid_mask.sum() > 0:
                omega_local = F.normalize(lgtInputRays["lgtE"] - ref_point)
                L = omega_local
                assert torch.max(torch.abs(1 - torch.norm(L, p=2, dim=1))) < 1e-4
                V = omega_o
                assert torch.max(torch.abs(1 - torch.norm(V, p=2, dim=1))) < 1e-4
                H = F.normalize(L + V)
                normal3_NdotH = (normal3_npp * H).sum(1)
            else:
                normal3_NdotH = torch.zeros((normal3_npp.shape[0],), dtype=torch.float32, device=normal3_npp.device)
            results["normal3DotH"] = normal3_NdotH
        if lgtMode == "envmapMode":
            results["hintsRefLevelsColor"] = torch.where(valid_mask[:, None, None], hint_ref_levels_color, 0)

        results["statNumSampledPoints"] = torch.cuda.FloatTensor([int(xyzs.shape[0])])
        assert results["statNumSampledPoints"].device == xyzs.device
        results["statP2Rratio"] = torch.cuda.FloatTensor([float(xyzs.shape[0]) / float(rays_o.shape[0])])
        assert results["statP2Rratio"].device == xyzs.device

        return results

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_hdr: [B, N, 3]

        if self.cuda_ray:
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
