import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

# from nerfacc.volrend import render_weight_from_density, accumulate_along_rays
from codes_py.toolbox_nerfacc.volrend import render_weight_from_density, accumulate_along_rays

from .. import raymarching
from .utils import custom_meshgrid


def get_render_step_size(iterCount, render_step_size_keys, render_step_size_vals):
    assert render_step_size_keys[0] == 0
    assert iterCount >= 0
    b = np.where(iterCount >= render_step_size_keys)[0][-1]
    return render_step_size_vals[b]


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
    modelDensityTeacher = kwargs["modelDensityTeacher"]

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
                modelDensityTeacher=modelDensityTeacher,
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
                    # ref_t_starts <= 0.025 * 10000  # config.reflection_surface_cutoff
                    ref_t_starts <= config.reflection_surface_cutoff
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
            ref_sigmas = modelRef.density(
                ref_positions, if_do_bound_clamp=True, if_output_only_sigma=True, if_compute_normal3=False,
                modelDensityTeacher=modelDensityTeacher,
            ) * renderer.density_scale
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

        else:
            if "opacities" in requesting_list:
                out["opacities"] = torch.zeros_like(ref_rays_o[:, :1])
    else:
        if "opacities" in requesting_list:
            out["opacities"] = torch.zeros_like(ref_rays_o[:, :1])

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
    
    def run_cuda(self, rays_o, rays_d, models, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        batch_head_index = kwargs.get("batch_head_index", None)

        occGrid = models["occGrid"]
        model = models["model"]

        config = self.config

        lgtInput = kwargs["lgtInput"]

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
            sigmas = model.density(
                positions,
                if_do_bound_clamp=True,
                if_output_only_sigma=True,
                if_compute_normal3=False,
                modelDensityTeacher=models.get("modelDensityTeacher", None),
            ) * self.density_scale
            return sigmas

        ray_indices, t_starts, t_ends = occGrid.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            near_plane=nears.min(), far_plane=(fars[fars < 1000].max() if torch.any(fars < 1000) else 10 * nears.min()),
            early_stop_eps=1e-4, alpha_thre=1e-5,
            render_step_size=get_render_step_size(iterCount, self.render_step_size_keys, self.render_step_size_vals),
            batch_head_index=batch_head_index,
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
        lgtInputPoints = {k: lgtInputRays[k][ray_indices] for k in lgtInputRays}
        
        assert self.enable_refnerf
        assert iterCount >= self.empty_cache_stop_iter
        preAppearancePredictionDict = model.preAppearance(
            xyzs,
            if_do_bound_clamp=True,
            requires_grad=requires_grad,
            lgtInputPoints=lgtInputPoints,
            iterCount=iterCount,
            modelDensityTeacher=models.get("modelDensityTeacher", None),
        )
        sigmas = preAppearancePredictionDict["out_tau_spatial"] * self.density_scale
        normal_pred_point = preAppearancePredictionDict["out_normal_pred_spatial"]
        normal_from_tau_point = preAppearancePredictionDict["out_normal_from_tau_spatial"]

        weights, trans, alphas = render_weight_from_density(
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

        # appearance
        H = None
        with torch.no_grad():
            if config.get("ablation_tag", 0) == 2:
                normal_1_unormalized = accumulate_along_rays(
                    weights,
                    values=normal_from_tau_point,
                    ray_indices=ray_indices,
                    n_rays=int(rays_o.shape[0]),
                )
                normal_1 = F.normalize(normal_1_unormalized)
            normal_2_unormalized = accumulate_along_rays(
                weights,
                values=normal_pred_point,
                ray_indices=ray_indices,
                n_rays=int(rays_o.shape[0]),
            )
            # First order reflection
            valid_mask = opacities >= 0.99
            valid_mask[depths < torch.finfo(torch.float32).eps] = False
            valid_mask[torch.norm(normal_2_unormalized, p=2, dim=1) < 0.9] = False
            normal_2 = F.normalize(normal_2_unormalized)
            valid_mask_3 = valid_mask[:, None].repeat(1, 3)
            ref_point = torch.where(
                valid_mask_3,
                rays_o + depths[:, None] * rays_d,  # depths is euclidean depth
                0,
            )

            # pointlight_opacities hint
            omega_local = F.normalize(lgtInput["lgtE"] - ref_point)
            pointlight_out = surface_eye(
                ref_point, omega_local, valid_mask,
                renderer=self,
                modelRef=model,
                occGrid=occGrid,
                requesting_list=["opacities"],
                iterCount=iterCount,
                config=config,
                modelDensityTeacher=models.get("modelDensityTeacher", None),
            )

            # pointlight_ggx_cooktorrance hint
            omega_o = -rays_d
            normal2_dot_h_record = torch.zeros(
                omega_o.shape[0], dtype=torch.float32, device=omega_o.device
            )
            if config.get("ablation_tag", 0) == 2:
                normal1_dot_h_record = torch.zeros(
                    omega_o.shape[0], dtype=torch.float32, device=omega_o.device
                )
            if valid_mask.sum() > 0:
                L = omega_local
                assert torch.max(torch.abs(1 - torch.norm(L, p=2, dim=1))) < 1e-4
                V = omega_o
                assert torch.max(torch.abs(1 - torch.norm(V, p=2, dim=1))) < 1e-4
                H = F.normalize(L + V)
                N = normal_2
                assert torch.max(torch.abs(1 - torch.norm(N[valid_mask, :], p=2, dim=1))) < 1e-4
                n_dot_h = (N[valid_mask, :] * H[valid_mask, :]).sum(1).clip(0, 1)

                normal2_dot_h_record[valid_mask] = n_dot_h

                if config.get("ablation_tag", 0) == 2:
                    normal1_dot_h_record[valid_mask] = (normal_1[valid_mask, :] * H[valid_mask, :]).sum(1).clip(0, 1)

        hints_pointlight_opacities = pointlight_out["opacities"][ray_indices]  # put to points
        fullPredictionDict = model.appearance(
            xyzs, dirs, preAppearancePredictionDict,
            lgtInputPoints=lgtInputPoints,
            hints_pointlight_opacities=hints_pointlight_opacities,
        )

        if self.enable_refnerf:
 
            loss_tie_points = ((normal_pred_point - normal_from_tau_point) ** 2).sum(1)
            loss_tie_rays = accumulate_along_rays(
                weights, values=loss_tie_points[:, None], ray_indices=ray_indices, n_rays=int(rays_o.shape[0]))

            loss_backface_points = (  # you do not need to multiply the weights here
                F.relu((normal_pred_point * dirs).sum(1))
            )
            loss_backface_rays = accumulate_along_rays(
                weights, values=loss_backface_points[:, None], ray_indices=ray_indices, n_rays=int(rays_o.shape[0]))

        normal2_toadd = accumulate_along_rays(
            weights,
            values=fullPredictionDict["out_hdr"],
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )

        if config.get("if_normal3_tied_to_normal2", False):
            normal3_npp = normal_2.detach().clone()
        else:
            normal3_npp = F.normalize(accumulate_along_rays(
                weights,
                values=fullPredictionDict["out_normal3_npp"],
                ray_indices=ray_indices,
                n_rays=int(rays_o.shape[0]),
            ), dim=1, p=2)
        normal3_alpha = accumulate_along_rays(
            weights,
            values=fullPredictionDict["out_normal3_alpha"],
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )
        normal3_hdr = accumulate_along_rays(
            weights,
            values=fullPredictionDict["out_normal3_hdr"],
            ray_indices=ray_indices,
            n_rays=int(rays_o.shape[0]),
        )
        if H is None:
            normal3_NdotH = torch.zeros((normal3_npp.shape[0],), dtype=torch.float32, device=normal3_npp.device)
        else:
            normal3_NdotH = (normal3_npp * H).sum(1)
        if config.get("ablation_tag", 0) == 0:  # original model
            normal3_toadd = normal3_hdr * (torch.pow(F.relu(normal3_NdotH)[:, None], normal3_alpha))
        elif config.get("ablation_tag", 0) == 1:
            normal3_toadd = normal3_hdr * (torch.pow(F.relu(normal2_dot_h_record)[:, None], normal3_alpha))
        elif config.get("ablation_tag", 0) == 2:
            normal3_toadd = normal3_hdr * (torch.pow(F.relu(normal1_dot_h_record)[:, None], normal3_alpha))
        else:
            raise ValueError("Unknown config.ablation_tag: %d" % config.get("ablation_tag", 0))

        assert len(opacities.shape) == 1
        assert len(depths.shape) == 1
        # colors = colors + (1 - opacities[:, None]) * bg_color
        # normal2_toadd = torch.clamp(colors, min=0, max=2)  # 2: a hyper parameter
        if config.if_highlight_case:
            colors = normal2_toadd + normal3_toadd + (1 - opacities[:, None]) * bg_color
        else:
            colors = normal2_toadd + (1 - opacities[:, None]) * bg_color

        results["hdr"] = colors
        results["hdr2"] = normal2_toadd
        results["hdr3"] = normal3_toadd
        results["normal3"] = normal3_npp
        results["depth"] = depths
        results["opacity"] = opacities
        if self.enable_refnerf:
            results["lossTieRays"] = loss_tie_rays
            results["lossBackfaceRays"] = loss_backface_rays
            results["normal2"] = normal_2
            results["normal2DotH"] = normal2_dot_h_record
            results["normal3DotH"] = normal3_NdotH

        # hints
        if callFlag != "forwardNetNeRF":
            results["hintsPointlightOpacities"] = pointlight_out["opacities"][:, 0]

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
