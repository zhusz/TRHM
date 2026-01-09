
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import tinycudann as tcnn

from ..activation import trunc_exp
from .ide import get_ml_array, integrated_dir_enc_fn, sph_harm_coeff  # noqa


class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class NeRFNetwork(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFNetwork, self).__init__()

        config = kwargs["config"]
        self.config = config
        self.minBound = config.minBound
        self.maxBound = config.maxBound

        if config.get("needDensityFixer", False):
            self.modelDensityTeacher = kwargs["modelDensityTeacher"]

        D = config.N_depth
        D_appearance = config.N_depth_appearance
        W = config.N_width
        W_bottleneck = config.N_width_bottleneck
        W_appearance = config.N_width_appearance
        deg_emb_xyz = config.N_emb_xyz
        deg_emb_dir = config.N_emb_dir
        deg_directional_enc = config.deg_directional_enc
        skips = config.skips
        skips_appearance = config.skips_appearance
        density_fix = config.density_fix

        self.D = D
        self.D_appearance = D_appearance
        self.W = W
        self.W_appearance = W_appearance
        self.deg_emb_xyz = deg_emb_xyz
        self.deg_emb_dir = deg_emb_dir
        self.deg_directional_enc = deg_directional_enc
        in_channels_xyz = 6 * deg_emb_xyz + 3
        in_channels_dir = 6 * deg_emb_dir + 3
        in_channels_hint_pointlight_opacities = 2 * config.N_emb_hint_pointlight_opacities + 1
        in_channels_hint_pointlight_ggx = 4 * (2 * config.N_emb_hint_pointlight_ggx + 1)
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.skips_appearance = skips_appearance

        # fixed hyper parameters
        self.epsilon = 1.0e-6  # avoid divided by zero
        self.hdr_clipped_clip = config.hdr_clipped_clip

        # normal3
        bound = 1
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
        self.normal3_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float32,
        )
        self.normal3_npp = tcnn.Network(  # npp stands for n-prime-prime
            n_input_dims=32,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",  # need F.normalize in forward
                "n_neurons": 64,
                "n_hidden_layers": 3,
            }
        )
        self.normal3_envmap = tcnn.Network(
            n_input_dims=in_channels_hint_pointlight_opacities + (config.pyramid_len * 1) + 3 + in_channels_xyz,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3,
            }
        )

        # pe
        self.embeddingXyz = Embedding(deg_emb_xyz)
        self.embeddingDir = Embedding(deg_emb_dir)
        self.embeddingOmegaGlobal = Embedding(deg_emb_dir)  # write as a separate module to emphasize
        self.embeddingOmegaLocal = Embedding(deg_emb_dir)  # write as a separate module to emphasize
        self.embeddingNormal3 = Embedding(deg_emb_dir)
        self.embeddingHintPointlightOpacities = Embedding(config.N_emb_hint_pointlight_opacities)

        # ide
        ml_array = get_ml_array(deg_directional_enc)
        l_max = 2 ** (deg_directional_enc - 1)
        mat = np.zeros((l_max + 1, ml_array.shape[1]))
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = sph_harm_coeff(l, m, k)
        ml_array = ml_array.astype(np.float32)
        mat = mat.astype(np.float32)
        self.ml_array = torch.from_numpy(ml_array)  # no grad
        self.mat = torch.from_numpy(mat)  # no grad

        in_channels_ide = self.mat.shape[1] * 2
        self.in_channels_ide = in_channels_ide

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)

        # outputs after the density part (no any dir)
        # Note here tau is already after softplus, which is different from the previous sigma
        # Already Softplus here, no need to do it again in the rendering.py 
        self.sigma = nn.Linear(W, 1)  # for tau. Set the name to be "self.sigma" for the purpose of finetuning
        self.out_color_diffuse_spatial = nn.Sequential(
            nn.Linear(W + in_channels_dir, W), nn.ReLU(True),
            nn.Linear(W, W // 2), nn.ReLU(True),
            nn.Linear(W // 2, 3)
        )
        self.out_b_spatial = nn.Sequential(nn.Linear(W, W_bottleneck), nn.ReLU(True))
        self.out_tint_spatial = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())
        self.out_roughness_spatial = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.out_unormalized_normal_pred_spatial = nn.Linear(W, 3)

        # direction encoding layers
        assert D_appearance is not None
        assert W_appearance is not None
        assert type(skips_appearance) is list
        for i in range(D_appearance):
            if i == 0:
                # layer = nn.Linear(W_bottleneck + in_channels_ide + 1 + in_channels_dir, W_appearance)
                layer = nn.Linear(
                    W_bottleneck + in_channels_ide + 1 + in_channels_dir * 2 + in_channels_hint_pointlight_opacities,  # + in_channels_hint_pointlight_ggx,
                    W_appearance
                )
                # normal3 injection
                setattr(self, f"dir_encoding_normal3_injection_{i+1}", nn.Linear(in_channels_dir, W_appearance))
            elif i in skips_appearance:
                layer = nn.Linear(
                    W_appearance + W_bottleneck + in_channels_ide + 1 + in_channels_dir * 2 + in_channels_hint_pointlight_opacities,  # + in_channels_hint_pointlight_ggx,
                    W_appearance
                )
                # normal3 injection
                setattr(self, f"dir_encoding_normal3_injection_{i+1}", nn.Linear(in_channels_dir, W_appearance))
            elif i == D_appearance - 1:
                layer = nn.Linear(W_appearance, W_appearance // 2)
            else:
                layer = nn.Linear(W_appearance, W_appearance)
            # layer = nn.Sequential(layer, nn.ReLU(True))  # You now need to relu yourself!
            setattr(self, f"dir_encoding_{i+1}", layer)
        self.out_color_specular_appearance = nn.Linear(W_appearance // 2, 3)

    def density(self, xyz, if_do_bound_clamp, if_output_only_sigma, if_compute_normal3, **kwargs):
        if if_compute_normal3:
            iterCount = kwargs["iterCount"]
        if if_output_only_sigma:
            if_output_sigma_before_activation = kwargs.get("if_output_sigma_before_activation", False)

        x_input = xyz
        xyz_embedded = self.embeddingXyz(xyz)
        xyz_ = xyz_embedded  # used for forwarding
        if self.config.density_fix:
            with torch.no_grad():
                for i in range(self.D):
                    if i in self.skips:
                        xyz_ = torch.cat([xyz_embedded, xyz_], -1)
                    xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
                sigma = self.sigma(xyz_)
        else:
            for i in range(self.D):
                if i in self.skips:
                    xyz_ = torch.cat([xyz_embedded, xyz_], -1)
                xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
            sigma = self.sigma(xyz_)
        sigma_before_activation = sigma[:, 0]

        config = self.config
        if config.get("needDensityFixer", False):
            raise NotImplementedError("TODO")

        if self.config.density_activation == "exp":
            out_tau_spatial = trunc_exp(sigma)  # F.softplus(sigma)
        elif self.config.density_activation == "softplus":
            out_tau_spatial = F.softplus(sigma)
        else:
            raise NotImplementedError("Unknown density_activation: %s" % self.config.density_activation)
        out_tau_spatial = out_tau_spatial[:, 0]
        if if_do_bound_clamp:
            flagInside = (
                (x_input[:, 0] >= self.minBound[0]) & (x_input[:, 0] <= self.maxBound[0]) &
                (x_input[:, 1] >= self.minBound[1]) & (x_input[:, 1] <= self.maxBound[1]) &
                (x_input[:, 2] >= self.minBound[2]) & (x_input[:, 2] <= self.maxBound[2])
            )
            out_tau_spatial = torch.where(flagInside, out_tau_spatial, 0)

        # normal3
        if if_compute_normal3:
            normal3_feat = self.normal3_encoder(xyz) * 100  # 100: because I am lazy to do tcnn module initialization
            normal3_npp = F.normalize(self.normal3_npp(normal3_feat), p=2, dim=1)

        if if_output_only_sigma:
            if if_output_sigma_before_activation:
                return sigma_before_activation
            else:
                return out_tau_spatial  # most general cases, e.g. sampling
        else:
            if not if_compute_normal3:
                return out_tau_spatial, xyz_
            else:
                return out_tau_spatial, xyz_, normal3_npp, None, None, sigma_before_activation

    def preAppearance(self, x, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]
        density_fix = self.config.density_fix
        iterCount = kwargs["iterCount"]

        input_xyz = x

        if requires_grad:
            xyz = input_xyz.requires_grad_(True)
        else:
            xyz = input_xyz

        tmp = self.density(
            xyz,
            if_do_bound_clamp=if_do_bound_clamp,
            if_output_only_sigma=False,
            if_compute_normal3=True,
            iterCount=iterCount,
        )
        out_tau_spatial, xyz_, normal3_npp, normal3_alpha, normal3_hdr, sigma_before_activation = (
            tmp[0][:, None], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])

        if requires_grad and not density_fix:
            out_unormalized_normal_from_tau_spatial = -torch.autograd.grad(
                out_tau_spatial.sum(), [xyz], create_graph=True
            )[0]
        else:
            out_unormalized_normal_from_tau_spatial = torch.nan * torch.zeros_like(xyz)
        out_normal_from_tau_spatial = torch.div(
            out_unormalized_normal_from_tau_spatial,
            torch.clamp(
                torch.norm(out_unormalized_normal_from_tau_spatial, dim=1, p=2)[:, None],
                min=self.epsilon,
            )
        )       

        out_roughness_spatial = self.out_roughness_spatial(xyz_)
        if self.config.density_fix:
            with torch.no_grad():
                # out_roughness_spatial = self.out_roughness_spatial(xyz_)
                out_unormalized_normal_pred_spatial = self.out_unormalized_normal_pred_spatial(xyz_)
                out_normal_pred_spatial = torch.div(
                    out_unormalized_normal_pred_spatial,
                    torch.clamp(
                        torch.norm(out_unormalized_normal_pred_spatial, dim=1, p=2)[:, None],
                        min=self.epsilon,
                    )
                )
        else:
            # out_roughness_spatial = self.out_roughness_spatial(xyz_)
            out_unormalized_normal_pred_spatial = self.out_unormalized_normal_pred_spatial(xyz_)
            out_normal_pred_spatial = torch.div(
                out_unormalized_normal_pred_spatial,
                torch.clamp(
                    torch.norm(out_unormalized_normal_pred_spatial, dim=1, p=2)[:, None],
                    min=self.epsilon,
                )
            )

        preAppearancePredictionDict = {
            "sigma_before_activation": sigma_before_activation,
            "out_tau_spatial": out_tau_spatial[:, 0],  # tau, which is the density (not yet multiplied with 10)
            "xyz_": xyz_,  # the feat
            "out_roughness_spatial": out_roughness_spatial,  # rho, which is kappa_inv
            "out_normal_pred_spatial": out_normal_pred_spatial,  # n2
            "out_normal_from_tau_spatial": out_normal_from_tau_spatial,  # n1
            "out_normal3_npp": normal3_npp,  # n3
        }

        return preAppearancePredictionDict

    def forward(self, x, d, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]
        lgtInputPoints = kwargs["lgtInputPoints"]
        iterCount = kwargs["iterCount"]

        hints_pointlight_opacities = kwargs["hints_pointlight_opacities"]

        input_xyz, input_dir = x, d

        preAppearancePredictionDict = self.preAppearance(
            x,
            if_do_bound_clamp=if_do_bound_clamp,
            requires_grad=requires_grad,
            iterCount=iterCount,
        )

        fullPredictionDict = self.appearance(
            x, d, preAppearancePredictionDict, lgtInputPoints=lgtInputPoints,
            hints_pointlight_opacities=hints_pointlight_opacities,
        )

        return fullPredictionDict

    def appearance(self, x, d, preAppearancePredictionDict, **kwargs):

        hints_pointlight_opacities = kwargs["hints_pointlight_opacities"]

        xyz_ = preAppearancePredictionDict["xyz_"]
        lgtInputPoints = kwargs["lgtInputPoints"]

        lgtE = lgtInputPoints["lgtE"]
        objCentroid = lgtInputPoints["objCentroid"]
        omegaGlobal = F.normalize(lgtE - objCentroid)  # This is different from the old convention
        omegaGlobal_encoded = self.embeddingOmegaGlobal(omegaGlobal)

        input_xyz, input_dir = x, d

        out_color_diffuse_spatial = self.out_color_diffuse_spatial(torch.cat([xyz_, omegaGlobal_encoded], 1))
        out_b_spatial = self.out_b_spatial(xyz_)
        out_tint_spatial = self.out_tint_spatial(xyz_)

        out_roughness_spatial = preAppearancePredictionDict["out_roughness_spatial"]
        out_normal_pred_spatial = preAppearancePredictionDict["out_normal_pred_spatial"]
 
        input_dir_norm = torch.norm(input_dir, dim=1, p=2)
        assert input_dir_norm.min() >= 1.0 - self.epsilon, input_dir_norm.min()
        assert input_dir_norm.max() <= 1.0 + self.epsilon, input_dir_norm.max()
        omega_o = -input_dir
        omega_r = (
            2.0
            * (omega_o * out_normal_pred_spatial).sum(1)[:, None]
            * out_normal_pred_spatial
            - omega_o
        )
        omega_r_encoded = integrated_dir_enc_fn(
            omega_r,
            out_roughness_spatial,
            self.ml_array.to(omega_r.device),
            self.mat.to(omega_r.device),
        )
        dot = (omega_o * out_normal_pred_spatial).sum(1)[:, None]

        omegaLocal = F.normalize(lgtE - x)
        omegaLocal_encoded = self.embeddingOmegaLocal(omegaLocal)
        normal3_point = preAppearancePredictionDict["out_normal3_npp"]
        normal3_point_encoded = self.embeddingNormal3(F.normalize(normal3_point, p=2, dim=1))

        hints_pointlight_opacities_embedded = self.embeddingHintPointlightOpacities(
            hints_pointlight_opacities
        )

        forwarding = torch.cat([
            out_b_spatial,
            omega_r_encoded,
            dot,
            omegaGlobal_encoded,
            omegaLocal_encoded,
            hints_pointlight_opacities_embedded,
        ],1)

        for i in range(self.D_appearance):
            if i in self.skips_appearance:
                forwarding = torch.cat([
                    forwarding,
                    out_b_spatial,
                    omega_r_encoded,
                    dot,
                    omegaGlobal_encoded,
                    omegaLocal_encoded,
                    # normal3_point_encoded,
                    hints_pointlight_opacities_embedded,
                    # hints_pointlight_ggx_embedded,
                ], 1)
            result = getattr(self, f"dir_encoding_{i+1}")(forwarding)
            if (i == 0) or (i in self.skips_appearance):
                result += getattr(self, f"dir_encoding_normal3_injection_{i+1}")(normal3_point_encoded.float())
            forwarding = F.relu(result)
        out_color_specular_appearance = self.out_color_specular_appearance(forwarding)

        # Suggest to put more layers before getting out_color_diffuse_spatial, to make the numerical better
        x = out_color_diffuse_spatial + out_tint_spatial * out_color_specular_appearance
 
        out_hdr = self.hdr_clipped_clip * torch.sigmoid(x)

        fullPredictionDict = preAppearancePredictionDict
        fullPredictionDict["out_hdr"] = out_hdr

        return fullPredictionDict

    @staticmethod
    def safe_exp(x, shift, thre):  # the classic setting: shift is 1, and thre is 10
        x = x - shift
        out = torch.where(
            x <= thre,
            torch.exp(torch.clamp(x, max=thre)),
            float(math.e ** thre + 0.01) + (x - thre),
        )
        return out

    def color(self, **kwargs):
        raise NotImplementedError

    def get_params(self, **kwargs):
        raise NotImplementedError
