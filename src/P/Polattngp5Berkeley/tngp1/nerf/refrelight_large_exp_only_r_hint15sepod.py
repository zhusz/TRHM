
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

        deg_emb_xyz = config.N_emb_xyz
        deg_emb_dir = config.N_emb_dir

        # fixed hyper parameters
        self.epsilon = 1.0e-6  # avoid divided by zero
        self.hdr_clipped_clip = config.hdr_clipped_clip

        # pe
        self.embeddingXyz = Embedding(deg_emb_xyz)
        self.embeddingDir = Embedding(deg_emb_dir)
        self.embeddingOmegaGlobal = Embedding(deg_emb_dir)  # write as a separate module to emphasize
        self.embeddingOmegaLocal = Embedding(deg_emb_dir)  # write as a separate module to emphasize
        self.embeddingNormal3 = Embedding(deg_emb_dir)
        self.embeddingOmegaR = Embedding(deg_emb_dir)
        self.embeddingHintPointlightOpacities = Embedding(deg_emb_dir)
        in_channels_hint_pointlight_opacities = 2 * deg_emb_dir + 1

        # normal3
        bound = 1
        self.bound = bound
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
        in_channels_xyz = 6 * deg_emb_xyz + 3
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

        # density and colors
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": config.hash_n_features_per_level,  # 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float32,
        )
        self.sigma_net = tcnn.Network(
            n_input_dims=16 * config.hash_n_features_per_level,  # 32
            n_output_dims=1 + config.sigma_b_dim,  # 1 + 15
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": config.sigma_n_hidden_layers,  # 1
            },
        )
        in_channels_dir = 6 * deg_emb_dir + 3
        if config.inject_normal3:
            t = 4
        else:
            t = 3
        self.color_net = tcnn.Network(
            n_input_dims=config.sigma_b_dim + 1 + t * in_channels_dir + in_channels_hint_pointlight_opacities,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": config.color_net_hidden_layers,
            },
        )

    def density(self, xyz, if_do_bound_clamp, if_output_only_sigma, if_compute_normal3, **kwargs):
        if if_compute_normal3:
            iterCount = kwargs["iterCount"]
        if if_output_only_sigma:
            if_output_sigma_before_activation = kwargs.get("if_output_sigma_before_activation", False)

        x_input = xyz
        xyz_ = xyz
        if self.config.density_fix:
            with torch.no_grad():
                raise NotImplementedError
        else:
            xyz_ = (xyz_ + self.bound) / (2 * self.bound)
            xyz_ = self.encoder(xyz_)
            xyz_ = self.sigma_net(xyz_)
        sigma_before_activation = xyz_[:, 0]
        if self.config.density_activation == "exp":
            out_tau_spatial = self.safe_exp(xyz_[:, 0], shift=0, thre=10)  # torch.float16 cannot do thre > 10
        elif self.config.density_activation == "softplus":
            out_tau_spatial = F.softplus(xyz_[:, 0])
        else:
            raise NotImplementedError("Unknown density_activation: %s" % self.config.density_activation)
        xyz_ = xyz_[:, 1:]
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
            normal3_npp_unormalize_raw = self.normal3_npp(normal3_feat)
            bound = 6e4  # torch.finfo(torch.float16).max is 65504.0
            normal3_npp = F.normalize(
                torch.clamp(normal3_npp_unormalize_raw, min=-bound, max=bound).float(),
                p=2, dim=1
            )

        if if_output_only_sigma:
            if if_output_sigma_before_activation:
                return sigma_before_activation
            else:
                return out_tau_spatial  # most general cases, e.g. sampling
        else:
            if not if_compute_normal3:
                return out_tau_spatial, xyz_
            else:
                return out_tau_spatial, xyz_, normal3_npp, None, None, sigma_before_activation, normal3_feat

    def preAppearance(self, x, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]
        density_fix = self.config.density_fix
        iterCount = kwargs["iterCount"]

        assert requires_grad == False

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
        out_tau_spatial, xyz_, normal3_npp, normal3_alpha, normal3_hdr, sigma_before_activation, normal3_feat = (
            tmp[0][:, None], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6])

        preAppearancePredictionDict = {
            "sigma_before_activation": sigma_before_activation,
            "out_tau_spatial": out_tau_spatial[:, 0],  # tau, which is the density (not yet multiplied with 10)
            "xyz_": xyz_,  # the feat
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

        input_dir_norm = torch.norm(input_dir, dim=1, p=2)
        assert input_dir_norm.min() >= 1.0 - self.epsilon, input_dir_norm.min()
        assert input_dir_norm.max() <= 1.0 + self.epsilon, input_dir_norm.max()

        omegaLocal = F.normalize(lgtE - x)
        omegaLocal_encoded = self.embeddingOmegaLocal(omegaLocal)
        normal3_point = preAppearancePredictionDict["out_normal3_npp"]
        normal3_point_encoded = self.embeddingNormal3(F.normalize(normal3_point, p=2, dim=1))

        hints_pointlight_opacities_embedded = self.embeddingHintPointlightOpacities(
            hints_pointlight_opacities
        )

        omega_o = -input_dir
        dot = (omega_o * normal3_point).sum(1)[:, None]
        omega_r = (
            2.0
            * dot # * (omega_o * normal3_point).sum(1)[:, None]
            * normal3_point
            - omega_o
        )
        omega_r_encoded = self.embeddingOmegaR(omega_r)

        out_b_spatial = preAppearancePredictionDict["xyz_"]

        if self.config.inject_normal3:
            forwarding = torch.cat([
                out_b_spatial,
                omega_r_encoded,
                dot,
                omegaGlobal_encoded,
                omegaLocal_encoded,
                normal3_point_encoded,
                hints_pointlight_opacities_embedded,
            ], 1)
        else:
            forwarding = torch.cat([
                out_b_spatial,
                omega_r_encoded,
                dot,
                omegaGlobal_encoded,
                omegaLocal_encoded,
                # normal3_point_encoded,
                hints_pointlight_opacities_embedded,
            ], 1)

        x = self.color_net(forwarding)
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
