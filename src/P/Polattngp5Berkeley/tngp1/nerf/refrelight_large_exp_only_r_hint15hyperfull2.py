
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import tinycudann as tcnn
import opt_einsum as oe

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
        self.df = kwargs["df"]  # each instance define its own density_fix

        # fixed hyper parameters
        self.epsilon = 1.0e-6  # avoid divided by zero
        self.hdr_clipped_clip = config.hdr_clipped_clip

        # normal3
        self.if_need_normal3_net = kwargs["if_need_normal3_net"]
        bound = 1
        self.bound = bound
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
        if kwargs["if_need_normal3_net"]:
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

        # pe
        self.embeddingXyz = Embedding(deg_emb_xyz)
        self.embeddingDir = Embedding(deg_emb_dir)
        self.embeddingOmegaGlobal = Embedding(deg_emb_dir)  # write as a separate module to emphasize
        self.embeddingOmegaLocal = Embedding(deg_emb_dir)  # write as a separate module to emphasize
        self.embeddingNormal3 = Embedding(deg_emb_dir)
        self.embeddingOmegaR = Embedding(deg_emb_dir)
        self.embeddingHintPointlightOpacities = Embedding(deg_emb_dir)
        in_channels_hint_pointlight_opacities = 2 * deg_emb_dir + 1

        in_channels_xyz = 6 * deg_emb_xyz + 3
        self.normal3_envmap = tcnn.Network(
            n_input_dims=in_channels_hint_pointlight_opacities + (config.pyramid_len * 1) + 3 + in_channels_xyz,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",  # self.safe_exp in forward
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

        self.if_need_hyper_net = kwargs["if_need_hyper_net"]
        if kwargs["if_need_hyper_net"]:
            in_channels_dir = 6 * deg_emb_dir + 3
            if config.inject_normal3:
                t = 4
            else:
                t = 3
            self.w_dims_in = [
                config.sigma_b_dim + (t - 3) * in_channels_dir,
                64, 64, 64,
            ]
            self.w_dims_out = [
                64, 64, 64, 3
            ]
            self.w_dims_params = [(i + 1) * o for i, o in zip(self.w_dims_in, self.w_dims_out)]
            assert len(self.w_dims_in) == len(self.w_dims_out) == len(self.w_dims_params)
            self.hyper_color_net = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(16 * 32 * 3, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 64),
                    nn.ReLU(True),
                    nn.Linear(64, d),
                )
                for d in self.w_dims_params
            ])

    def hyper(self, envmapAlphaed):
        config = self.config

        B_lgt = int(envmapAlphaed.shape[0])
        assert envmapAlphaed.shape == (B_lgt, 16, 32, 3)
        x = envmapAlphaed.reshape(B_lgt, 16 * 32 * 3)

        wbs = [
            self.hyper_color_net[i](x).float() * config.paramInitialScale
            for i in range(len(self.w_dims_params))
        ]

        hyperPredictionDict = {
            "wbs": wbs,
        }
        return hyperPredictionDict

    def density_internal(self, xyz, if_do_bound_clamp, if_compute_normal3):
        x_input = xyz
        xyz_ = xyz
        xyz_ = (xyz_ + self.bound) / (2 * self.bound)
        xyz_ = self.encoder(xyz_)
        xyz_ = self.sigma_net(xyz_)
        sigma_before_activation = xyz_[:, 0]
        if self.config.density_activation == "exp":
            out_tau_spatial = self.safe_exp(xyz_[:, 0], shift=0, thre=10)
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
        if if_compute_normal3:
            normal3_feat = self.normal3_encoder(xyz) * 100  # 100: because I am lazy to do tcnn module initialization
            normal3_npp_unormalize_raw = self.normal3_npp(normal3_feat)
            bound = 6e4  # torch.finfo(torch.float16).max is 65504.0
            normal3_npp = F.normalize(
                torch.clamp(normal3_npp_unormalize_raw, min=-bound, max=bound).float(),
                p=2, dim=1
            )
        else:
            normal3_npp = None
            normal3_feat = None
        return sigma_before_activation, xyz_, out_tau_spatial, normal3_npp, normal3_feat

    def density(self, xyz, if_do_bound_clamp, if_output_only_sigma, if_compute_normal3, **kwargs):
        if if_compute_normal3:
            iterCount = kwargs["iterCount"]
        if if_output_only_sigma:
            if_output_sigma_before_activation = kwargs.get("if_output_sigma_before_activation", False)

        if self.df:
            with torch.no_grad():
                sigma_before_activation, xyz_, out_tau_spatial, normal3_npp, normal3_feat = self.density_internal(
                    xyz, if_do_bound_clamp, if_compute_normal3
                )
        else:
            sigma_before_activation, xyz_, out_tau_spatial, normal3_npp, normal3_feat = self.density_internal(
                xyz, if_do_bound_clamp, if_compute_normal3
            )

        if if_output_only_sigma:
            if if_output_sigma_before_activation:
                return sigma_before_activation
            else:
                return out_tau_spatial  # most general cases, e.g. sampling
        else:
            if not if_compute_normal3:
                return out_tau_spatial, xyz_, sigma_before_activation
            else:
                # return out_tau_spatial, xyz_, normal3_npp, normal3_alpha, normal3_hdr, sigma_before_activation, normal3_feat
                return out_tau_spatial, xyz_, normal3_npp, sigma_before_activation, normal3_feat

    def preAppearance(self, x, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]
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
            if_compute_normal3=False,
            iterCount=iterCount,
        )
        out_tau_spatial, xyz_, sigma_before_activation = (
            tmp[0][:, None], tmp[1], tmp[2])

        preAppearancePredictionDict = {
            "sigma_before_activation": sigma_before_activation,
            "out_tau_spatial": out_tau_spatial[:, 0],  # tau, which is the density (not yet multiplied with 10)
            "xyz_": xyz_,  # the feat
        }

        return preAppearancePredictionDict

    def forward(self, x, d, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]
        lgtInputPoints = kwargs["lgtInputPoints"]
        iterCount = kwargs["iterCount"]

        input_xyz, input_dir = x, d

        preAppearancePredictionDict = self.preAppearance(
            x,
            if_do_bound_clamp=if_do_bound_clamp,
            requires_grad=requires_grad,
            iterCount=iterCount,
        )

        fullPredictionDict = self.appearance(
            x, d, preAppearancePredictionDict, lgtInputPoints=lgtInputPoints)

        return fullPredictionDict

    def appearance(self, x, d, preAppearancePredictionDict, hyperPredictionDict, **kwargs):

        iterCount = kwargs["iterCount"]

        wbs = hyperPredictionDict["wbs"]
        insideBatchInd_fgmgbg_point_lgt = hyperPredictionDict["insideBatchInd_fgmgbg_point_lgt"]

        input_xyz, input_dir = x, d

        input_dir_norm = torch.norm(input_dir, dim=1, p=2)
        assert input_dir_norm.min() >= 1.0 - self.epsilon, input_dir_norm.min()
        assert input_dir_norm.max() <= 1.0 + self.epsilon, input_dir_norm.max()

        normal3_point = preAppearancePredictionDict["out_normal3_npp"]
        normal3_point_encoded = self.embeddingNormal3(F.normalize(normal3_point, p=2, dim=1))

        out_b_spatial = preAppearancePredictionDict["xyz_"]

        # hdr2        
        if self.config.inject_normal3:
            forwarding = torch.cat([
                out_b_spatial,
                normal3_point_encoded,
            ], 1)
        else:
            forwarding = out_b_spatial

        for i in range(len(self.w_dims_in)):
            forwarding = torch.cat([forwarding, torch.ones_like(forwarding[:, :1])], 1).float()
            wb_lgt = wbs[i].reshape(wbs[i].shape[0], self.w_dims_in[i] + 1, self.w_dims_out[i])
            forwarding = self.indexedMatmul(forwarding, wb_lgt, insideBatchInd_fgmgbg_point_lgt)
            if i < len(self.w_dims_in) - 1:
                forwarding = F.relu(forwarding)
        hdr2 = F.softplus(forwarding)

        fullPredictionDict = preAppearancePredictionDict
        fullPredictionDict["out_hdr2"] = hdr2

        return fullPredictionDict

    @staticmethod
    def indexedMatmul(x, wb_lgt, ind):
        # x: (b_rays, w_dim_in)  # already augmented by the bias all-one-vector
        # wb_lgt: (b_lgt, w_dim_in, w_dim_out)  # this is the weight and bias
        # ind: (b_rays,)  # meaning the i-th sample in x, i.e. x[i, :], should be matmuled with wb_lgt[ind[i], :, :]

        # the key is to generate this intermediate proxy from the ind
        # proxy (b_rays, b_lgt)

        b_lgt, w_dim_in, w_dim_out = wb_lgt.shape  # w_dim_in already includes the augmented one-vector (for bias)
        b_rays = int(ind.shape[0])
        assert x.shape == (b_rays, w_dim_in)
        assert x.dtype == torch.float32
        assert wb_lgt.dtype == torch.float32
        assert ind.dtype == torch.int64
        assert 0 <= ind.min()
        assert ind.max() < b_lgt
        assert x.device == wb_lgt.device == ind.device

        proxy = torch.zeros(b_rays, b_lgt, dtype=torch.float32, device=x.device)
        proxy[torch.arange(b_rays), ind] = 1

        out = oe.contract("r i, l i o, r l -> r o", x, wb_lgt, proxy)

        return out

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
