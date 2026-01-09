
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from ..activation import trunc_exp
from .ide import get_ml_array, integrated_dir_enc_fn, sph_harm_coeff  # noqa
from .renderer_hint2 import NeRFRenderer


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


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        D=8,
        D_appearance=None,
        W=256,
        W_bottleneck=None,
        W_appearance=None,
        deg_emb_xyz=None,  # pe
        deg_emb_dir=None,  # pe
        deg_directional_enc=None,  # this instead will determine the in_channels_ide
        # in_channels_xyz=63,  # do the embedding outside as before
        # in_channels_dir=27,  # in_channels_dir is different from in_channels_ide
        skips=[4],
        skips_appearance=None,
        rgb_sigmoid_extend_epsilon=None,
        density_fix=None,
        **kwargs,
    ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFNetwork, self).__init__(**kwargs)

        self.minBound = kwargs["minBound"]
        self.maxBound = kwargs["maxBound"]
        config = kwargs["config"]
        self.config = config

        self.D = D
        self.D_appearance = D_appearance
        self.W = W
        self.W_appearance = W_appearance
        self.deg_emb_xyz = deg_emb_xyz
        self.deg_emb_dir = deg_emb_dir
        self.deg_directional_enc = deg_directional_enc
        in_channels_xyz = 6 * deg_emb_xyz + 3
        in_channels_dir = 6 * deg_emb_dir + 3
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.skips_appearance = skips_appearance
        assert rgb_sigmoid_extend_epsilon is not None
        self.rgb_sigmoid_extend_epsilon = rgb_sigmoid_extend_epsilon
        assert density_fix is not None
        # self.density_fix = density_fix
        assert density_fix == False  # We do not support density fix here. 
        # If you wish to reopen it, follow Polat... to inject the related requires_grad_(False) again.

        # fixed hyper parameters
        self.epsilon = 1.0e-6  # avoid divided by zero

        # pe
        self.embeddingXyz = Embedding(deg_emb_xyz)
        self.embeddingDir = Embedding(deg_emb_dir)
        self.embeddingOmegaGlobal = Embedding(deg_emb_dir)  # write as a separate module to emphasize

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
        # self.out_tau_spatial = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        # self.out_color_diffuse_spatial = nn.Linear(W, 3)
        self.out_color_diffuse_spatial = nn.Sequential(
            nn.Linear(W, W), nn.ReLU(True),
            nn.Linear(W, W // 2), nn.ReLU(True),
            nn.Linear(W // 2, 3)
        )
        self.out_b_spatial = nn.Sequential(nn.Linear(W, W_bottleneck), nn.ReLU(True))
        # self.xyz_encoding_final = nn.Linear(W, W)
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
                layer = nn.Linear(W_bottleneck + in_channels_ide + 1 + in_channels_dir, W_appearance)
            elif i in skips_appearance:
                # layer = nn.Linear(W_appearance + W_bottleneck + in_channels_ide + 1 + in_channels_dir, W_appearance)
                layer = nn.Linear(W_appearance + W_bottleneck + in_channels_ide + 1 + in_channels_dir, W_appearance)
            elif i == D_appearance - 1:
                layer = nn.Linear(W_appearance, W_appearance // 2)
            else:
                layer = nn.Linear(W_appearance, W_appearance)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"dir_encoding_{i+1}", layer)
        self.out_color_specular_appearance = nn.Linear(W_appearance // 2, 3)

    def density(self, xyz, if_do_bound_clamp, if_output_only_sigma, **kwargs):
        x_input = xyz
        xyz_embedded = self.embeddingXyz(xyz)
        xyz_ = xyz_embedded  # used for forwarding
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([xyz_embedded, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        # out_tau_spatial = self.out_tau_spatial(xyz_)
        sigma = self.sigma(xyz_)
        out_tau_spatial = trunc_exp(sigma)  # F.softplus(sigma)
        out_tau_spatial = out_tau_spatial[:, 0]
        if if_do_bound_clamp:
            flagInside = (
                (x_input[:, 0] >= self.minBound[0]) & (x_input[:, 0] <= self.maxBound[0]) &
                (x_input[:, 1] >= self.minBound[1]) & (x_input[:, 1] <= self.maxBound[1]) &
                (x_input[:, 2] >= self.minBound[2]) & (x_input[:, 2] <= self.maxBound[2])
            )
            out_tau_spatial = torch.where(flagInside, out_tau_spatial, 0)
        if if_output_only_sigma:
            return out_tau_spatial
        else:
            return out_tau_spatial, xyz_

    def preAppearance(self, x, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]

        input_xyz = x

        if requires_grad:
            xyz = input_xyz.requires_grad_(True)
        else:
            xyz = input_xyz

        tmp = self.density(xyz, if_do_bound_clamp=if_do_bound_clamp, if_output_only_sigma=False)
        out_tau_spatial, xyz_ = tmp[0][:, None], tmp[1]       

        if requires_grad:
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
        out_unormalized_normal_pred_spatial = self.out_unormalized_normal_pred_spatial(xyz_)
        out_normal_pred_spatial = torch.div(
            out_unormalized_normal_pred_spatial,
            torch.clamp(
                torch.norm(out_unormalized_normal_pred_spatial, dim=1, p=2)[:, None],
                min=self.epsilon,
            )
        )

        preAppearancePredictionDict = {
            "out_tau_spatial": out_tau_spatial[:, 0],  # tau, which is the density (not yet multiplied with 10)
            "xyz_": xyz_,  # the feat
            "out_roughness_spatial": out_roughness_spatial,  # rho, which is kappa_inv
            "out_normal_pred_spatial": out_normal_pred_spatial,  # n2
            "out_normal_from_tau_spatial": out_normal_from_tau_spatial,  # n1
        }

        return preAppearancePredictionDict

    def forward(self, x, d, **kwargs):
        if_do_bound_clamp = kwargs["if_do_bound_clamp"]
        requires_grad = kwargs["requires_grad"]
        lgtInputPoints = kwargs["lgtInputPoints"]

        input_xyz, input_dir = x, d

        preAppearancePredictionDict = self.preAppearance(
            x,
            if_do_bound_clamp=if_do_bound_clamp,
            requires_grad=requires_grad,
        )

        fullPredictionDict = self.appearance(
            x, d, preAppearancePredictionDict, lgtInputPoints=lgtInputPoints)

        return fullPredictionDict

    def appearance(self, x, d, preAppearancePredictionDict, **kwargs):

        hints_pointlight_opacities = kwargs["hints_pointlight_opacities"]

        xyz_ = preAppearancePredictionDict["xyz_"]
        lgtInputPoints = kwargs["lgtInputPoints"]

        input_xyz, input_dir = x, d

        out_color_diffuse_spatial = self.out_color_diffuse_spatial(xyz_)
        out_b_spatial = self.out_b_spatial(xyz_)
        out_tint_spatial = self.out_tint_spatial(xyz_)

        out_roughness_spatial = preAppearancePredictionDict["out_roughness_spatial"]
        out_normal_pred_spatial = preAppearancePredictionDict["out_normal_pred_spatial"]

        # appearance
        # out_b_spatial = out_b_spatial / 10.  # numerical concerns

        # viewdir_normalized = torch.div(
        #     input_dir,
        #     torch.clamp(torch.norm(input_dir, dim=1, p=2)[:, None], min=self.epsilon)
        # )
        # Shoule have already been normalized
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

        lgtE = lgtInputPoints["lgtE"]
        objCentroid = lgtInputPoints["objCentroid"]
        omegaGlobal = F.normalize(lgtE - objCentroid)  # This is different from the old convention
        omegaGlobal_encoded = self.embeddingOmegaGlobal(omegaGlobal)

        """
        lgtOmegaInput = lgtInputPoints["omegaInput"]  # Note this always refers to the global omegaInput. It is not about local omegaInput.
        lgtOmegaInput_encoded = self.embeddingOmegaGlobal(lgtOmegaInput)
        """

        # input_dir_embedded = self.embeddingDir(input_dir)
        # out_normal_pred_spatial_embedded = self.embeddingDir(out_normal_pred_spatial)
        # input_olat_embedded = self.embeddingDir(input_omega)
        # forwarding = torch.cat([
        #     out_b_spatial,
        #     omega_r_encoded, dot, input_olat_embedded], 1)
        forwarding = torch.cat([out_b_spatial, omega_r_encoded, dot, omegaGlobal_encoded], 1)
        for i in range(self.D_appearance):
            if i in self.skips_appearance:
                forwarding = torch.cat(
                    [forwarding, out_b_spatial,
                     omega_r_encoded, dot, omegaGlobal_encoded], 1
                     # omega_r_encoded, dot, input_olat_embedded], 1
                )
            forwarding = getattr(self, f"dir_encoding_{i+1}")(forwarding)
        out_color_specular_appearance = self.out_color_specular_appearance(forwarding)

        # Suggest to put more layers before getting out_color_diffuse_spatial, to make the numerical better
        x = out_color_diffuse_spatial + out_tint_spatial * out_color_specular_appearance
        x = x - 1.0
        thre = float(10.0)
        out_hdr = torch.where(
            x <= thre,
            torch.exp(torch.clamp(x, max=thre)),
            float(math.e ** thre + 0.01) + (x - thre),
        )
        # out_ldr = torch.sigmoid(x) * (1.0 + 2.0 * self.rgb_sigmoid_extend_epsilon) - self.rgb_sigmoid_extend_epsilon

        # flag_normal_from_tau = out_tau_spatial < 0  # False is OK

        # return out_tau_spatial[:, 0], out_hdr, out_normal_pred_spatial, out_normal_from_tau_spatial, flag_normal_from_tau[:, 0]

        fullPredictionDict = preAppearancePredictionDict
        fullPredictionDict["out_hdr"] = out_hdr

        return fullPredictionDict

    def color(self, **kwargs):
        raise NotImplementedError

    def get_params(self, **kwargs):
        raise NotImplementedError
