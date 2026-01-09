import torch
import torch.nn.functional as F
import numpy as np


def tonemap_srgb_to_rgb(hdr):
    thre = 0.0031308
    a = 0.055

    # assert torch.all(hdr > -1)

    gamma_input = torch.log(F.relu(hdr) + 1)
    ldr = torch.where(
        gamma_input <= thre,
        gamma_input * 12.92,
        (1 + a) * torch.pow(gamma_input, 1.0 / 2.4) - a,
    )
    return ldr


def tonemap_srgb_to_rgb_np(hdr):
    thre = 0.0031308
    a = 0.055

    # assert torch.all(hdr > -1)

    gamma_input = np.log(np.clip(hdr, a_min=0, a_max=np.inf) + 1)
    ldr = np.where(
        gamma_input <= thre,
        gamma_input * 12.92,
        (1 + a) * np.power(gamma_input, 1.0 / 2.4) - a,
    )
    return ldr


def grad_tonemap_srgb_to_rgb(hdr):
    # If you wish to detach (e.g. when computing the weight of the loss in raw-nerf)
    # you should do this outside of this function on your own

    thre = 0.0031308
    a = 0.055

    gamma_input = torch.log(F.relu(hdr) + 1)
    grad_tonemap = torch.where(
        gamma_input <= thre,
        12.92 / (1.0 + hdr),
        (1 + a) / 2.4 * torch.pow(gamma_input, 1.0 / 2.4 - 1.0) / (1.0 + hdr),
    )
    return grad_tonemap
