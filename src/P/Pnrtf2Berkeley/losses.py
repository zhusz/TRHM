# Written with the help of codebase of https://github.com/kwea123/nerf_pl, released
# under the following license:
#
# MIT License
#
# Copyright (c) 2020 Quei-An Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn


class ColorLoss(nn.Module):
    def __init__(self, coef=1, masked_in_weight=None, masked_out_weight=None):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="sum")
        assert masked_in_weight
        self.masked_in_weight = masked_in_weight
        assert masked_out_weight
        self.masked_out_weight = masked_out_weight

    def forward(self, inputs, targets, valid_mask):
        valid_mask_3 = valid_mask[:, None].repeat(1, 3)
        loss = (
            self.masked_in_weight
            * self.loss(
                inputs["rgb_coarse"][valid_mask_3 == 1], targets[valid_mask_3 == 1]
            )
            + self.masked_out_weight
            * self.loss(
                inputs["rgb_coarse"][valid_mask_3 == 0], targets[valid_mask_3 == 0]
            )
        ) / (valid_mask_3.shape[0] * 3)
        if "rgb_fine" in inputs:
            loss += (
                self.masked_in_weight
                * self.loss(
                    inputs["rgb_fine"][valid_mask_3 == 1], targets[valid_mask_3 == 1]
                )
                + self.masked_out_weight
                * self.loss(
                    inputs["rgb_fine"][valid_mask_3 == 0], targets[valid_mask_3 == 0]
                )
            ) / (valid_mask_3.shape[0] * 3)

        return self.coef * loss


class HdrLoss(nn.Module):
    def __init__(self, coef=1, masked_in_weight=None, masked_out_weight=None):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="sum")
        assert masked_in_weight
        self.masked_in_weight = masked_in_weight
        assert masked_out_weight
        self.masked_out_weight = masked_out_weight

    def forward(self, inputs, targets, valid_mask):
        valid_mask_3 = valid_mask[:, None].repeat(1, 3)
        loss = (
            self.masked_in_weight
            * self.loss(
                inputs["hdr_coarse"][valid_mask_3 == 1], targets[valid_mask_3 == 1]
            )
            + self.masked_out_weight
            * self.loss(
                inputs["hdr_coarse"][valid_mask_3 == 0], targets[valid_mask_3 == 0]
            )
        ) / (valid_mask_3.shape[0] * 3)
        if "hdr_fine" in inputs:
            loss += (
                self.masked_in_weight
                * self.loss(
                    inputs["hdr_fine"][valid_mask_3 == 1], targets[valid_mask_3 == 1]
                )
                + self.masked_out_weight
                * self.loss(
                    inputs["hdr_fine"][valid_mask_3 == 0], targets[valid_mask_3 == 0]
                )
            ) / (valid_mask_3.shape[0] * 3)

        return self.coef * loss


class HdrRawNerfLoss(nn.Module):
    def __init__(self, coef=1, eps=None, masked_in_weight=None, masked_out_weight=None):
        super().__init__()
        self.coef = coef
        self.eps = eps  # the raw-nerf paper recommends 1.0e-3
        assert eps is not None
        assert masked_in_weight
        self.masked_in_weight = masked_in_weight
        assert masked_out_weight
        self.masked_out_weight = masked_out_weight

    def loss(self, inputs, targets, masks):  # raw-nerf
        y_hat = inputs[masks]
        y = targets[masks]
        return (((y_hat - y) ** 2) / (torch.abs(y_hat).detach() + self.eps)).sum()

    def forward(self, inputs, targets, valid_mask):
        valid_mask_3 = valid_mask[:, None].repeat(1, 3)
        loss = (
            self.masked_in_weight
            * self.loss(
                inputs["hdr_coarse"], targets, valid_mask_3 == 1
            )
            + self.masked_out_weight
            * self.loss(
                inputs["hdr_coarse"], targets, valid_mask_3 == 0
            )
        ) / (valid_mask_3.shape[0] * 3)
        if "hdr_fine" in inputs:
            loss += (
                self.masked_in_weight
                * self.loss(
                    inputs["hdr_fine"], targets, valid_mask_3 == 1
                )
                + self.masked_out_weight
                * self.loss(
                    inputs["hdr_fine"], targets, valid_mask_3 == 0
                )
            ) / (valid_mask_3.shape[0] * 3)

        return self.coef * loss



class HdrClipRawNerfLoss(nn.Module):
    def __init__(self, coef=1, eps=None, masked_in_weight=None, masked_out_weight=None):
        super().__init__()
        self.coef = coef
        self.eps = eps  # the raw-nerf paper recommends 1.0e-3
        assert eps is not None
        assert masked_in_weight
        self.masked_in_weight = masked_in_weight
        assert masked_out_weight
        self.masked_out_weight = masked_out_weight

        self.hdr_clip = 2.5

    def loss(self, inputs, targets, masks):  # raw-nerf
        y_hat = inputs[masks]
        y = targets[masks]
        y = torch.clamp(y, max=self.hdr_clip)
        return (((y_hat - y) ** 2) / (torch.abs(y_hat).detach() + self.eps)).sum()

    def forward(self, inputs, targets, valid_mask):
        valid_mask_3 = valid_mask[:, None].repeat(1, 3)
        loss = (
            self.masked_in_weight
            * self.loss(
                inputs["hdr_coarse"], targets, valid_mask_3 == 1
            )
            + self.masked_out_weight
            * self.loss(
                inputs["hdr_coarse"], targets, valid_mask_3 == 0
            )
        ) / (valid_mask_3.shape[0] * 3)
        if "hdr_fine" in inputs:
            loss += (
                self.masked_in_weight
                * self.loss(
                    inputs["hdr_fine"], targets, valid_mask_3 == 1
                )
                + self.masked_out_weight
                * self.loss(
                    inputs["hdr_fine"], targets, valid_mask_3 == 0
                )
            ) / (valid_mask_3.shape[0] * 3)

        return self.coef * loss


class MaskLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, inputs, masks):
        assert torch.all((masks == 0) | (masks == 1))
        loss = (
            (1.0 - masks.float())
            * self.loss(
                inputs["opacity_coarse"], torch.zeros_like(inputs["opacity_coarse"])
            )
        ).mean()

        """
        import ipdb
        ipdb.set_trace()
        print(1 + 1)
        """

        if "rgb_fine" in inputs:
            loss += (
                (1.0 - masks.float())
                * self.loss(
                    inputs["opacity_fine"], torch.zeros_like(inputs["opacity_fine"])
                )
            ).mean()

        return self.coef * loss


# loss_dict = {"color": ColorLoss, "mask": MaskLoss}
