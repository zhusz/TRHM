import torch
import numpy as np
from kornia.losses.ssim import ssim_loss as ssim_loss_kornia
from socket import gethostname
import lpips

from codes_py.toolbox_3D.nerf_metrics_v1 import psnr
from codes_py.toolbox_graphics.tonemap_v1 import (  # noqa
    tonemap_srgb_to_rgb,
)


class BenchmarkingPsnrLdrRenderingNerfBlenderFuncObj(object):  # This is all for initiailizing lpips
    def __init__(self):
        self.lpips_loss_fn_vgg = None

    def __call__(self, bsv0, **kwargs):

        ifRequiresPredictingHere = kwargs["ifRequiresPredictingHere"]
        ifRequiresBenchmarking = kwargs["ifRequiresBenchmarking"]
        ifRequiresDrawing = kwargs["ifRequiresDrawing"]
        cudaDevice = kwargs["cudaDevice"]
        datasetObj = kwargs["datasetObj"]
        iterCount = kwargs["iterCount"]

        ssimWindow = kwargs["ssimWindow"]

        # lpips initialize (cached)
        if self.lpips_loss_fn_vgg is None:
            self.lpips_loss_fn_vgg = lpips.LPIPS(net="vgg").to(cudaDevice)  # We assume all the invoke of this functions bring in the same "cudaDevice" value

        # info putting
        bsv0["iterCount"] = int(iterCount)

        # labelled rgb to img (convenient for htmls)
        winWidth = bsv0["winWidth"]
        winHeight = bsv0["winHeight"]

        # put in geometry evaluation range (for visualizing the density)
        bsv0["minBound"] = datasetObj.datasetConf["minBound"]
        bsv0["maxBound"] = datasetObj.datasetConf["maxBound"]

        # doPred
        if ifRequiresPredictingHere:
            bsv0 = kwargs["doPred0Func"](bsv0, **kwargs)

        if ifRequiresBenchmarking:
            bsv0["imgs"] = bsv0["rgbs"].reshape((winHeight, winWidth, 3))
            bsv0["ldrs"] = tonemap_srgb_to_rgb(torch.from_numpy(bsv0["hdrs"])).numpy()
            bsv0["imgldrs"] = bsv0["ldrs"].reshape((winHeight, winWidth, 3))
            bsv0["imghdrs"] = bsv0["hdrs"].reshape((winHeight, winWidth, 3))

            hdrFinePred0_thgpu = torch.from_numpy(bsv0["imghdrFinePred"].reshape((-1, 3))).to(cudaDevice)
            ldrFinePred0_thgpu = torch.from_numpy(bsv0["ldrFinePred"]).to(cudaDevice)

            hdrs0_thgpu = torch.from_numpy(bsv0["imghdrs"].reshape((-1, 3))).to(cudaDevice)
            ldrs0_thgpu = torch.from_numpy(bsv0["ldrs"]).to(cudaDevice)
    
            ldrFinePred_transposed_thgpu = ldrFinePred0_thgpu.reshape(1, winHeight, winWidth, 3).permute(0, 3, 1, 2)
            ldrs_transposed_thgpu = ldrs0_thgpu.reshape(1, winHeight, winWidth, 3).permute(0, 3, 1, 2)
    
            # Draw delta (pixel wise psnr)
            bsv0["pixelwisePSNRFine"] = psnr(ldrFinePred0_thgpu, ldrs0_thgpu, reduction=None).detach().cpu().numpy().reshape((winHeight, winWidth, 3)).mean(2)
    
            # benchmarking
            bsv0["finalBenRuntimeFull"] = float(bsv0.get("runtimeFull", float("nan")))
            bsv0["finalBenRuntimeRelight"] = float(bsv0.get("runtimeRelight", float("nan")))
            bsv0["finalBenRuntimeMachine"] = bsv0.get("runtimeMachine", gethostname())
            bsv0["finalBenPSNRFine"] = psnr(ldrFinePred0_thgpu, ldrs0_thgpu).detach().cpu().numpy()
            if ("pixelID_highlight" not in bsv0.keys()) or ("pixelID_sublight" not in bsv0.keys()):
                tt = np.array([], dtype=np.int32)
            else:
                assert np.intersect1d(bsv0["pixelID_highlight"], bsv0["pixelID_sublight"]).shape == (0, )
                tt = torch.from_numpy(
                    np.union1d(bsv0["pixelID_highlight"], bsv0["pixelID_sublight"]),
                ).long().to(cudaDevice)
            if tt.shape[0] == 0:
                bsv0["finalBenPSNRHighSubLightFine"] = np.nan
                bsv0["finalBenPSNRHighSubLightHDRFine"] = np.nan
            else:
                bsv0["finalBenPSNRHighSubLightFine"] = psnr(
                    ldrFinePred0_thgpu[tt, :],
                    ldrs0_thgpu[tt, :],
                ).detach().cpu().numpy()
                assert np.isfinite(bsv0["finalBenPSNRHighSubLightFine"])
                bsv0["finalBenPSNRHighSubLightHDRFine"] = psnr(
                    torch.clamp(hdrFinePred0_thgpu[tt, :], 0, 1),
                    torch.clamp(hdrs0_thgpu[tt, :], 0, 1),
                ).detach().cpu().numpy()
                assert np.isfinite(bsv0["finalBenPSNRHighSubLightHDRFine"])
            if "pixelID_highlight" in bsv0.keys():
                tt_highlight = torch.from_numpy(bsv0["pixelID_highlight"]).long().to(cudaDevice)
            else:
                tt_highlight = np.array([], dtype=np.int32)
            if (
                (tt_highlight.shape[0] == 0) or
                (bsv0["dataset"] == "renderingNerfBlender58Pointk8L10V10ben" and bsv0["index"] in [342, 371, 500, 502, 516, 523, 554, 557, 573, 591, 593]) or  # these test samples got perfect highlight results (all RGB 255 saturated for both pred_NRTF and label, resulting in inf PSNR) so we exclude these test samples
                (bsv0["dataset"] == "renderingNerfBlender58Pointk8L10V10ben" and bsv0["index"] in [40, 505, 340, 521, 534, 536, 545, 550, 559, 565, 574, 580, 581]) or  # Ours
                (bsv0["dataset"] == "renderingNerfBlender58Pointk8L10V10ben" and bsv0["index"] in [552]) or  # NRHints
                (bsv0["dataset"] == "renderingNerfBlender58Pointk8L10V10ben" and bsv0["index"] in [511])  # NeRF-tex
            ):  # there are too many of such kind, so we won't evaluate this on the synthetic datasets
                bsv0["finalBenPSNRHighLightFine"] = np.nan
                bsv0["finalBenPSNRHighLightHDRFine"] = np.nan
            else:
                bsv0["finalBenPSNRHighLightFine"] = psnr(
                    ldrFinePred0_thgpu[tt_highlight, :],
                    ldrs0_thgpu[tt_highlight, :],
                ).detach().cpu().numpy()
                assert np.isfinite(bsv0["finalBenPSNRHighLightFine"])
                bsv0["finalBenPSNRHighLightHDRFine"] = psnr(
                    torch.clamp(hdrFinePred0_thgpu[tt_highlight, :], 0, 1),
                    torch.clamp(hdrs0_thgpu[tt_highlight, :], 0, 1),
                ).detach().cpu().numpy()
                assert np.isfinite(bsv0["finalBenPSNRHighLightHDRFine"])
            bsv0["finalBenSSIMFine"] = 1. - 2. * ssim_loss_kornia(ldrFinePred_transposed_thgpu, ldrs_transposed_thgpu, ssimWindow).detach().cpu().numpy()
            with torch.no_grad():
                bsv0["finalBenLPIPSFine"] = float(self.lpips_loss_fn_vgg(ldrFinePred_transposed_thgpu, ldrs_transposed_thgpu, normalize=True).detach().cpu().numpy())
                bsv0["finalBenLPIPSunnormalizedFine"] = float(self.lpips_loss_fn_vgg(ldrFinePred_transposed_thgpu, ldrs_transposed_thgpu).detach().cpu().numpy())

        return bsv0


benchmarkingPsnrLdrRenderingNerfBlenderFunc = BenchmarkingPsnrLdrRenderingNerfBlenderFuncObj()
# This is an OK temporary solution (let external reference to just import this obj) just for this project, as whatever B or P running just need one version of this obj.
# However, furture more complicated projects might invalidate this. 
# So in the future, you may want to put this whole benchmarking function as a class, and use its class form directly in external references.
