import torch
import os
import copy
import socket
from collections import OrderedDict
from multiprocessing import shared_memory
from configs_registration import getConfigGlobal
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from Bprelight4.testDataEntry.renderingNerfBlender.renderingLightStageLfhyper import RenderingLightStageLfhyper
from Bprelight4.testDataEntry.renderingNerfBlender.renderingNerfBlenderDatasetLfhyper import RenderingNerfBlenderDatasetLfhyper
from Bprelight4.testDataEntry.testDataEntryPool import getTestDataEntryDict
from ..trainer import Trainer
from ..tngp1.nerf.refrelight_large_exp_only_r_hint15hyperfull import NeRFNetwork as NeRFNetwork15
from ..tngp1.nerf.renderer_hint15hyperfull import NeRFRenderer, soft_clamp
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
# import nerfacc
from codes_py.toolbox_nerfacc.estimators.occ_grid import OccGridEstimator
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.rotations_v1 import ELU2cam
from torch.cuda.amp.grad_scaler import GradScaler  # You need to import nerfacc first before this one
from collections import defaultdict
from functools import partial
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb
from codes_py.toolbox_3D.nerf_metrics_v1 import psnr
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc
import numpy as np
import math
import time


bt = lambda s: s[0].upper() + s[1:]


class DTrainer(Trainer):
    def metaDataLoading(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaDataLoading")

        datasetConfDict = config.datasetConfDict

        datasetObjDict = OrderedDict([])

        for datasetConf in datasetConfDict.values():
            if datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                datasetObj = RenderingNerfBlenderDatasetLfhyper(
                    datasetConf,
                    if_need_metaDataLoading=True,
                    cudaDevice="cuda:0",
                )
                datasetObjDict[datasetConf["dataset"]] = datasetObj
            elif datasetConf["class"] == "RenderingLightStageLfhyper":
                datasetObj = RenderingLightStageLfhyper(
                    datasetConf,
                    if_need_metaDataLoading=True,
                    cudaDevice=self.cudaDeviceForAll,
                )
                datasetObjDict[datasetConf["dataset"]] = datasetObj
            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                datasetObj = PRenderingLightStageOrigianl3(datasetConf)
                datasetObjDict[datasetConf["dataset"]] = datasetObj
            else:
                raise NotImplementedError(
                    "Unknown dataset class: %s" % datasetConf["class"]
                )

        self.datasetObjDict = datasetObjDict

    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        model15 = NeRFNetwork15(config=config)
        model15 = model15.to(self.cudaDeviceForAll)

        renderer = NeRFRenderer(
            config=config,
        )
        renderer = renderer.to(self.cudaDeviceForAll)

        occupancy_grid = OccGridEstimator(
            roi_aabb=torch.FloatTensor([-1, -1, -1, 1, 1, 1]),
            resolution=128,
            levels=4,
        )
        occupancy_grid = occupancy_grid.to(self.cudaDeviceForAll)

        models = {}
        models["model17"] = model15
        models["renderer"] = renderer
        models["occGrid"] = occupancy_grid
        meta = {
            "nonLearningModelNameList": ["occGrid", "renderer", "model16"],
            "nonLearningButToSave": ["occGrid"],
            "if_add_in_hdr3": config.if_add_in_hdr3,
        }
        models["meta"] = meta

        self.models = models

    def setupMonitor(self):
        self.monitorImmediateFlowLogDir = self.logDir + "monitorImmediateFlow/"
        self.monitorTrainLogDir = self.logDir + "monitorTrain/"
        self.monitorValLogDir = self.logDir + "monitorVal/"

        # Immediate Flow
        self.htmlStepperImmediateFlow = HTMLStepper(
            self.monitorImmediateFlowLogDir, 40, "monitorImmediateFlow"
        )
        datasetConf_testingOnTraining = copy.deepcopy(
            list(self.config.datasetConfDict.values())[0]
        )
        datasetConf_testingOnTraining["singleImageMode"] = True
        if datasetConf_testingOnTraining["class"] == "RenderingNerfBlenderDatasetLfhyper":
            self.testingOnTrainingDatasetObj = RenderingNerfBlenderDatasetLfhyper(
                datasetConf_testingOnTraining,
                if_need_metaDataLoading=False,
                cudaDevice=self.cudaDeviceForAll,
            )
        elif datasetConf_testingOnTraining["class"] == "RenderingLightStageLfhyper":
            self.testingOnTrainingDatasetObj = RenderingLightStageLfhyper(
                datasetConf_testingOnTraining,
                if_need_metaDataLoading=False,
                cudaDevice=self.cudaDeviceForAll,
            )
        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf_testingOnTraining["class"])

        # Val Vis
        self.htmlStepperVal = HTMLStepper(
            self.monitorValLogDir, 40, "monitorVal"
        )
        datasetConf_testing = copy.deepcopy(list(self.config.datasetConfDict.values())[0])
        datasetConf_testing["singleImageMode"] = True
        if datasetConf_testing["class"] == "RenderingNerfBlenderDatasetLfhyper":
            testDataNickName = "%sCase%dSplit%s" % (
                list(self.config.datasetConfDict.values())[0]["dataset"],
                list(self.config.datasetConfDict.values())[0]["caseID"],
                "Val",
            )
        elif datasetConf_testing["class"] == "RenderingLightStageLfhyper":
            testDataNickName = "%sCase%dSplit%s" % (
                list(self.config.datasetConfDict.values())[0]["dataset"],
                list(self.config.datasetConfDict.values())[0]["caseID"],
                "Val",
            )
        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf_testing["class"])
        
        self.testDataEntry = getTestDataEntryDict(
            wishedTestDataNickName=[testDataNickName],
            fyiDatasetConf=datasetConf_testing,
            cudaDevice=self.cudaDeviceForAll,
        )[testDataNickName]

    def stepBatch(self, **kwargs):
        # What to do in stepBatch? (In contrast to batchPreProcessing)
        #   Randomness is allowed - after the end of this class method running, the identity of each 
        #       training samples are established and fixed thereon. No more sampling randomness after 
        #       the finishing of the running of this class method
        #   multi-threading is allowed (though not used in this approach)
        #   GPUs is not allowed - this is exclusive with multi-threading allowing

        # However, this approach is an exception - batchPreprocessing will be a "pass"-only class method!
        with torch.no_grad():
            cudaDevice = self.cudaDeviceForAll

            config = self.config
            tmp = OrderedDict([])
            for dataset in config.datasetConfDict.keys():
                datasetConf = config.datasetConfDict[dataset]
                datasetObj = self.datasetObjDict[dataset]
                # elif datasetConf["class"] == "RenderingLightStageLfhyper":
                caseID = datasetConf["caseID"]

                # sample the lgt
                lgtDatasetCache = datasetObj.lgtDatasetCache
                mTrain_lgt = lgtDatasetCache.mTrain
                indTrain_lgt = lgtDatasetCache.indTrain
                i_lgt = np.random.choice(
                    indTrain_lgt.shape[0],
                    size=(datasetConf["batchSizeEnvmapLgtPerProcess"]),
                    replace=False,
                )
                ind_lgt = indTrain_lgt[i_lgt]
                assert np.all(lgtDatasetCache.A0["flagSplit"][ind_lgt] == 1)
                envmap_normalized_thgpu = lgtDatasetCache.queryEnvmap(
                    ind_lgt, if_augment=True, if_return_all=False).float()
                assert envmap_normalized_thgpu.shape == (
                    datasetConf["batchSizeEnvmapLgtPerProcess"], 16, 32, 3
                )
                # To update to the colorful envmap hyper training
                envmap_to_record_thgpu = envmap_normalized_thgpu  # (B_lgt, 16, 32, 3)
                alpha_min, alpha_max = 0.4, 1.6
                alpha_thgpu = torch.rand(envmap_to_record_thgpu.shape[0], dtype=torch.float32, device=envmap_to_record_thgpu.device)
                envmap_to_apply_thgpu = (envmap_to_record_thgpu / 16 / 32 * 3) * alpha_thgpu[:, None, None, None]
                envmap_to_apply_thgpu = envmap_to_apply_thgpu.reshape((envmap_to_apply_thgpu.shape[0], 16 * 32, 3))

                # sample the rays (and also combined with the lgt)
                if datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                    mbtot_per_process = int(datasetObj.all_fg_indexID.shape[0])
                    nOlat = 16 * 32
                    assert datasetObj.all_fg_indexID.shape == (mbtot_per_process,)
                    assert datasetObj.all_fg_pixelID.shape == (mbtot_per_process,)
                    shared_names = [shm.name for shm, _ in datasetObj.shared_memories]
                    shm3 = shared_memory.SharedMemory(name=shared_names[3])
                    all_fg_hdrs = np.ndarray(datasetObj.shared_shapes[3], dtype=datasetObj.shared_dtypes[3], buffer=shm3.buf)
                    assert all_fg_hdrs.shape == (mbtot_per_process, nOlat, 3)
                    ind_fg = np.random.randint(
                        0,
                        mbtot_per_process,
                        datasetConf["batchSizeFgPerProcess"],
                        dtype=np.int64,
                    )
                elif datasetConf["class"] == "RenderingLightStageLfhyper":
                    mbtot_per_process = int(datasetObj.all_fg_groupViewID.shape[0])
                    nOlat = 331
                    assert datasetObj.all_fg_groupViewID.shape == (mbtot_per_process,)
                    assert datasetObj.all_fg_pixelID.shape == (mbtot_per_process,)
                    assert datasetObj.all_fg_hdrs.shape == (mbtot_per_process, nOlat, 3)
                    assert datasetObj.all_fg_omegas.shape == (mbtot_per_process, nOlat, 3)
                    ind_fg = np.random.randint(
                        0,
                        mbtot_per_process,
                        datasetConf["batchSizeFgPerProcess"],
                        dtype=np.int64,
                    )
                    omegas_fg_thgpu = torch.from_numpy(datasetObj.all_fg_omegas[ind_fg, :, :]).to(cudaDevice)
                else:
                    raise ValueError("Unknown datasetConf class: %s" % datasetConf["class"])
                
                assert datasetConf["white_back"] == False
                if datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                    hdrs_fg_thgpu = torch.from_numpy(all_fg_hdrs[ind_fg, :, :]).to(cudaDevice)
                elif datasetConf["class"] == "RenderingLightStageLfhyper":
                    hdrs_fg_thgpu = torch.from_numpy(datasetObj.all_fg_hdrs[ind_fg, :, :]).to(cudaDevice)
                else:
                    raise ValueError("Unknown datasetConf class: %s" % datasetConf["class"])
                insideBatchInd_lgt_thgpu = torch.randint(
                    low=0,
                    high=int(envmap_to_apply_thgpu.shape[0]),
                    size=(ind_fg.shape[0],),
                    dtype=torch.int64,
                    device=cudaDevice,
                    requires_grad=False,
                )
                with torch.no_grad():
                    if datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                        hdrs_fg_applied_thgpu = (
                            hdrs_fg_thgpu # (B_fg_rays, 16*32, 3)
                            *
                            envmap_to_apply_thgpu[insideBatchInd_lgt_thgpu, :, :]  # (B_fg_rays, 16*32, 3)
                        ).sum(1)
                    elif datasetConf["class"] == "RenderingLightStageLfhyper":
                        # phi_ and theta_ size: (B_rays, 331)
                        phi = torch.atan2(omegas_fg_thgpu[:, :, 1], omegas_fg_thgpu[:, :, 0])  # (-pi, pi)
                        phi = torch.pi - phi  # (0, 2*pi)
                        phi_ = phi / torch.pi - 1  # (-1, 1)
                        assert torch.all(torch.abs(omegas_fg_thgpu[:, :, 2]) < 1.00001)
                        theta = torch.acos(torch.clamp(omegas_fg_thgpu[:, :, 2], min=-1, max=1))  # [0, np.pi]
                        theta_ = theta / torch.pi * 2 - 1  # [-1, 1]
                        tmp = -torch.ones(ind_fg.shape[0], nOlat, 3, dtype=torch.float32, device=phi_.device)
                        for i in range(ind_lgt.shape[0]):
                            # print(i)
                            t = torch.where(insideBatchInd_lgt_thgpu == i)[0]
                            tm = t.shape[0]
                            tmp_current = F.grid_sample(  # (1, 3, tm, 331)
                                # (1, 3, 16, 32)
                                envmap_to_apply_thgpu[i : i + 1, :, :].view(1, 16, 32, 3).permute(0, 3, 1, 2),
                                # (1, tm, 331, 2)
                                torch.stack([phi_[t, :], theta_[t, :]], 2)[None, :, :, :],
                                mode="bilinear",
                                padding_mode="zeros",
                                align_corners=False,
                            )[0, :, :, :].permute(1, 2, 0)  # (tm, 331, 3)
                            tmp[t, :, :] = tmp_current
                            del t, tm, tmp_current
                        assert torch.all(tmp > -0.01)
                        hdrs_fg_applied_thgpu = (
                            hdrs_fg_thgpu  # (B_fg_rays, 331, 3)
                            * 
                            tmp  # (B_fg_rays, 331, 3)
                        ).sum(1)  # (B_fg_rays, 3)
                    else:
                        raise ValueError("Unknown datasetConf class: %s" % datasetConf["class"])
                envmap_alphaed_thgpu = envmap_to_record_thgpu * alpha_thgpu[:, None, None, None]  # (B_lgt, 16, 32, 3)  # update to the colorful envmap

                # rays info (as usual) including bg and mg
                if datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                    ind_bg = np.random.randint(
                        0, int(datasetObj.all_bg_pixelID.shape[0]),
                        datasetConf["batchSizeBgPerProcess"], dtype=np.int64
                    )
                    viewID = np.concatenate([
                        datasetObj.all_fg_indexID[ind_fg],  # view only, has not yet multiplied with nOlat
                        datasetObj.all_bg_indexID[ind_bg],  # view only, has not yet multiplied with nOlat
                    ], 0)
                    pixelID = np.concatenate([
                        datasetObj.all_fg_pixelID[ind_fg],
                        datasetObj.all_bg_pixelID[ind_bg],
                    ], 0)
                    assert datasetObj.A0["m"] % datasetObj.A0["caseTot"] == 0
                    mPerCase_A0 = datasetObj.A0["m"] // datasetObj.A0["caseTot"]
                    # Since in A0, it is olatID-major / viewID-minor,
                    #   we can directly get all the A0 info directly with
                    #   A0[k][mPerCase_A0 * caseID + 0(which is lgtID) * mTrain_view + viewID]
                    focalLengthWidth = datasetObj.A0["focalLengthWidth"][mPerCase_A0 * caseID + 0 + viewID]
                    focalLengthHeight = datasetObj.A0["focalLengthHeight"][mPerCase_A0 * caseID + 0 + viewID]
                    E = datasetObj.A0["E"][mPerCase_A0 * caseID + 0 + viewID]  # In this context, sceneShiftFirst and sceneScaleSecond are both processed later on (which is different from the convention of olat training of synthetic data)
                    L = datasetObj.A0["L"][mPerCase_A0 * caseID + 0 + viewID]
                    U = datasetObj.A0["U"][mPerCase_A0 * caseID + 0 + viewID]
                    w2c = ELU2cam(np.concatenate([E, L, U], 1))
                    c2w = np.linalg.inv(w2c)
                elif datasetConf["class"] == "RenderingLightStageLfhyper":
                    ind_mg = np.random.randint(
                        0, int(datasetObj.all_mg_pixelID.shape[0]),
                        datasetConf["batchSizeMgPerProcess"], dtype=np.int64
                    )
                    ind_bg = np.random.randint(
                        0, int(datasetObj.all_bg_pixelID.shape[0]),
                        datasetConf["batchSizeBgPerProcess"], dtype=np.int64
                    )
                    groupViewID = np.concatenate([
                        datasetObj.all_fg_groupViewID[ind_fg],
                        datasetObj.all_mg_groupViewID[ind_mg],
                        datasetObj.all_bg_groupViewID[ind_bg],
                    ], 0)
                    pixelID = np.concatenate([
                        datasetObj.all_fg_pixelID[ind_fg],
                        datasetObj.all_mg_pixelID[ind_mg],
                        datasetObj.all_bg_pixelID[ind_bg],
                    ], 0)
                    focalLengthWidth = datasetObj.focalLengthWidth[groupViewID]
                    focalLengthHeight = datasetObj.focalLengthHeight[groupViewID]
                    c2w = datasetObj.c2w[groupViewID]
                else:
                    raise ValueError("Unknown datasetConf class: %s" % datasetConf["class"])
                sceneShiftFirst = datasetConf["sceneShiftFirst"]
                sceneScaleSecond = datasetConf["sceneScaleSecond"]
                rowID = pixelID // (datasetObj.winWidth + 2 * datasetConf["rays_background_pad_width"])  # correct
                colID = pixelID % (datasetObj.winWidth + 2 * datasetConf["rays_background_pad_width"])  # correct
                directions_unnormalized_cam = np.stack([
                    (colID.astype(np.float32) + 0.5 - datasetObj.winWidth / 2.0 - datasetConf["rays_background_pad_width"]) / focalLengthWidth,  # correct
                    (rowID.astype(np.float32) + 0.5 - datasetObj.winHeight / 2.0 - datasetConf["rays_background_pad_height"]) / focalLengthHeight,  # correct
                    np.ones((rowID.shape[0],), dtype=np.float32),
                ], 1)
                directions_unnormalized = (c2w[:, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
                directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
                directions_normalized = directions_unnormalized / directions_norm[:, None]
                # random_background_color = np.random.rand(groupViewID.shape[0], 3).astype(np.float32)
                random_background_color = np.zeros((rowID.shape[0], 3), dtype=np.float32)
                rays = np.concatenate([
                    (c2w[:, :3, 3] - sceneShiftFirst[None, :]) * sceneScaleSecond,  # rays_o
                    directions_normalized,  # rays_d
                    np.nan * np.zeros((rowID.shape[0], 4), dtype=np.float32),
                    random_background_color,
                ], 1)

                batch_thgpu = {}
                batch_thgpu["rays_%s" % dataset] = torch.from_numpy(rays).to(cudaDevice)
                batch_thgpu["insideBatchIndLgt_%s" % dataset] = insideBatchInd_lgt_thgpu  # fg only
                if datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                    batch_thgpu["hdrs_%s" % dataset] = torch.cat([
                        hdrs_fg_applied_thgpu,
                        torch.zeros(ind_bg.shape[0], 3, dtype=torch.float32, device=cudaDevice),
                    ], 0)
                    batch_thgpu["indexID_%s" % dataset] = torch.cat([  # Note this one would be multiplied with the envmapID
                        torch.from_numpy(
                            viewID[:ind_fg.shape[0]] * datasetObj.lgtDatasetCache.mTrain + i_lgt[insideBatchInd_lgt_thgpu.detach().cpu().numpy().astype(np.int32)]
                        ).int().to(cudaDevice),
                        torch.from_numpy(
                            viewID[ind_fg.shape[0]:] * datasetObj.lgtDatasetCache.mTrain + 0,  # background does not care which lgt, so always treats ind_lgt to be zero
                        ).int().to(cudaDevice),
                    ], 0)
                    batch_thgpu["valid_mask_%s" % dataset] = torch.cat([
                        torch.ones(ind_fg.shape[0], dtype=bool, device=cudaDevice),
                        torch.ones(ind_bg.shape[0], dtype=bool, device=cudaDevice)
                    ], 0)
                # Note the indexing / flagSplit is very different between class RenderingNerfBlenderDatasetLfhyper and class RenderingLightStageLfhyper
                # The class above (RenderingNerfBlenderDatasetLfhyper) does not contain any flagSplit == 0, and m == mTrain + mVal + mTest, mInvalid is always 0
                # The class below (RenderingLightStageLfhyper) contains flagSplit == 0 (val view x train lgt), and mInvalid > 0
                elif datasetConf["class"] == "RenderingLightStageLfhyper":
                    batch_thgpu["hdrs_%s" % dataset] = torch.cat([
                        hdrs_fg_applied_thgpu,
                        torch.zeros(int(ind_mg.shape[0] + ind_bg.shape[0]), 3, dtype=torch.float32, device=cudaDevice),
                    ], 0)
                    batch_thgpu["indexID_%s" % dataset] = torch.cat([
                        torch.from_numpy(
                            datasetObj.all_fg_groupViewID[ind_fg] * datasetObj.lgtDatasetCache.m
                        ).int().to(cudaDevice)
                        + torch.from_numpy(ind_lgt).int().to(cudaDevice)[insideBatchInd_lgt_thgpu],  # fg
                        torch.from_numpy(
                            datasetObj.all_mg_groupViewID[ind_mg] * datasetObj.lgtDatasetCache.m + 0   
                        ).int().to(cudaDevice),  # mg (assume lgtID is always 0)
                        torch.from_numpy(
                            datasetObj.all_bg_groupViewID[ind_bg] * datasetObj.lgtDatasetCache.m + 0
                        ).int().to(cudaDevice),  # bg (assume lgtID is always 0)
                    ], 0)
                    batch_thgpu["valid_mask_%s" % dataset] = torch.cat([
                        torch.ones(ind_fg.shape[0], dtype=bool, device=cudaDevice),
                        torch.ones(ind_mg.shape[0] + ind_bg.shape[0], dtype=bool, device=cudaDevice)
                    ], 0)
                else:
                    raise ValueError("Unknown datasetConf class: %s" % datasetConf["class"])
                batch_thgpu["random_background_color_%s" % dataset] = torch.from_numpy(random_background_color).float().to(cudaDevice)
                batch_thgpu["envmapAlphaed_%s" % "lgt"] = envmap_alphaed_thgpu  # torch.from_numpy(envmap_alphaed).to(cudaDevice)

        return None, batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        return batch_thgpu  # do nothing

    def forwardNetNeRF(self, batch_thgpu, **kwargs):
        models = kwargs["models"]
        config = kwargs["config"]
        dataset = kwargs["dataset"]
        iterCount = kwargs["iterCount"]

        wl = config.wl

        self.assertModelsTrainingMode(models)

        if (iterCount > 0) and (config.memory_bank_read_out_freq > 0) and (iterCount % config.memory_bank_read_out_freq == 0):
            assert len(self.datasetObjDict) == 1
            datasetObj = list(self.datasetObjDict.values())[0]
            datasetObj.doMemoryBankReadOutOnce(specifiedGroupViewID=None)

        if iterCount < config.empty_cache_stop_iter:
            torch.cuda.empty_cache()

        _dataset = "_" + dataset
        _lgt = "_" + "lgt"
        # For this mode, you need to modify the rays[:, 10:13]
        # (where valid_mask == False pixels need to be set to 1 or 0)
        lgtInput = {
            # "lgtE": batch_thgpu["lgtE" + _dataset],
            # "objCentroid": batch_thgpu["objCentroid" + _dataset],
            "envmapAlphaed": batch_thgpu["envmapAlphaed" + _lgt],
            "insideBatchInd_lgt": batch_thgpu["insideBatchIndLgt" + _dataset],
        }
        results = self.forwardRendering(
            batch_thgpu["rays" + _dataset],
            models=models,
            config=config,
            iterCount=iterCount,
            cpu_ok=False,
            batch_thgpu=batch_thgpu,  # debug purpose
            force_all_rays=False,
            requires_grad=True,
            if_query_the_expected_depth=False,
            lgtInput=lgtInput,
            callFlag="forwardNetNeRF",
            logDir=self.logDir,
            lgtMode="pointLightMode",
            envmap0_thgpu=None,
            envmapNormalizingFactor=None,
            if_only_predict_hdr2=False,
        )

        # We shall be disable any use of background color, as in here, density / geometry is no longer updated
        assert torch.all(batch_thgpu["random_background_color_%s" % dataset] == 0)
        assert torch.all(batch_thgpu["rays_%s" % dataset][:, 10:13] == 0)

        assert torch.all(
            batch_thgpu["random_background_color_%s" % dataset] == 
            batch_thgpu["rays_%s" % dataset][:, 10:13]
        )
        gt_hdr = torch.where(
            batch_thgpu["valid_mask_%s" % dataset][:, None].repeat(1, 3),
            batch_thgpu["hdrs_%s" % dataset],
            batch_thgpu["random_background_color_%s" % dataset],
        )
       
        lossMain = wl["lossMain"] * self.criterion_highlight(results["hdr_fine"], gt_hdr).nanmean()

        batch_thgpu["lossMain"] = lossMain
        batch_thgpu["loss"] = lossMain  # + lossMainHighlight + lossSuppressHdr3

        # stat
        gt_rgb = torch.clamp(tonemap_srgb_to_rgb(gt_hdr), min=0, max=1)
        for coarseOrFine in ["fine"]:
            batch_thgpu["statPsnr%s" % bt(coarseOrFine)] = psnr(
                torch.clamp(tonemap_srgb_to_rgb(results[("hdr_%s" % coarseOrFine)]), min=0, max=1),
                gt_rgb
            )
 
            batch_thgpu["statNumSampledPoints%s" % bt(coarseOrFine)] = (
                results["statNumSampledPoints_%s" % coarseOrFine][0]
            )
            batch_thgpu["statP2Rratio%s" % bt(coarseOrFine)] = (
                results["statP2Rratio_%s" % coarseOrFine][0]
            )

        return batch_thgpu

    @staticmethod
    def forwardRendering(rays, **kwargs):
        models = kwargs["models"]
        meta = models["meta"]
        config = kwargs["config"]
        iterCount = kwargs["iterCount"]
        cpu_ok = kwargs["cpu_ok"]
        force_all_rays = kwargs["force_all_rays"]
        requires_grad = kwargs["requires_grad"]
        if_query_the_expected_depth = kwargs["if_query_the_expected_depth"]
        lgtInput = kwargs["lgtInput"]
        callFlag = kwargs["callFlag"]
        logDir = kwargs["logDir"]
        hyperOutputs = kwargs.get("hyperOutputs", None)
        lgtMode = kwargs.get("lgtMode", "pointLightMode")
        envmap0 = kwargs["envmap0_thgpu"]
        envmapNormalizingFactor = kwargs["envmapNormalizingFactor"]
        if_only_predict_hdr2 = kwargs["if_only_predict_hdr2"]
        batch_head_ray_index = kwargs.get("batch_head_ray_index", None)
        envmapFiltered = kwargs.get("envmapFiltered", None)
        hyperPredictionDict = kwargs.get("hyperPredictionDict", models["model17"].hyper(lgtInput["envmapAlphaed"]))
        B = rays.shape[0]
        results = defaultdict(list)
        # batch_thgpu = kwargs["batch_thgpu"]  # debug purpose - for inspecting - do not use for computation

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        bg_color = rays[:, 10:13]

        # We shall be disable any use of background color, as in here, density / geometry is no longer updated
        assert torch.all(bg_color == 0)

        outputs = models["renderer"].render(
            rays_o, rays_d, models=models,
            iterCount=iterCount,
            lgtInputRays=lgtInput,
            lgtMode=lgtMode,
            envmap0=envmap0,
            envmapNormalizingFactor=envmapNormalizingFactor,
            envmapFiltered=envmapFiltered,
            hyperPredictionDict=hyperPredictionDict,
            batch_head_ray_index=batch_head_ray_index,
            recursive_depth=0,
            # some specs that shall be different between out-of-scene cameras and in-scene cameras
            min_near=models["renderer"].min_near,
            if_do_reflection_surface_cutoff=False,
            if_only_predict_hdr2=if_only_predict_hdr2,
            callFlag=callFlag,
        )

        # for k in ["rgb", "depth", "opacity", "lossTieRays", "lossBackfaceRays", "statNumSampledPoints"]:
        for k in outputs.keys():
            if k in outputs.keys():
                v = outputs[k]
                if cpu_ok:
                    v = v.detach().cpu()
                results[k + "_fine"] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    @classmethod
    def doPred0(cls, bsv0, **kwargs):
        datasetObj = kwargs["datasetObj"]
        cudaDevice = kwargs["cudaDevice"]
        config = kwargs["config"]
        iterCount = kwargs["iterCount"]
        callFlag = kwargs["callFlag"]
        logDir = kwargs["logDir"]
        models = kwargs["models"]
        ifRequiresMesh = kwargs["ifRequiresMesh"]
        orthogonalDensityResolution = kwargs["orthogonalDensityResolution"]
        marching_cube_thre = kwargs["marching_cube_thre"]
        hyperPredictionDict = kwargs.get("hyperPredictionDict", None)

        dataset = datasetObj.datasetConf["dataset"]
        assert dataset == bsv0["dataset"]
        caseID = datasetObj.datasetConf["caseID"]

        cls.assertModelEvalMode(models)
        rays_thgpu = torch.from_numpy(bsv0["rays"]).float().to(cudaDevice)

        with torch.no_grad():
            # chunk = 683 * 128
            if rays_thgpu.shape[0] % (683 * 64) == 0:  # 2048 * 1366 speed test
                chunk = (683 * 64)  # * 4  # times 4 for a100
                # We do not do explicit runtime benchmarking on resolution other than 800x800
            elif rays_thgpu.shape[0] % (40000) == 0:  # 800 * 800 speed test
                name = torch.cuda.get_device_name(cudaDevice)
                if ("3090" in name) and ("NVIDIA" in name):  # nvidia 3090
                    if ("cap" in dataset.lower()) and (caseID == 2):
                        chunk = 32000
                    elif ("cap" in dataset.lower()) and (caseID == 3):
                        chunk = 40000
                    else:
                        chunk = 20000  # small enough to avoid out-of-memory, but if you wish faster performance, change it larger
                elif ("A100" in name) and ("NVIDIA" in name):  # nvidia a100 (80g memory)
                    chunk = 40000 * 4
                else:
                    raise NotImplementedError("You need to set the chunk to be the largest value that no test sample leads to out-of-memory, and it is divisible by 800*800")
            else:
                raise NotImplementedError("Not yet set for the case for rays_thgpu.shape[0] == %d" % rays_thgpu.shape[0])
            results_list = []

            # envmap pre-filtering
            envmap0_thgpu = (
                torch.from_numpy(bsv0["envmapOriginalHighres"]).float().to(rays_thgpu.device)
                * 16 * 32 / bsv0["envmapNormalizingFactor"]
            )
            assert envmap0_thgpu.shape[0] > 17  # not to do 16x32
            assert envmap0_thgpu.shape[1] > 33  # not to do 16x32
            assert len(envmap0_thgpu.shape) == 3 and envmap0_thgpu.shape[2] == 3
            # 1. Smooth the envmap to each level
            envmapFiltered = []
            renderer = models["renderer"]
            for l in range(renderer.gauL):
                print("Filtering envmap at level %d / total %d" % (l, renderer.gauL))
                gauKSize = int(renderer.gauKSizeList[l])
                gauKSigma = float(renderer.gauKSigmaList[l])
                if gauKSize == 0:
                    envmapFiltered.append(envmap0_thgpu.permute(2, 0, 1))
                else:
                    tmp = gaussian_blur(
                        envmap0_thgpu.permute(2, 0, 1),
                        gauKSize,
                        gauKSigma,
                    )
                    envmapFiltered.append(tmp)
            envmapFiltered = torch.stack(envmapFiltered, 0)  # # (6(L), 3(RGB), 512(H), 1024(W))

            lgtInput = {
                "envmapAlphaed": torch.from_numpy(bsv0["envmapAlphaed"][None, :, :]).float().to(cudaDevice),
                "insideBatchInd_lgt": torch.zeros(chunk, dtype=torch.int64, device=cudaDevice),
            }
            if hyperPredictionDict is None:
                # t0 = time.time()
                hyperPredictionDict = models["model17"].hyper(lgtInput["envmapAlphaed"])
                # print("------------------------------" + str(time.time() - t0))
                # Note the time spent on hypernet forwarding is neglegible, shorter than one millisecond.
            else:
                pass  # hyperPredictionDict already exists
            t0 = time.time()
            time_record = []
            for i in range(0, rays_thgpu.shape[0], chunk):
                # print((i, rays_thgpu.shape[0], i / rays_thgpu.shape[0]))
                tail = min(i + chunk, rays_thgpu.shape[0])
                # lgtInput = {
                #     "envmapAlphaed": torch.from_numpy(bsv0["envmapAlphaed"][None, :, :]).float().to(cudaDevice),
                #     "insideBatchInd_lgt": torch.zeros(tail - i, dtype=torch.int64, device=cudaDevice),
                # }
            
                results_tmp = cls.forwardRendering(
                    rays_thgpu[i : tail],
                    models=models,
                    config=config,
                    iterCount=iterCount,
                    cpu_ok=True,
                    force_all_rays=False,
                    requires_grad=False,
                    if_query_the_expected_depth=False,
                    lgtInput=lgtInput,
                    callFlag=callFlag,
                    logDir=logDir,
                    lgtMode="pointLightMode",
                    # envmap0_thgpu=torch.from_numpy(
                    #     bsv0["envmapOriginalNormalized"]
                    # ).float().to(rays_thgpu.device) * bsv0["envmapNormalizingFactor"] / 16 / 32,  # high-freq branch needs the very original envmap
                    # envmap0_thgpu=torch.from_numpy(bsv0["envmapOriginalHighres"]).float().to(rays_thgpu.device),  # this produces exactly the same value (if it is 1k-resolution) as in the above commented lines (except for numerical very small differences)
                    envmap0_thgpu=envmap0_thgpu,
                    envmapNormalizingFactor=bsv0["envmapNormalizingFactor"],
                    envmapFiltered=envmapFiltered,
                    hyperPredictionDict=hyperPredictionDict,
                    if_only_predict_hdr2=False,
                    batch_head_ray_index=i,
                )
                results_list.append(results_tmp)
                time_record.append(time.time() - t0)
            time_record = np.array(time_record, dtype=np.float32)
            td = time_record[1:] - time_record[:-1]
            results = {k: torch.cat([x[k] for x in results_list], 0) for k in results_list[0].keys()}
        for k in results.keys():
            assert "float32" in str(results[k].dtype), k
        # bsv0["rgbFinePred"] = results["rgb_fine"].detach().cpu().numpy()
        winWidth = int(bsv0["winWidth"])
        winHeight = int(bsv0["winHeight"])
        if "hdr_fine" in results.keys():
            hdrFinePred = results["hdr_fine"].detach().cpu().numpy()
            bsv0["imghdrFinePred"] = hdrFinePred  # .reshape((winHeight, winWidth, 3))
            ldrFinePred = torch.clamp(tonemap_srgb_to_rgb(results["hdr_fine"].detach()), min=0, max=1).cpu().numpy()
            bsv0["imgFinePred"] = ldrFinePred  # .reshape((winHeight, winWidth, 3))
            bsv0["ldrFinePred"] = ldrFinePred  # benchmarking needs this one
        if "rawhdr2_fine" in results.keys():
            bsv0["imgrawhdr2FinePred"] = results["rawhdr2_fine"].detach().cpu().numpy()
        if "hdr2_fine" in results.keys():
            bsv0["imghdr2FinePred"] = results["hdr2_fine"].detach().cpu().numpy()  # .reshape((winHeight, winWidth, 3))
        if "hdr3_fine" in results.keys():
            bsv0["imghdr3FinePred"] = results["hdr3_fine"].detach().cpu().numpy()  # .reshape((winHeight, winWidth, 3))
        else:
            bsv0["imghdr3FinePred"] = np.zeros_like(bsv0["imghdr2FinePred"])

        if "depth_fine" in results.keys():
            bsv0["depthFinePred"] = (
                results["depth_fine"].detach().cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
            )
        if "opacity_fine" in results.keys():
            bsv0["opacityFinePred"] = (
                results["opacity_fine"].detach().cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
            )
        if "normal2_fine" in results.keys():
            bsv0["normal2"] = results["normal2_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "normal3_fine" in results.keys():
            bsv0["normal3"] = results["normal3_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "normal2DotH_fine" in results.keys():
            bsv0["normal2DotH"] = results["normal2DotH_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "normal3DotH_fine" in results.keys():
            bsv0["normal3DotH"] = results["normal3DotH_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsPointlightOpacities_fine" in results.keys():
            bsv0["hintsPointlightOpacities"] = results["hintsPointlightOpacities_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        for i in range(4):
            if ("hintsPointlightGGX%d_fine" % i) in results.keys():
                bsv0["hintsPointlightGGX%d" % i] = results["hintsPointlightGGX%d_fine" % i].detach(
                    ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefOpacities_fine" in results.keys():
            bsv0["hintsRefOpacities"] = results["hintsRefOpacities_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefSelf_fine" in results.keys():
            bsv0["hintsRefSelf"] = results["hintsRefSelf_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "hintsRefLevels_fine" in results.keys():
            bsv0["hintsRefLevels"] = results["hintsRefLevels_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, -1)).astype(np.float32)
        if "hintsRefLevelsColor_fine" in results.keys():
            bsv0["hintsRefLevelsColor"] = results["hintsRefLevelsColor_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # reshape((winHeight, winWidth, -1, 3)).astype(np.float32)
        if "hintsRefEnv_fine" in results.keys():
            bsv0["hintsRefEnv"] = results["hintsRefEnv_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefDistribute_fine" in results.keys():
            bsv0["hintsRefDistribute"] = results["hintsRefDistribute_fine"].detach(
                ).cpu().numpy().astype(np.float32)  # .reshape((winHeight, winWidth, -1)).astype(np.float32)
        if "hintsRefColor_fine" in results.keys():
            bsv0["hintsRefColor"] = results["hintsRefColor_fine"].detach(
                ).cpu().numpy().astype(np.float32)
        bsv0["timeElapsed"] = float(td.sum())
        bsv0["timeEvalMachine"] = socket.gethostname()

        if ifRequiresMesh:
            # raise ValueError("We disable this branch for now.")
            if orthogonalDensityResolution > 0:
                tmp = cls.forwardOrthogonalDensity(
                    models,
                    Lx=orthogonalDensityResolution,
                    Ly=orthogonalDensityResolution,
                    Lz=orthogonalDensityResolution,
                    cudaDevice=cudaDevice,
                    chunk=config.chunk,
                    marching_cube_thre=marching_cube_thre,
                    minBound=datasetObj.datasetConf["minBound"],
                    maxBound=datasetObj.datasetConf["maxBound"],
                )
                for meshName in ["fine"]:
                    for k in ["VertWorld", "Face"]:
                        bsv0[meshName + k] = tmp[meshName + k]

        return bsv0

    def _netFinetuningInitialization(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netFinetuningInitialization")

        finetuningPDSRI = self.config.finetuningPDSRI
        if finetuningPDSRI is not None:
            for kNow, kPre in [("model17", "modelFast"), ("occGrid", "occGrid")]:
                fn = (
                    self.projRoot
                    + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth"
                    % (
                        finetuningPDSRI["P"],
                        finetuningPDSRI["D"],
                        finetuningPDSRI["S"],
                        finetuningPDSRI["R"],
                        kPre,
                        finetuningPDSRI["I"],
                    )
                )
                assert os.path.isfile(fn), fn
                with open(fn, "rb") as f:
                    print("Finetuning from %s" % fn)
                    loaded_state_dict = torch.load(f, map_location=self.cudaDeviceForAll)
                keys = list(loaded_state_dict.keys())
                keys_to_del = [k for k in keys if k in [
                    # "normal3_alpha.params", "normal3_hdr.params", "color_net.params"
                    "color_net.params"
                ]]
 
                for _ in keys_to_del:
                    del loaded_state_dict[_]
                self.models[kNow].load_state_dict(  # k == "model"
                    loaded_state_dict,
                    strict=False,
                )

    def _optimizerSetups(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _optimizerSetups")

        # unless the two optimizers are different, you should write in this form
        modelKeys = [
            k
            for k in self.models.keys()
            if k != "meta" and k not in self.models["meta"]["nonLearningModelNameList"]
        ]
        params = list(self.models[modelKeys[0]].parameters())
        for j in range(1, len(modelKeys)):
            params += list(self.models[modelKeys[j]].parameters())
        self.optimizerModels = {}
        self.optimizerModels["all"] = optim.Adam(
            params,
            lr=config.adam_lr,
            betas=config.adam_betas,
            eps=config.adam_epsilon,
        )

        # gradScaler
        self.gradScalerModels = {}
        self.gradScalerModels["all"] = GradScaler(
            enabled=True
        )  # when you use tcnn with half precision, it is important to use grad_scaler

        # lr_scheduler
        self.schedulerModels = {}
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iterCount: config.scheduler_base ** min(iterCount / 30000, 1)
        )
        self.schedulerModels["all"] = scheduler(self.optimizerModels["all"])

        # ema
        self.emaModels = {}
        self.emaModels["all"] = ExponentialMovingAverage(self.models["model17"].parameters(), decay=0.95)

    @classmethod
    def forwardOrthogonalDensity(cls, models, **kwargs):
        cls.assertModelEvalMode(models)
        chunk = kwargs["chunk"]
        minBound = kwargs["minBound"]
        maxBound = kwargs["maxBound"]
        Lx = kwargs["Lx"]
        Ly = kwargs["Ly"]
        Lz = kwargs["Lz"]
        cudaDevice = kwargs["cudaDevice"]
        marching_cube_thre = 10  # this should the property of this approach.
        epsilon = 1.0e-5
        sCell = (maxBound - minBound) / np.array([Lx, Ly, Lz], dtype=np.float32)
        goxyz = minBound + 0.5 * sCell
        xi = np.linspace(goxyz[0], goxyz[0] + (Lx - 1) * sCell[0], Lx).astype(
            np.float32
        )
        yi = np.linspace(goxyz[1], goxyz[1] + (Ly - 1) * sCell[1], Ly).astype(
            np.float32
        )
        zi = np.linspace(goxyz[2], goxyz[2] + (Lz - 1) * sCell[2], Lz).astype(
            np.float32
        )
        x, y, z = np.meshgrid(xi, yi, zi)  # YXZ volume
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        xyz = np.stack([x, y, z], 1)
        xyz_thgpu = torch.from_numpy(xyz).to(cudaDevice)
        dir_chunk_thgpu = torch.zeros(chunk, 3, dtype=torch.float32, device=cudaDevice)
        batchTot = int(math.ceil(float(xyz.shape[0]) / float(chunk)))
        outFine = np.zeros((xyz.shape[0],), dtype=np.float32)
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=True):
            for b in range(batchTot):
                head = b * chunk
                tail = min((b + 1) * chunk, xyz.shape[0])

                xyz_chunk_thgpu = xyz_thgpu[head:tail]
                # t = models["model"].density(
                t = models["model17"].density(
                    xyz_chunk_thgpu, if_do_bound_clamp=True, if_output_only_sigma=True,
                    if_compute_normal3=False)
                outFine[head:tail] = t.detach().cpu().numpy()
        outFine = outFine.reshape((Ly, Lx, Lz))
        # marching cube
        if outFine.max() < marching_cube_thre:
            outFine[0, 0, 0] = marching_cube_thre + epsilon
        elif outFine.min() > marching_cube_thre:
            outFine[-1, -1, -1] = marching_cube_thre - epsilon
        fineVert0, fineFace0 = voxSdfSign2mesh_skmc(
            outFine, goxyz, sCell, level=marching_cube_thre
        )
        return {
            "fineVertWorld": fineVert0,
            "fineFace": fineFace0,
        }


def returnExportedClasses(
    wishedClassNameList,
):

    exportedClasses = {}

    if wishedClassNameList is None or "DTrainer" in wishedClassNameList:
        exportedClasses["DTrainer"] = DTrainer

    return exportedClasses
