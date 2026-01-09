import torch
import os
import copy
from collections import OrderedDict
from ..trainer import Trainer
import pickle
from ..models.model import PRTNetwork1
from ..models.hash_encoding import HashEmbedder, SHEncoder
import numpy as np
from codes_py.toolbox_3D.mesh_io_v1 import load_ply_np, load_obj_np
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.rotations_v1 import ELU2cam
from Bprelight4.testDataEntry.testDataEntryPool import getTestDataEntryDict
from codes_py.toolbox_framework.framework_util_v4 import (
    bsv02bsv,
    bsv2bsv0,
    castAnything,
    checkPDSRLogDirNew,
    constructInitialBatchStepVis0,
    load_network,
    mergeFromAnotherBsv0,
    probe_load_network,
    save_network,
)

from Bprelight4.testDataEntry.renderingNerfBlender.renderingNerfBlenderDatasetNew2 import (
    RenderingNerfBlenderDataset,
)
from Bprelight4.testDataEntry.renderingNerfBlender.renderingLightStageOriginal3 import (
    RenderingLightStageOriginal3,
)

class DTrainer(Trainer):
    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        models = {}

        # load the fixed surface point clouds
        assert len(config.datasetConfDict) == 1
        datasetConf = list(config.datasetConfDict.values())[0]
        caseID = int(datasetConf["caseID"])
        if datasetConf["dataset"].startswith("renderingNerfBlender"):
            fn = self.projRoot + "v_external_codes/NeuS/exp/expp500R%d/wmask/expp500R%d_00300000_resolution_512.obj" % (
                caseID, caseID
            )
            assert os.path.isfile(fn), fn
            points, _ = load_obj_np(fn)
        elif datasetConf["dataset"].startswith("renderingCapture7jf") or datasetConf["dataset"].startswith("capture7jf"):
            fn = self.projRoot + "v_external_codes/NeuS/exp/capture7jfR%d/wmask/capture7jfR%d_00300000_resolution_512.obj" % (
                caseID, caseID
            )
            assert os.path.isfile(fn), fn
            points, _ = load_obj_np(fn)
        else:
            raise NotImplementedError("Unknown dataset: %s" % datasetConf["dataset"])
        tight_min_bound = points.min(0)
        tight_max_bound = points.max(0)

        hashEmbed = HashEmbedder(
            bounding_box=[
                torch.from_numpy(tight_min_bound).float().to(self.cudaDeviceForAll),
                torch.from_numpy(tight_max_bound).float().to(self.cudaDeviceForAll),
            ],
            n_features_per_level=config.hash_n_features_per_level,
            base_resolution=config.hash_base_resolution,
            n_levels=config.hash_n_levels,
            finest_resolution=config.hash_finest_resolution,
            sparse=config.hash_sparse,
            vertices=torch.from_numpy(points).float().to(self.cudaDeviceForAll),
        ).to(self.cudaDeviceForAll)

        pts_ch = hashEmbed.out_dim
        models["hashEmbed"] = hashEmbed  # already put into cuda

        viewEmbed = SHEncoder().to(self.cudaDeviceForAll)
        view_ch = viewEmbed.out_dim
        models["viewEmbed"] = viewEmbed

        input_features = pts_ch + view_ch + view_ch
        renderer = PRTNetwork1(
            W=config.nrtf_W,
            D=config.nrtf_D,
            skips=config.nrtf_skips,
            din=input_features,
            dout=3,
            activation=config.nrtf_activation,
        ).to(self.cudaDeviceForAll)
        models["renderer"] = renderer
        meta = {
            "nonLearningModelNameList": ["viewEmbed"],
        }
        models["meta"] = meta
        self.models = models

    def metaDataLoading(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaDataLoading")

        datasetConfDict = config.datasetConfDict

        datasetObjDict = OrderedDict([])

        for datasetConf in datasetConfDict.values():
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                datasetObj = RenderingNerfBlenderDataset(datasetConf, if_need_metaDataLoading=True)
                datasetObjDict[datasetConf["dataset"]] = datasetObj
            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                datasetObj = RenderingLightStageOriginal3(datasetConf)
                datasetObjDict[datasetConf["dataset"]] = datasetObj
            else:
                raise NotImplementedError(
                    "Unknown dataset class: %s" % datasetConf["class"]
                )

        self.datasetObjDict = datasetObjDict

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
        if datasetConf_testingOnTraining["class"] == "RenderingNerfBlenderDataset":
            self.testingOnTrainingDatasetObj = RenderingNerfBlenderDataset(
                datasetConf_testingOnTraining, if_need_metaDataLoading=False
            )
        elif datasetConf_testingOnTraining["class"] == "RenderingLightStageOriginal3":
            self.testingOnTrainingDatasetObj = RenderingLightStageOriginal3(
                datasetConf_testingOnTraining, if_need_metaDataLoading=False
            )

        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf_testingOnTraining["class"])

        # Val Vis
        self.htmlStepperVal = HTMLStepper(
            self.monitorValLogDir, 40, "monitorVal"
        )
        datasetConf_testing = copy.deepcopy(list(self.config.datasetConfDict.values())[0])
        datasetConf_testing["singleImageMode"] = True
        if datasetConf_testing["class"] == "RenderingNerfBlenderDataset":
            testDataNickName = "%sCase%dSplit%s" % (
                list(self.config.datasetConfDict.values())[0]["dataset"],
                list(self.config.datasetConfDict.values())[0]["caseID"],
                "Val",
            )
        elif datasetConf_testing["class"] == "RenderingLightStageOriginal3":
            initialBucket = datasetConf_testing["bucket"]
            if initialBucket.startswith("ol3t"):
                datasetConf_testing["dataset"] = "capture7jfFree3"
                datasetConf_testing["bucket"] = "ol3tFree3"
            elif initialBucket.startswith("envmap"):
                datasetConf_testing["dataset"] = "capture7jf"
                datasetConf_testing["bucket"] = initialBucket  # Currently do not change
            else:
                raise NotImplementedError("Unknown bucket: %s" % initialBucket)
            if datasetConf_testing["bucket"].startswith("envmap"):
                testDataNickName = "capture7jf%sCase%dSplit%s" % (
                    bt(datasetConf_testing["bucket"]), datasetConf_testing["caseID"], "Val"  # This bucket does not have Test
                )
            elif datasetConf_testing["bucket"].startswith("ol3t"):
                testDataNickName = "capture7jfCase%dSplit%s" % (datasetConf_testing["caseID"], "Test")
            else:
                raise NotImplementedError("Unknown bucket: %s" % datasetConf_testing["bucket"])
        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf_testing["class"])
        
        self.testDataEntry = getTestDataEntryDict(
            wishedTestDataNickName=[testDataNickName],
            fyiDatasetConf=datasetConf_testing,
            cudaDevice=self.cudaDeviceForAll,
        )[testDataNickName]

    def stepBatch(self, **kwargs):
        config = self.config
        tmp = OrderedDict([])
        for dataset in config.datasetConfDict.keys():
            datasetConf = config.datasetConfDict[dataset]
            datasetObj = self.datasetObjDict[dataset]
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                ind_fg = np.random.randint(
                    0,
                    datasetObj.fg_count,
                    datasetConf["batchSizeFgPerProcess"],
                    dtype=np.int64,
                )
                ind_bg = np.random.randint(
                    0,
                    datasetObj.bg_count,
                    datasetConf["batchSizeBgPerProcess"],
                    dtype=np.int64,
                )
                valid_mask =np.concatenate([
                    datasetObj.fg_masks[ind_fg],
                    np.zeros((ind_bg.shape[0],), dtype=bool),
                ], 0)

                assert not config.white_back
                random_background_color = np.zeros((
                    datasetConf["batchSizeFgPerProcess"] + datasetConf["batchSizeBgPerProcess"], 3
                ), dtype=np.float32)
                # This is different from Polattngp4Berkeley-D2hint14fgbg

                batchDid_thcpu = {
                    "rgbs": torch.from_numpy(
                        np.concatenate([
                            datasetObj.fg_rgbs[ind_fg].astype(np.float32) / 255.0,
                            np.zeros((ind_bg.shape[0], 3), dtype=np.float32),
                        ], 0)
                    ),
                    "valid_mask": torch.from_numpy(valid_mask),
                    "random_background_color": torch.from_numpy(random_background_color),
                    "indexID": torch.from_numpy(
                        np.concatenate([
                            datasetObj.fg_indexID[ind_fg],
                            datasetObj.bg_indexID[ind_bg],
                        ], 0)
                    ),
                    "pixelID": torch.from_numpy(
                        np.concatenate([
                            datasetObj.fg_pixelID[ind_fg],
                            datasetObj.bg_pixelID[ind_bg],
                        ], 0)
                    ),
                    "hdrs": torch.from_numpy(
                        np.concatenate([
                            datasetObj.fg_hdrs[ind_fg],
                            np.zeros((ind_bg.shape[0], 3), dtype=np.float32),
                        ], 0)
                    ),
                }
                if datasetObj.ifLoadDepth:
                    batchDid_thcpu["depths"] = torch.from_numpy(
                        np.concatenate([
                            datasetObj.fg_scaledDepths[ind_fg],
                            np.zeros((ind_bg.shape[0],), dtype=np.float32),
                        ], 0)
                    )
                if datasetObj.ifLoadNormal:
                    raise NotImplementedError("Net yet implemented for this situation")
                    batchDid_thcpu["normals"] = torch.from_numpy(datasetObj.all_normals[ind])
            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                rays_fg_count = datasetObj.all_fg_indexID.shape[0]
                ind_fg = np.random.randint(
                    0,
                    rays_fg_count,
                    datasetConf["batchSizeFgPerProcess"],
                    dtype=np.int64,
                )
                
                assert datasetConf["white_back"] == False  # real data always have the bg as zero
                batchDid_fg_np = {
                    "hdrs": datasetObj.all_fg_hdrs[ind_fg],
                    "valid_mask": np.ones((ind_fg.shape[0],), dtype=bool),
                    "indexID": datasetObj.all_fg_indexID[ind_fg],
                    "pixelID": datasetObj.all_fg_pixelID[ind_fg],
                    "random_background_color": np.zeros((ind_fg.shape[0], 3), dtype=np.float32)
                }
                rays_mg_count = datasetObj.all_mg_groupViewID.shape[0] * datasetObj.nOlat  # correct
                ind_mg = np.random.randint(
                    0,
                    rays_mg_count,
                    datasetConf["batchSizeMgPerProcess"],
                    dtype=np.int64,
                )
                batchDid_mg_np = {  
                    # mg is just an emphasized smapling portion of bg
                    # All its property (like below) would be exactly following bg
                    # just that it will be more likely to be sampled compared to the pixel locations in bg
                    "hdrs": np.zeros((ind_mg.shape[0], 3), dtype=np.float32),
                    "valid_mask": np.zeros((ind_mg.shape[0],), dtype=bool),
                    "indexID": datasetObj.all_mg_groupViewID[ind_mg // datasetObj.nOlat] * datasetObj.nOlat + \
                        (ind_mg % datasetObj.nOlat),
                    "pixelID": datasetObj.all_mg_pixelID[ind_mg // datasetObj.nOlat],
                    "random_background_color": np.zeros((ind_mg.shape[0], 3)).astype(np.float32),  # debug not random now
                }
                rays_bg_count = datasetObj.all_bg_groupViewID.shape[0] * datasetObj.nOlat # correct
                ind_bg = np.random.randint(
                    0,
                    rays_bg_count,
                    datasetConf["batchSizeBgPerProcess"],
                    dtype=np.int64,
                )
                batchDid_bg_np = {
                    "hdrs": np.zeros((ind_bg.shape[0], 3), dtype=np.float32),
                    "valid_mask": np.zeros((ind_bg.shape[0],), dtype=bool),
                    "indexID": datasetObj.all_bg_groupViewID[ind_bg // datasetObj.nOlat] * datasetObj.nOlat + \
                        (ind_bg % datasetObj.nOlat),
                    "pixelID": datasetObj.all_bg_pixelID[ind_bg // datasetObj.nOlat],
                    "random_background_color": np.zeros((ind_bg.shape[0], 3)).astype(np.float32),  # debug not random now
                }
                batchDid_np = {
                    k: np.concatenate([batchDid_fg_np[k], batchDid_mg_np[k], batchDid_bg_np[k]], 0)
                    for k in batchDid_fg_np.keys()}
                if datasetConf.get("ifLoadDepth", False):
                    batchDid_np["depths"] = np.concatenate([
                        datasetObj.all_fg_zbuffer[ind_fg],
                        np.zeros((ind_mg.shape[0],), dtype=np.float32),
                        np.zeros((ind_bg.shape[0],), dtype=np.float32),
                    ], 0)
                batchDid_thcpu = castAnything(batchDid_np, "np2thcpu")
            else:
                raise NotImplementedError(
                    "Unknown dataset class: %s" % datasetConf["class"]
                )
            tmp[dataset] = batchDid_thcpu

        batch_thcpu = {}
        for dataset in tmp.keys():
            for k in tmp[dataset].keys():
                batch_thcpu[k + "_" + dataset] = tmp[dataset][k]

        batch_thgpu = castAnything(
            batch_thcpu, "thcpu2thgpu", device=self.cudaDeviceForAll
        )

        return batch_thcpu, batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        assert len(self.datasetObjDict.keys()) == 1
        datasetObj = list(self.datasetObjDict.values())[0]
        datasetConf = datasetObj.datasetConf
        dataset = datasetObj.datasetConf["dataset"]

        # rays
        if datasetConf["class"] == "RenderingNerfBlenderDataset":
            sceneShiftFirst = datasetConf["sceneShiftFirst"]
            assert sceneShiftFirst.shape == (3,) and sceneShiftFirst.dtype == np.float32
            sceneScaleSecond = float(datasetConf["sceneScaleSecond"])
            assert sceneScaleSecond > 0
            # inside_cube_coordinates = (original_coordinates - shiftFirst) * scaleSecond
            if datasetConf["getRaysMeans"] == "ELU":
                E = (datasetObj.A0["E"][batch_thcpu["indexID_%s" % dataset].numpy(), :] - sceneShiftFirst[None, :]) * sceneScaleSecond
                L = (datasetObj.A0["L"][batch_thcpu["indexID_%s" % dataset].numpy(), :] - sceneShiftFirst[None, :]) * sceneScaleSecond
                U = datasetObj.A0["U"][batch_thcpu["indexID_%s" % dataset].numpy(), :]
                caseSplitInsideIndexList = \
                    datasetObj.A0["caseSplitInsideIndexList"][batch_thcpu["indexID_%s" % dataset].numpy()]
                w2c = torch.FloatTensor(ELU2cam(np.concatenate([E, L, U], 1))).to(
                    self.cudaDeviceForAll
                )
                c2w = torch.linalg.inv(w2c)
            elif datasetConf["getRaysMeans"] == "camInv64":
                c2w = torch.from_numpy(datasetObj.A0["camInv64"][batch_thcpu["indexID_%s" % dataset].numpy(), :, :]).to(self.cudaDeviceForAll)
                c2w[:, :3, 3] = (c2w[:, :3, 3] - torch.from_numpy(datasetConf["sceneShiftFirst"][None, :]).to(c2w.device)) * datasetConf["sceneScaleSecond"]
            else:
                raise ValueError("Unknown getRaysMeans: %s" % datasetConf["getRaysMeans"])
            rays_o = c2w[:, :3, 3].float()
            directions = torch.DoubleTensor(
                datasetObj.directions_flattened[
                    batch_thcpu["pixelID_%s" % dataset].numpy(), :
                ]
            ).to(self.cudaDeviceForAll)
            rays_d = (c2w[:, :3, :3] * directions[:, None, :]).sum(2)
            rays_d = (
                rays_d / torch.norm(rays_d, p=2, dim=1)[:, None]
            ).float()  # This was missing in the past
            near = datasetObj.near * torch.ones_like(rays_o[:, :1])
            far = datasetObj.far * torch.ones_like(rays_o[:, :1])
            # Now 8:9 would be pixel_area
            pixel_area = torch.FloatTensor(
                datasetObj.pixel_area_flattened[
                    batch_thcpu["pixelID_%s" % dataset].numpy(),
                ]
            ).to(self.cudaDeviceForAll)[:, None]
            # Now 9:10 would be directions_norm (which is just zbuffer_to_euclidean ratio)
            zbuffer_to_euclidean_ratio = torch.FloatTensor(
                datasetObj.zbuffer_to_euclidean_ratio[
                    batch_thcpu["pixelID_%s" % dataset].numpy(),
                ]
            ).to(self.cudaDeviceForAll)[:, None]
            # Now 11:13 wouldbe random_background_color
            random_background_color = batch_thcpu["random_background_color_%s" % dataset].to(
                self.cudaDeviceForAll
            )
            # put together
            batch_thgpu["rays_%s" % dataset] = torch.cat(
                [rays_o, rays_d, near, far, pixel_area, zbuffer_to_euclidean_ratio, random_background_color], 1
            ).contiguous()
        elif datasetConf["class"] == "RenderingLightStageOriginal3":
            indexID = batch_thcpu["indexID_%s" % dataset].numpy()  # 0~(180*nOlat-1)
            groupViewID = indexID // datasetObj.nOlat
            focalLengthWidth = datasetObj.focalLengthWidth[groupViewID]
            focalLengthHeight = datasetObj.focalLengthHeight[groupViewID]
            c2w = datasetObj.c2w[groupViewID]

            sceneShiftFirst = datasetConf["sceneShiftFirst"]
            sceneScaleSecond = datasetConf["sceneScaleSecond"]

            pixelID = batch_thcpu["pixelID_%s" % dataset].numpy()
            rowID = pixelID // (datasetObj.winWidth + 2 * datasetConf["rays_background_pad_width"])  # correct
            colID = pixelID % (datasetObj.winWidth + 2 * datasetConf["rays_background_pad_width"])  # correct

            directions_unnormalized_cam = np.stack([
                (colID.astype(np.float32) + 0.5 - datasetObj.winWidth / 2.0 - datasetConf["rays_background_pad_width"]) / focalLengthWidth,  # correct
                (rowID.astype(np.float32) + 0.5 - datasetObj.winHeight / 2.0 - datasetConf["rays_background_pad_height"]) / focalLengthHeight,  # correct
                np.ones((indexID.shape[0],), dtype=np.float32),
            ], 1)
            directions_unnormalized = (c2w[:, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
            directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
            directions_normalized = directions_unnormalized / directions_norm[:, None]
            zbuffer_to_euclidean_ratio = directions_norm

            random_background_color = batch_thcpu["random_background_color_%s" % dataset].numpy()

            rays = np.concatenate([
                (c2w[:, :3, 3] - sceneShiftFirst[None, :]) * sceneScaleSecond,  # rays_o
                directions_normalized,  # rays_d
                # datasetConf["ray_near"] * np.ones((indexID.shape[0], 1), dtype=np.float32),
                # datasetConf["ray_far"] * np.ones((indexID.shape[0], 1), dtype=np.float32),
                np.nan * np.zeros((indexID.shape[0], 3), dtype=np.float32),
                zbuffer_to_euclidean_ratio[:, None],
                random_background_color,
            ], 1)
            batch_thgpu["rays_%s" % dataset] = torch.from_numpy(rays).float().to(self.cudaDeviceForAll)

        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf["class"])

        # lgt
        if datasetObj.datasetConf["lgtLoadingMode"] == "QSPL":
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                indexID = batch_thcpu["indexID_%s" % dataset].numpy()
                for k in ["lgtE", "objCentroid"]:
                    batch_thgpu["%s_%s" % (k, dataset)] = torch.from_numpy(
                        (datasetObj.A0[k][indexID] - datasetConf["sceneShiftFirst"]) * datasetConf["sceneScaleSecond"]
                    ).to(self.cudaDeviceForAll)
            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                indexID = batch_thcpu["indexID_%s" % dataset].numpy()
                groupViewID = indexID // datasetObj.nOlat
                ccID = indexID % datasetObj.nOlat
                batch_thgpu["lgtE_%s" % dataset] = torch.from_numpy(
                    (datasetObj.olat_lgtEs[groupViewID, ccID, :] - datasetConf["sceneShiftFirst"][None, :]) * datasetConf["sceneScaleSecond"]
                ).to(self.cudaDeviceForAll)
                bsz = int(datasetConf["batchSizeFgPerProcess"] + datasetConf["batchSizeMgPerProcess"] + datasetConf["batchSizeBgPerProcess"])
                batch_thgpu["objCentroid_%s" % dataset] = torch.from_numpy(
                    np.tile(
                        ((datasetObj.objCentroid0 - datasetConf["sceneShiftFirst"]) * datasetConf["sceneScaleSecond"])[None, :],
                        (bsz, 1),
                    )
                ).to(self.cudaDeviceForAll)
            else:
                raise NotImplementedError("Unknown dataset class: %s" % datasetConf["class"])
        elif datasetObj.datasetConf["lgtLoadingMode"] == "OMEGA":
            raise NotImplementedError
            lgtDatasetCache = datasetObj.meta["lgtDatasetCache"].cache
            assert len(lgtDatasetCache.keys()) == 1  # only one lighting dataset is assumed (the whole training set shares the same lighting dataset)
            A0_lgt = list(lgtDatasetCache.values())[0]["A0"]
            lgtID = datasetObj.A0["lgtIDList"][batch_thcpu["indexID_%s" % dataset].numpy()]
            omega = A0_lgt["omega"][lgtID]
            batch_thgpu["omegaInput_%s" % dataset] = torch.FloatTensor(omega).to(self.cudaDeviceForAll)
        elif datasetObj.datasetConf["lgtLoadingMode"] in ["ENVMAP"]:
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                raise NotImplementedError
                indexID = batch_thcpu["indexID_%s" % dataset].numpy()
                envmap = datasetObj.meta["lgtEnvmapPreload"][datasetObj.indTrainInverse[indexID]]
                batch_thgpu["envmap_%s" % dataset] = torch.FloatTensor(envmap).to(self.cudaDeviceForAll)
            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                lgtDataset = datasetConf["envmapDataset"]
                indexID = batch_thcpu["indexID_%s" % dataset].numpy()
                lgtID = datasetObj.all_fg_lgtID[indexID]
                envmaps = datasetObj.envmapLgtDatasetCache.cache[lgtDataset]["envmaps"]
                flagAlreadyRead = datasetObj.envmapLgtDatasetCache.cache[lgtDataset]["flagAlreadyRead"]
                assert np.all(flagAlreadyRead[lgtID])
                batch_thgpu["envmaps_%s" % dataset] = torch.from_numpy(envmaps[lgtID]).float().to(self.cudaDeviceForAll)
            else:
                raise NotImplementedError("Unknown class: %s" % datasetConf["class"])
        elif datasetObj.datasetConf["lgtLoadingMode"] in ["NONE"]:
            pass  # do nothing
        else:
            raise NotImplementedError("Unknown lgtLoadingMode: %s" % datasetObj.datasetConf["lgtLoadingMode"])

        if datasetConf.get("batchSizeHighlightPerProcess", 0) > 0:
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                sceneShiftFirst = datasetConf["sceneShiftFirst"]
                assert sceneShiftFirst.shape == (3,) and sceneShiftFirst.dtype == np.float32
                sceneScaleSecond = float(datasetConf["sceneScaleSecond"])
                assert sceneScaleSecond > 0
                for highOrSubLight in ["highlight", "sublight"]:
                    tot = int(datasetObj.highlight["indexID_%s" % highOrSubLight].shape[0])
                    ind = np.random.randint(
                        0,
                        tot,
                        datasetConf["batchSize%sPerProcess" % bt(highOrSubLight)],
                        dtype=np.int64,
                    )
                    indexID_highOrSubLight = datasetObj.highlight["indexID_%s" % highOrSubLight][ind]
                    pixelID_highOrSubLight = datasetObj.highlight["pixelID_%s" % highOrSubLight][ind]
                    hdr_highOrSubLight = datasetObj.highlight["hdr_%s" % highOrSubLight][ind, :]

                    assert datasetConf["getRaysMeans"] == "ELU"
                    E = (datasetObj.A0["E"][indexID_highOrSubLight, :] - sceneShiftFirst[None, :]) * sceneScaleSecond
                    L = (datasetObj.A0["L"][indexID_highOrSubLight, :] - sceneShiftFirst[None, :]) * sceneScaleSecond
                    U = datasetObj.A0["U"][indexID_highOrSubLight, :]
                    lgtE_highOrSubLight = (datasetObj.A0["lgtE"][indexID_highOrSubLight, :] - sceneShiftFirst[None, :]) * sceneScaleSecond
                    objCentroid_highOrSubLight = (datasetObj.A0["objCentroid"][indexID_highOrSubLight] - sceneShiftFirst[None, :]) * sceneScaleSecond
                    w2c = torch.FloatTensor(ELU2cam(np.concatenate([E, L, U], 1))).to(
                        self.cudaDeviceForAll
                    )
                    c2w = torch.linalg.inv(w2c)
                    rays_o = c2w[:, :3, 3].float()
                    directions = torch.DoubleTensor(
                        datasetObj.directions_flattened[pixelID_highOrSubLight, :]
                    ).to(self.cudaDeviceForAll)
                    rays_d = (c2w[:, :3, :3] * directions[:, None, :]).sum(2)
                    rays_d = (
                        rays_d / torch.norm(rays_d, p=2, dim=1)[:, None]
                    ).float()  # This was missing in the past

                    batch_thgpu["%sRays_%s" % (highOrSubLight, dataset)] = torch.cat([
                        rays_o,
                        rays_d,
                        torch.zeros(ind.shape[0], 4, dtype=torch.float32, device=rays_o.device) * np.nan,
                        torch.zeros(ind.shape[0], 3, dtype=torch.float32, device=rays_o.device),
                    ], 1)
                    batch_thgpu["%sRandomBackgroundColor_%s" % (highOrSubLight, dataset)] = torch.zeros(
                        ind.shape[0], 3, dtype=torch.float32, device=self.cudaDeviceForAll
                    )
                    batch_thgpu["%sLgtEs_%s" % (highOrSubLight, dataset)] = torch.from_numpy(
                        lgtE_highOrSubLight
                    ).float().to(self.cudaDeviceForAll)
                    batch_thgpu["%sHdrs_%s" % (highOrSubLight, dataset)] = torch.from_numpy(
                        hdr_highOrSubLight
                    ).float().to(self.cudaDeviceForAll)
                    batch_thgpu["%sObjCentroid_%s" % (highOrSubLight, dataset)] = torch.from_numpy(
                        objCentroid_highOrSubLight,
                    ).float().to(self.cudaDeviceForAll)

                    # validating
                    continue  # The below have been validated
                    indexPixelID_sampled = (
                        batch_thgpu["indexID_%s" % dataset].long() * 800 * 800 + batch_thgpu["pixelID_%s" % dataset].long()
                    ).detach().cpu().numpy()
                    indexPixelID_light = (
                        indexID_highOrSubLight.astype(np.int64) * 800 * 800 + pixelID_highOrSubLight.astype(np.int64)
                    )
                    indexPixelID, common_sampled, common_light = np.intersect1d(
                        indexPixelID_sampled, indexPixelID_light, return_indices=True
                    )

                    hdrs_sampled = batch_thgpu["hdrs_%s" % dataset][common_sampled].detach().cpu().numpy()
                    hdrs_light = hdr_highOrSubLight[common_light]
                    assert np.all(hdrs_sampled == hdrs_light)

                    rays_sampled = batch_thgpu["rays_%s" % dataset][common_sampled, :6]
                    rays_light = batch_thgpu["%sRays_%s" % (highOrSubLight, dataset)][common_light, :6]
                    assert torch.all(rays_sampled == rays_light)

                    lgtE_sampled = batch_thgpu["lgtE_%s" % dataset][common_sampled, :]
                    lgtE_light = batch_thgpu["%sLgtEs_%s" % (highOrSubLight, dataset)][common_light, :]
                    assert torch.all(lgtE_sampled == lgtE_light)
                    print("Validated!")

            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                _dataset = "_" + dataset

                highlight_tot = int(datasetObj.highlight_hdrs.shape[0])
                highlight_ind = np.random.randint(
                    0,
                    highlight_tot,
                    datasetConf["batchSizeHighlightPerProcess"],
                    dtype=np.int64,
                )
                batch_thgpu["highlightRays" + _dataset] = torch.from_numpy(
                    np.concatenate([
                        datasetObj.highlight_rays[highlight_ind],
                        np.zeros((highlight_ind.shape[0], 4), dtype=np.float32) * np.nan, # near/far/??/ratio 6:10
                        np.zeros((highlight_ind.shape[0], 3), dtype=np.float32),  # random_background 10:13
                    ], 1)
                ).float().to(self.cudaDeviceForAll)
                batch_thgpu["highlightRandomBackgroundColor" + _dataset] = torch.zeros(
                    highlight_ind.shape[0], 3, dtype=torch.float32, device=self.cudaDeviceForAll)
                batch_thgpu["highlightHdrs" + _dataset] = torch.from_numpy(
                    datasetObj.highlight_hdrs[highlight_ind]
                ).float().to(self.cudaDeviceForAll)
                batch_thgpu["highlightLgtEs" + _dataset] = torch.from_numpy(
                    datasetObj.highlight_lgtEs[highlight_ind]
                ).float().to(self.cudaDeviceForAll)
                # We still need the highlightObjCentroid to make the consistent bsz
                batch_thgpu["highlightObjCentroid" + _dataset] = torch.from_numpy(
                    np.tile((
                        (datasetObj.objCentroid0 - sceneShiftFirst) * sceneScaleSecond
                    )[None, :], (highlight_ind.shape[0], 1))
                ).float().to(self.cudaDeviceForAll)

                sublight_tot = int(datasetObj.sublight_hdrs.shape[0])
                sublight_ind = np.random.randint(
                    0,
                    sublight_tot,
                    datasetConf["batchSizeSublightPerProcess"],
                    dtype=np.int64,
                )
                batch_thgpu["sublightRays" + _dataset] = torch.from_numpy(
                    np.concatenate([
                        datasetObj.sublight_rays[sublight_ind],
                        np.zeros((sublight_ind.shape[0], 4), dtype=np.float32) * np.nan,
                        np.zeros((sublight_ind.shape[0], 3), dtype=np.float32),
                    ], 1)
                ).float().to(self.cudaDeviceForAll)
                batch_thgpu["subllightRandomBackgroundColor" + _dataset] = torch.zeros(
                    sublight_ind.shape[0], 3, dtype=torch.float32, device=self.cudaDeviceForAll
                )
                batch_thgpu["sublightHdrs" + _dataset] = torch.from_numpy(
                    datasetObj.sublight_hdrs[sublight_ind],
                ).float().to(self.cudaDeviceForAll)
                batch_thgpu["sublightLgtEs" + _dataset] = torch.from_numpy(
                    datasetObj.sublight_lgtEs[sublight_ind],
                ).float().to(self.cudaDeviceForAll)
                batch_thgpu["sublightObjCentroid" + _dataset] = torch.from_numpy(
                    np.tile(
                        ((datasetObj.objCentroid0 - sceneShiftFirst) * sceneScaleSecond)[None, :],
                        (sublight_ind.shape[0], 1),
                    )
                ).float().to(self.cudaDeviceForAll)

            else:
                raise NotImplementedError("Unknown class: %s" % datasetConf["class"])

        return batch_thgpu
        

def returnExportedClasses(
    wishedClassNameList,
):

    exportedClasses = {}

    if wishedClassNameList is None or "DTrainer" in wishedClassNameList:

        exportedClasses["DTrainer"] = DTrainer

    return exportedClasses
