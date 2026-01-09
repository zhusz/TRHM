import concurrent.futures
import os
import pickle
import random
import sys
from functools import partial
from socket import gethostname

import cv2
import numpy as np
import torch

from ..cache import LgtDatasetCache

from .ray_utils_np import get_ray_directions, get_rays, readjustE, get_ray_directions_with_shift

from .renderingLightStageLfhyper import OnlineAugmentableEnvmapDatasetCache


projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../"
sys.path.append(projRoot + "src/versions/")
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb_np
from codes_py.np_ext.data_processing_utils_v1 import normalize


def static_load_data_no_lgt(
    j,
    indList,  # indList could be indTrain, or just [0]. For indTrain: index == indTrain[j]
    projRoot,
    nameList,
    ifLoadHdr,
    ifLoadImg,
    flagSplit,
    nameListExr,
    white_back,
    img_wh,
    dataset,
    maskDilateKernel,
    ifLoadDepth,
    blenderTagDepth,
    nameListDepth,
    ifLoadNormal,
    blenderTagNormal,
    nameListNormal,
    hdrScaling,
    sceneScaleSecond,
    loadDepthTag,
):
    index = int(indList[j])

    # load raw
    if ifLoadImg:
        # img_raw = unio.load_image(unioRoot + nameList[index], uconf).numpy() / 255.0

        assert os.path.isfile(projRoot + nameList[index]), (projRoot + nameList[index], index)
        img_raw = cv2.imread(projRoot + nameList[index], flags=cv2.IMREAD_UNCHANGED)
        assert img_raw.dtype == np.uint8

        assert len(img_raw.shape) == 3
        assert img_raw.shape[2] == 4
        assert float(img_wh[0]) / float(img_wh[1]) == float(img_raw.shape[1]) / float(
            img_raw.shape[0]
        )
        img_raw = img_raw[:, :, [2, 1, 0, 3]].astype(np.float32) / 255.0  # bgra --> rgba, and [0, 255] to [0, 1], uint8 to float32
        img_raw = cv2.resize(
            img_raw, (img_wh[0], img_wh[1]), interpolation=cv2.INTER_CUBIC
        )
    else:
        img_raw = np.zeros((img_wh[1], img_wh[0], 4), dtype=np.float32)
    if ifLoadHdr:
        assert os.path.isfile(projRoot + nameListExr[index]), projRoot + nameListExr[index]
        hdr_raw = cv2.imread(projRoot + nameListExr[index], flags=cv2.IMREAD_UNCHANGED)
        assert hdr_raw.dtype == np.float32
        # with unio.open(unioRoot + nameListExrnpz[index], uconf, "rb") as f:
        #     hdr_raw = np.load(f)["arr_0"].astype(np.float32)
        assert len(hdr_raw.shape) == 3
        assert hdr_raw.shape[2] == 4
        assert float(img_wh[0]) / float(img_wh[1]) == float(hdr_raw.shape[1]) / float(
            hdr_raw.shape[0]
        )
        hdr_raw = hdr_raw[:, :, [2, 1, 0, 3]]  # bgra --> rgba
        hdr_raw = cv2.resize(
            hdr_raw, (img_wh[0], img_wh[1]), interpolation=cv2.INTER_CUBIC
        )
    else:
        hdr_raw = np.zeros((img_wh[1], img_wh[0], 4), dtype=np.float32)

    # masking (only when ifLoadImg is False we will use the masking from hdr, otherwise, ldr)
    if ifLoadHdr:  # in this case, even if you also load ldr, we use hdr as the masking
        valid_mask = hdr_raw[:, :, 3] > 1.0e-5
        soft_valid_mask = np.copy(hdr_raw[:, :, 3])
    elif ifLoadImg:
        valid_mask = img_raw[:, :, 3] > 1.0e-5
        soft_valid_mask = np.copy(img_raw[:, :, 3])
    else:
        valid_mask = np.ones_like(img_raw[:, :, -1]).astype(bool)
        soft_valid_mask = np.ones_like(img_raw[:, :, -1]).astype(np.float32)

    # dilated masking
    if maskDilateKernel is not None:
        dilated_mask = cv2.dilate(
            valid_mask.astype(np.float32), maskDilateKernel, iterations=1
        ).astype(bool)
    else:
        dilated_mask = np.copy(valid_mask)

    # apply the soft_valid_mask
    if ifLoadImg:
        if white_back:  # typically we use white_back for nerf
            img = (
                img_raw[:, :, :3] * soft_valid_mask[:, :, None]
                + 1.0
                - soft_valid_mask[:, :, None]
            )  # white background
        else:  # (however, light stage always)
            img = img_raw[:, :, :3] * soft_valid_mask[:, :, None]
    else:
        assert (
            not white_back
        )  # in the relighting setting, the background can only be zero
        img = np.copy(img_raw[:, :, :3])
    if ifLoadHdr:
        hdr = hdr_raw[:, :, :3] * soft_valid_mask[:, :, None]
        hdr *= hdrScaling
        # Make sure you always not to use white background in the model when training relighting
    else:
        hdr = np.copy(hdr_raw[:, :, :3])
        hdr *= hdrScaling

    if ifLoadNormal:
        assert ifLoadDepth  # loading the depth is now becoming the pre-requisite of loading the normals

    if ifLoadDepth and (flagSplit[index] > 0):
        if loadDepthTag in ["R2neusdepth", "R2neurecondepth", "R2neusdepth400"]:
            if dataset in ["renderingNerfBlender58Pointk8L112V100", "renderingNerfBlender58Pointk4L112V100"]:
                caseID = index // 11216
                assert 0 <= caseID < 32, caseID
                insideCaseIndex = index % 11216
                if 0 <= insideCaseIndex < 11200:  # flagSplit == 1
                    index_nerfsynthetic = (insideCaseIndex % 100) + 400 * caseID
                elif 11200 <= insideCaseIndex < 11216:
                    index_nerfsynthetic = (100 + (6 * (insideCaseIndex - 11200))) + 400 * caseID
                else:
                    raise ValueError("Range error for insideCaseIndex: %d" % insideCaseIndex)
            elif dataset == "renderingNerfBlender58Pointk8L10V10ben":
                caseID = index // 100
                assert 0 <= caseID < 32, caseID
                insideCaseIndex = index % 100
                index_nerfsynthetic = (200 + (20 * (insideCaseIndex // 10))) + 400 * caseID
            elif dataset == "renderingNerfBlender58Pointk8rot4":
                caseID = index // 424
                assert 0 <= caseID < 32, caseID
                insideCaseIndex = index % 424
                if insideCaseIndex < 224:
                    index_nerfsynthetic = 128 + 400 * caseID
                else:
                    index_nerfsynthetic = insideCaseIndex - 224 + 200 + 400 * caseID
            else:
                raise NotImplementedError("Not yet defined for loadDepthTag (%s) for dataset (%s)" % (loadDepthTag, dataset))
            depth_fn = projRoot + "v/R/%s/%s/%08d.pkl" % ("renderingNerfBlenderExisting", loadDepthTag, index_nerfsynthetic)
            assert os.path.isfile(depth_fn), depth_fn
            with open(depth_fn, "rb") as f:
                tmp = pickle.load(f)
                depth = tmp["zbufferDepth0"]
                depth[np.isfinite(depth) == 0] = 0
                assert depth.shape == (img_wh[1], img_wh[0])
                if ifLoadNormal:
                    normal = tmp["normalWorld0"]
                    assert np.all(np.isfinite(normal))
                    assert normal.shape == (img_wh[1], img_wh[0], 3)
                    normal_norm = np.linalg.norm(normal, axis=2, ord=2)
                    check1 = np.abs(normal_norm - 1) < 1.0e-4
                    check0 = np.abs(normal_norm - 0) < 1.0e-4
                    """
                    if index_nerfsynthetic not in [2842, 2865, 2897]:  # These are very rare cases where the foreground normal norm is too small (and get between 0~1 L2 norm)
                        if not np.all(check1 | check0):
                            print(index_nerfsynthetic)
                            print(np.where((check1 == 0) & (check0 == 0)))
                            import ipdb
                            ipdb.set_trace()
                            print(1 + 1)
                        assert np.all(check1 | check0)
                    """
                else:
                    normal = np.zeros((img_wh[1], img_wh[0], 3), dtype=np.float32)
        elif loadDepthTag in ["NeuS"]:
            if dataset.startswith("renderingCapture7jfHdrPointFree2"):  
                # light stage data arbitrary view
                # case major / lgt middle / view minor
                # so only need to render the first lgt's view
                mLgt = 10
                mView = 18
                caseID = index // (mLgt * mView)
                insideCaseIndex = index % (mLgt * mView)
                index_self = (insideCaseIndex % mView) + caseID * (mLgt * mView)
            else:
                raise NotImplementedError("Not yet defined for loadDepthTag (%s) for dataset (%s)" % (loadDepthTag, dataset))
            depth_fn = projRoot + "v/R/%s/R2neusdepth/%08d.pkl" % (dataset, index_self)
            assert os.path.isfile(depth_fn), depth_fn
            with open(depth_fn, "rb") as f:
                tmp = pickle.load(f)
                depth = tmp["zbufferDepth0"]
                depth[np.isfinite(depth) == 0] = 0
                assert depth.shape == (img_wh[1], img_wh[0])
                if ifLoadNormal:
                    normal = tmp["normalWorld0"]
                    assert np.all(np.isfinite(normal))
                    assert normal.shape == (img_wh[1], img_wh[0], 3)
                    normal_norm = np.linalg.norm(normal, axis=2, ord=2)
                    check1 = np.abs(normal_norm - 1) < 1.0e-4
                    check0 = np.abs(normal_norm - 0) < 1.0e-4
                    """
                    if index_nerfsynthetic not in [2842, 2865, 2897]:  # These are very rare cases where the foreground normal norm is too small (and get between 0~1 L2 norm)
                        if not np.all(check1 | check0):
                            print(index_nerfsynthetic)
                            print(np.where((check1 == 0) & (check0 == 0)))
                            import ipdb
                            ipdb.set_trace()
                            print(1 + 1)
                        assert np.all(check1 | check0)
                    """
                else:
                    normal = np.zeros((img_wh[1], img_wh[0], 3), dtype=np.float32)
        else:
            raise NotImplementedError("Unknown loadDepthTag: %s" % loadDepthTag)

        depth = np.ascontiguousarray(depth)
        normal = np.ascontiguousarray(normal)

        scaledDepth = depth * sceneScaleSecond
    else:
        scaledDepth = np.zeros((img_wh[1], img_wh[0]), dtype=np.float32)
        normal = np.zeros((img_wh[1], img_wh[0], 3), dtype=np.float32)

    # hdr clip
    if ifLoadHdr:
        hdr = np.clip(hdr, a_min=0, a_max=np.inf)

    # img into uint8
    img = (img * 255.0).astype(np.uint8)

    return (
        j,
        img.reshape((img_wh[0] * img_wh[1], 3)),
        valid_mask.reshape((img_wh[0] * img_wh[1],)),
        hdr.reshape((img_wh[0] * img_wh[1], 3)),
        dilated_mask.reshape((img_wh[0] * img_wh[1])),
        scaledDepth.reshape((img_wh[0] * img_wh[1])),
        normal.reshape((img_wh[0] * img_wh[1], 3)),
        img_raw.reshape((img_wh[0] * img_wh[1], 4)),
        hdr_raw.reshape((img_wh[0] * img_wh[1], 4)),
    )


class RenderingNerfBlenderDataset(object):
    def __init__(self, datasetConf, **kwargs):
        self.datasetConf = datasetConf

        self.if_need_metaDataLoading = kwargs["if_need_metaDataLoading"]

        self.rank = int(os.environ.get("RANK", 0))
        self.numMpProcess = int(os.environ.get("WORLD_SIZE", 0))

        split = datasetConf["split"]  # train, val
        img_wh = (datasetConf["winWidth"], datasetConf["winHeight"])  # e.g. (800, 800)
        singleImageMode = datasetConf["singleImageMode"]
        if split in ["val", "test"]:
            assert singleImageMode

        self.split = split
        self.singleImageMode = singleImageMode
        # assert img_wh[0] == img_wh[1], "image width must equal image height!"
        self.img_wh = img_wh

        # default from datasetConf
        #   Define your default datasetConf values only here, not anywhere below
        self.ifLoadDepth = datasetConf.get("ifLoadDepth", False)
        self.ifLoadNormal = datasetConf.get("ifLoadNormal", False)
        self.blenderFrameTag = datasetConf.get("blenderFrameTag", None)
        self.ifPreloadModeFgbg = datasetConf.get("ifPreloadModeFgbg", False)
        self.hdrScaling = datasetConf.get("hdrScaling", 1.0)
        # When your camera-to-objCentroid distance is out-of-distribution, readjust the camera location 
        # so that you can maintain your sampling depth range to be between the training scenario
        self.readjustERadius = datasetConf.get("readjustERadius", None)  
        self.partialPixelPred = datasetConf.get("partialPixelPred", False)
        self.ifPreloadHighlight = datasetConf.get("ifPreloadHighlight", False)

        # At the moment we assume a fixed bucket "nerf_dataset".
        self.projRoot = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../") + "/"

        # header files loading
        self.read_meta()

        # cache
        self.meta = {}
        if datasetConf["lgtLoadingMode"] in []:
            self.meta["lgtDatasetCache"] = LgtDatasetCache(self.projRoot)
        elif datasetConf["lgtLoadingMode"] in ["ENVMAPLFHYPER"]:
            assert len(set(self.A0["lgtDatasetList"])) == 1
            self.meta["lgtDatasetCache"] = OnlineAugmentableEnvmapDatasetCache(
                self.A0["lgtDatasetList"][0],
                projRoot=self.projRoot,
                quantile_cut_min=datasetConf["lgt_quantile_cut_min"],
                quantile_cut_max=datasetConf["lgt_quantile_cut_max"],
                quantile_cut_fixed=datasetConf["lgt_quantile_cut_fixed"],
                cudaDevice="cuda:0",  # hand-write fixed
                if_lgt_load_highres=datasetConf["if_lgt_load_highres"],
            )
        elif datasetConf["lgtLoadingMode"] in ["QSPL"]:
            pass
        elif datasetConf["lgtLoadingMode"] in ["NONE"]:
            pass
        else:
            raise NotImplementedError("Unknown datasetConf.lgtLoadingMode: %s" % datasetConf["lgtLoadingMode"])
        self.meta["lgtEnvmapPreload"] = None  # will be set to a (len(indTrain), height_lgt, width_lgt) matrix leter on.

        if self.ifPreloadHighlight:
            self.preload_highlight()

    def furtherReadjustFlagSplit(self):
        pass  # typically to be overriden

    def preload_highlight(self):
        # if not self.if_need_metaDataLoading:
        #     return

        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        caseID = int(datasetConf["caseID"])
        fn_highlight = projRoot + "v/misc/%s/%s_caseID_%d_highlight.pkl" % (
            dataset, dataset, caseID,           
        )
        assert os.path.isfile(fn_highlight), fn_highlight
        print("Loading highlights and sublights: %s" % fn_highlight)
        with open(fn_highlight, "rb") as f:
            self.highlight = pickle.load(f)

        for highOrSubLight in ["highlight", "sublight"]:
            self.highlight["hdr_%s" % highOrSubLight] *= self.hdrScaling

        # Remove any highlight label that is not part of the training set
        for highOrSubLight in ["highlight", "sublight"]:
            if datasetConf["highlightSplit"] == 1:
                flag_in = np.isin(self.highlight["indexID_%s"% highOrSubLight], self.indTrain)
            elif datasetConf["highlightSplit"] == 3:
                flag_in = np.isin(self.highlight["indexID_%s"% highOrSubLight], self.indTest)
            else:
                raise NotImplementedError("Unknown datasetConf.highlightSplit: %d" % datasetConf["highlightSplit"])
            assert len(flag_in.shape) == 1
            ind_in = np.where(flag_in)[0]
            for k in ["indexID", "pixelID", "hdr"]:
                self.highlight["%s_%s" % (k, highOrSubLight)] = self.highlight["%s_%s" % (k, highOrSubLight)][ind_in]
        
        # get the depth or the normal
        if self.ifLoadDepth or self.ifLoadNormal:
            # only necessary to training dataset
            if dataset in ["renderingNerfBlender58Pointk8L112V100"]:  # load from renderingNerfBlenderExisting
                baseDataset = "renderingNerfBlenderExisting"
                assert self.img_wh[0] == 800 and self.img_wh[1] == 800
                caseID = int(datasetConf["caseID"])
                if self.ifLoadDepth:
                    tmp_record_scaledDepth = np.zeros((100 * self.img_wh[0] * self.img_wh[1]), dtype=np.float32)
                if self.ifLoadNormal:
                    tmp_record_normal = np.zeros((100 * self.img_wh[0] * self.img_wh[1], 3), dtype=np.float32)
                    assert self.ifLoadDepth  # loading depth is a pre-requisite of loading normal
                for viewID in range(100):
                    baseJ = 400 * caseID + viewID
                    base_depth_fn = projRoot + "v/R/%s/%s/%08d.pkl" % (
                        baseDataset, datasetConf["loadDepthTag"], baseJ,
                    )
                    assert os.path.isfile(base_depth_fn), base_depth_fn
                    print("    Working for the highlight/sublight with depth/normal. Preloading %s" % base_depth_fn)
                    with open(base_depth_fn, "rb") as f:
                        pkl = pickle.load(f)
                    if self.ifLoadDepth:
                        tmp_record_scaledDepth[viewID * (self.img_wh[0] * self.img_wh[1]):(viewID + 1) * (self.img_wh[0] * self.img_wh[1])] = (
                            pkl["zbufferDepth0"].reshape(-1)
                        ) * datasetConf["sceneScaleSecond"]
                    if self.ifLoadNormal:
                        tmp_record_normal[viewID * (self.img_wh[0] * self.img_wh[1]):(viewID + 1) * (self.img_wh[0] * self.img_wh[1]), :] = (
                            pkl["normalWorld0"].reshape(-1, 3)
                        )
                for highOrSubLight in ["highlight", "sublight"]:
                    indexID = self.highlight["%s_%s" % ("indexID", highOrSubLight)]
                    pixelID = self.highlight["%s_%s" % ("pixelID", highOrSubLight)]
                    if dataset in ["renderingNerfBlender58Pointk8L112V100"]:  # the mapping from indexID to viewID is: viewID = (indexID - caseID * 400) % 100
                        viewID = (indexID - 11216 * caseID) % 100
                    else:
                        raise NotImplementedError("Unknown dataset: %s" % dataset)
                    viewPixelID = viewID.astype(np.int64) * (self.img_wh[0] * self.img_wh[1]) + pixelID.astype(np.int64)
                    found = viewPixelID  # no need for searchsorted
                    if self.ifLoadDepth:
                        self.highlight["%s_%s" % ("scaledDepth", highOrSubLight)] = tmp_record_scaledDepth[found]
                        # assert np.all(np.isfinite(tmp_record_scaledDepth[found]))
                        """
                        if not np.all(np.isfinite(tmp_record_scaledDepth[found])):
                            import ipdb
                            ipdb.set_trace()
                            print(1 + 1)
                        """
                    if self.ifLoadNormal:
                        self.highlight["%s_%s" % ("normal", highOrSubLight)] = tmp_record_normal[found]
            elif dataset in ["renderingNerfBlender58Pointk8L10V10ben"]:  # still load from renderingNerfBlenderExisting
                baseDataset = "renderingNerfBlenderExisting"
                assert self.img_wh[0] == 800 and self.img_wh[1] == 800
                caseID = int(datasetConf["caseID"])
                if self.ifLoadDepth:
                    tmp_record_scaledDepth = np.zeros((10 * self.img_wh[0] * self.img_wh[1]), dtype=np.float32)
                if self.ifLoadNormal:
                    tmp_record_normal = np.zeros((10 * self.img_wh[0] * self.img_wh[1], 3), dtype=np.float32)
                    assert self.ifLoadDepth  # loading depth is a pre-requisite of loading normal
                for viewID in range(10):
                    baseJ = 400 * caseID + 200 + 20 * viewID
                    base_depth_fn = projRoot + "v/R/%s/%s/%08d.pkl" % (
                        baseDataset, datasetConf["loadDepthTag"], baseJ,
                    )
                    assert os.path.isfile(base_depth_fn), base_depth_fn
                    print("    Working for the highlight/sublight with depth/normal. Preloading %s" % base_depth_fn)
                    with open(base_depth_fn, "rb") as f:
                        pkl = pickle.load(f)
                    if self.ifLoadDepth:
                        tmp_record_scaledDepth[viewID * (self.img_wh[0] * self.img_wh[1]):(viewID + 1) * (self.img_wh[0] * self.img_wh[1])] = (
                            pkl["zbufferDepth0"].reshape(-1)
                        ) * datasetConf["sceneScaleSecond"]
                    if self.ifLoadNormal:
                        tmp_record_normal[viewID * (self.img_wh[0] * self.img_wh[1]):(viewID + 1) * (self.img_wh[0] * self.img_wh[1]), :] = (
                            pkl["normalWorld0"].reshape(-1, 3)
                        )
                for highOrSubLight in ["highlight", "sublight"]:
                    indexID = self.highlight["%s_%s" % ("indexID", highOrSubLight)]
                    pixelID = self.highlight["%s_%s" % ("pixelID", highOrSubLight)]
                    # if dataset in ["renderingNerfBlender58Pointk8L112V100"]:  # the mapping from indexID to viewID is: viewID = (indexID - caseID * 400) % 100
                    #     viewID = (indexID - 11216 * caseID) % 100
                    if dataset in ["renderingNerfBlender58Pointk8L10V10ben"]:
                        viewID = (indexID - 100 * caseID) // 10
                        assert 0 <= viewID.min() < viewID.max() < 10
                    else:
                        raise NotImplementedError("Unknown dataset: %s" % dataset)
                    viewPixelID = viewID.astype(np.int64) * (self.img_wh[0] * self.img_wh[1]) + pixelID.astype(np.int64)
                    found = viewPixelID  # no need for searchsorted
                    if self.ifLoadDepth:
                        self.highlight["%s_%s" % ("scaledDepth", highOrSubLight)] = tmp_record_scaledDepth[found]
                        # assert np.all(np.isfinite(tmp_record_scaledDepth[found]))
                        """
                        if not np.all(np.isfinite(tmp_record_scaledDepth[found])):
                            import ipdb
                            ipdb.set_trace()
                            print(1 + 1)
                        """
                    if self.ifLoadNormal:
                        self.highlight["%s_%s" % ("normal", highOrSubLight)] = tmp_record_normal[found]
            else:
                raise NotImplementedError("Unknown dataset: %s" % dataset)

        else:
            pass  # do nothing

    def read_meta(self):
        with open(
            self.projRoot + "v/A/%s/A0_randomness.pkl" % self.datasetConf["dataset"],
            "rb",
        ) as f:
            self.A0 = pickle.load(f)

        # flagSplit
        self.flagSplit = self.A0["flagSplit"].copy()
        self.flagSplit[self.A0["caseIDList"] != self.datasetConf["caseID"]] = 0
        self.furtherReadjustFlagSplit()  # typically to be overriden - e.g. you only wish to train with a single group from the light stage
        if (
            (self.numMpProcess > 1)
            and ((self.flagSplit == 1).sum() > 100)
            and (not self.datasetConf.get("debugReadTwo", False))
            and (not self.datasetConf.get("debugReadEvery", False))
        ):
            # mask out the segments that does not belong to this GPU
            indTrainNow = np.where(self.flagSplit == 1)[0]
            p1 = int(self.rank * float(indTrainNow.shape[0]) / self.numMpProcess)
            p2 = int((self.rank + 1) * float(indTrainNow.shape[0] / self.numMpProcess))
            assert p2 - p1 > self.numMpProcess, (p2, p1, self.numMpProcess, self.rank)
            indTrainNow = indTrainNow[p1:p2]
            flagSplit = np.zeros_like(self.flagSplit)
            flagSplit[indTrainNow] = 1
            flagSplit[self.flagSplit == 2] = 2
            flagSplit[self.flagSplit == 3] = 3
            self.flagSplit = flagSplit
            # self.flagSplit[indTrainNow] = 1  # this line is redundent
        self.indTrain = np.where(self.flagSplit == 1)[0]

        # quick debugging - Make sure you distable this when you are not running Rtmp
        if self.datasetConf.get("debugReadTwo", False):
            assert not self.datasetConf.get("debugReadEvery", False)
            debugReadHowMany = int(self.datasetConf.get("debugReadHowMany", 2))
            assert debugReadHowMany < self.indTrain.shape[0]
            self.flagSplit[self.indTrain[debugReadHowMany:]] = 0
            self.indTrain = np.where(self.flagSplit == 1)[0]
        elif self.datasetConf.get("debugReadEvery", False):
            assert not self.datasetConf.get("debugReadTwo", False)
            debugReadEveryHowMany = int(
                self.datasetConf.get("debugReadEveryHowMany", 2)
            )
            assert debugReadEveryHowMany < self.indTrain.shape[0]
            self.indTrain = self.indTrain[::debugReadEveryHowMany]
            self.flagSplit[self.flagSplit == 1] = 0
            self.flagSplit[self.indTrain] = 1

        self.indVal = np.where(self.flagSplit == 2)[0]
        self.indTest = np.where(self.flagSplit == 3)[0]
        self.ind = np.where(self.flagSplit > 0)[0]
        self.mTrain = (self.flagSplit == 1).sum()
        self.mVal = (self.flagSplit == 2).sum()
        self.mTest = (self.flagSplit == 3).sum()
        assert len(self.ind) > 0

        # inverse mapping of indTrain
        indTrainInverse = (1 + self.A0["m"]) * np.ones((int(self.A0["m"]),), dtype=np.int32)  # (1 + m) aims to stop incorrect mapping of 0 of -1 (should have be nan but this is int32)
        indTrainInverse[self.indTrain] = np.arange(len(self.indTrain)).astype(np.int32)
        self.indTrainInverse = indTrainInverse

        # Compute OpenCV intrinsic parameters.
        w, h = self.img_wh
        if "fovRad" in self.A0.keys():
            assert len(np.unique(self.A0["fovRad"][self.ind])) == 1
            fovRad0 = float(self.A0["fovRad"][self.ind][0])
            focal = 0.5 * self.img_wh[0] / np.tan(0.5 * fovRad0)
            # Note that this is not self.A0["focalLength"] because now you customized your own winSize
            # self.fov = fovRad0
            self.fx = focal
            self.fy = focal
        else:
            assert len(np.unique(self.A0["focalLengthWidth"][self.ind])) == 1
            assert len(np.unique(self.A0["focalLengthHeight"][self.ind])) == 1
            self.fx = float(self.A0["focalLengthWidth"][self.ind][0])
            self.fy = float(self.A0["focalLengthHeight"][self.ind][0])
            assert self.fx == self.fy, (self.fx, self.fy)

        # For the principal point we assume that it is in the middle of the image,
        # and we use OpenCV notation, i.e. position (0, 0) is not on top-left corner,
        # but instead in the middle of the top-left pixel.
        self.cx = self.img_wh[0] / 2.0 - 0.5
        self.cy = self.img_wh[1] / 2.0 - 0.5

        # Bounds, common for all scenes
        # near and far: now requries explicit inputs from datasetConf
        self.near = float(self.datasetConf.get("ray_near", np.nan))  # 2.0
        self.far = float(self.datasetConf.get("ray_far", np.nan))  # 6.0
        # self.bounds = np.array([self.near, self.far])

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            h, w, self.fx, self.fy, self.cx, self.cy
        )  # (h, w, 3)
        self.directions_flattened = self.directions.reshape((-1, 3))  # (h * w, 3)
        assert np.all(self.directions_flattened[:, 2] == 1.0)
        self.zbuffer_to_euclidean_ratio = np.linalg.norm(self.directions_flattened, ord=2, axis=1)

        directions_normalized = normalize(self.directions_flattened, 1)
        directions_x_shift_1 = normalize(get_ray_directions_with_shift(
            h, w, self.fx, self.fy, self.cx, self.cy, 1, 0
        ).reshape((-1, 3)), 1)
        directions_y_shift_1 = normalize(get_ray_directions_with_shift(
            h, w, self.fx, self.fy, self.cx, self.cy, 0, 1
        ).reshape((-1, 3)), 1)
        dx = np.sqrt(np.sum((directions_normalized - directions_x_shift_1) ** 2, axis=-1))
        dy = np.sqrt(np.sum((directions_normalized - directions_y_shift_1) ** 2, axis=-1))
        pixel_area = dx * dy
        self.pixel_area_flattened = pixel_area.reshape(-1)

        # mask dilate
        if self.datasetConf["ifMaskDilate"] > 0:
            assert type(self.datasetConf["ifMaskDilate"]) is int
            L = self.datasetConf["ifMaskDilate"]
            self.maskDilateKernel = np.ones((L, L), dtype=np.uint8)
            del L
        else:
            self.maskDilateKernel = None

        if self.split == "train" and not self.singleImageMode and self.if_need_metaDataLoading:
            # index level is sufficient on memory, but the pixel level is insufficient
            # each pixel should only contain these things (for memory saving)
            #   index (connecting image and all camera pose / sh) (1 int),
            #   rayID (insde the same size of the image map) (1 int),
            #   rgb (3 uint8), mask_valid (1 bool), hdr (3 float32)
            # For sh, you need to do assert len(self.meta["lgtDatasetCache"]) == 1
            #   (only one lgtDataset throughout all the samples - for fast sh retrieval during __getitem__)

            # memory_bank: dynamically update the pre-stored dataset
            # only useful for a training dataset that requires data pre-load
            if "memory_bank_size_per_process" in self.datasetConf.keys() and self.datasetConf["memory_bank_size_per_process"] > 0:
                assert type(self.datasetConf["memory_bank_size_per_process"]) is int
                assert len(self.indTrain) >= self.datasetConf["memory_bank_size_per_process"], (len(self.indTrain), self.datasetConf["memory_bank_size_per_process"])
                initial_indTrain = self.indTrain[np.random.permutation(self.datasetConf["memory_bank_size_per_process"])]
            else:
                initial_indTrain = self.indTrain

            hostname = gethostname()
            world_size = max(int(os.environ.get("WORLD_SIZE", 0)), 1)
            assert 1 <= world_size <= 8
            assert 8 % world_size == 0
            max_workers = (8 // world_size) * 1
            # max_workers = min(4, max_workers)
            max_workers = 0
            if max_workers <= 2:
                max_workers = 0
            # max_workers_lgt = 0  # always only allow one thread for loading the lgt per DDP process

            f_partial = partial(
                static_load_data_no_lgt,
                indList=initial_indTrain,  # self.indTrain,
                projRoot=self.projRoot,
                nameList=self.A0.get("nameList", None),
                ifLoadImg=self.datasetConf.get("ifLoadImg", True),
                ifLoadHdr=self.datasetConf["ifLoadHdr"],
                flagSplit=self.A0["flagSplit"],
                nameListExr=self.A0.get("nameListExr", None),
                white_back=self.datasetConf["white_back"],
                img_wh=self.img_wh,
                dataset=self.datasetConf["dataset"],
                maskDilateKernel=self.maskDilateKernel,
                ifLoadDepth=self.datasetConf.get("ifLoadDepth", False),
                blenderTagDepth=self.blenderFrameTag,
                nameListDepth=self.A0.get(self.datasetConf.get("depthFileListA0Tag", None), None),
                ifLoadNormal=self.datasetConf.get("ifLoadNormal", False),
                blenderTagNormal=self.blenderFrameTag,
                nameListNormal=self.A0.get("nameListExrNormal", None),
                hdrScaling=self.hdrScaling,
                sceneScaleSecond=self.datasetConf["sceneScaleSecond"],
                loadDepthTag=self.datasetConf.get("loadDepthTag", None),
            )

            m_rays = self.mTrain * self.img_wh[0] * self.img_wh[1]
            # self.m_rays = m_rays
            m_memory_bank = len(initial_indTrain) * self.img_wh[0] * self.img_wh[1]
            self.m_memory_bank = m_memory_bank
            if "memory_bank_size_per_process" not in self.datasetConf.keys():
                self.m_rays = m_rays
            if self.ifPreloadModeFgbg:
                # fg_upper_bound = m_memory_bank // 2
                fg_upper_bound = int(m_memory_bank * 0.6)
                bg_upper_bound = m_memory_bank
                self.fg_rgbs = np.zeros((fg_upper_bound, 3), dtype=np.uint8)
                self.fg_masks = np.zeros((fg_upper_bound,), dtype=bool)
                self.fg_hdrs = np.zeros((fg_upper_bound, 3), dtype=np.float32)
                self.fg_dilates = np.zeros(fg_upper_bound, dtype=bool)  # * np.nan
                self.bg_dilates = np.zeros(bg_upper_bound, dtype=bool)  # * np.nan
                # for each pixel ray, which image sample does it belong to?
                self.fg_indexID = np.zeros((fg_upper_bound,), dtype=np.int32)
                self.bg_indexID = np.zeros((bg_upper_bound,), dtype=np.int32)
                # for each pixel ray, which pixel ID is it located on its own image map?
                self.fg_pixelID = np.zeros((fg_upper_bound,), dtype=np.int32)
                self.bg_pixelID = np.zeros((bg_upper_bound,), dtype=np.int32)
                if self.ifLoadDepth:
                    self.fg_scaledDepths = np.zeros((fg_upper_bound,), dtype=np.float32)
                if self.ifLoadNormal:
                    self.fg_normals = np.zeros((fg_upper_bound, 3), dtype=np.float32)
                fg_count = 0
                bg_count = 0
            else:
                self.all_rgbs = np.zeros((m_memory_bank, 3), dtype=np.uint8)
                self.all_masks = np.zeros((m_memory_bank,), dtype=bool)  # * np.nan
                self.all_hdrs = np.zeros((m_memory_bank, 3), dtype=np.float32)
                self.all_dilates = np.zeros(m_memory_bank, dtype=bool)  # * np.nan
                # for each pixel ray, which image sample does it belong to?
                self.all_indexID = np.zeros((m_memory_bank,), dtype=np.int32)
                # for each pixel ray, which pixel ID is it located on its own image map?
                self.all_pixelID = np.zeros((m_memory_bank,), dtype=np.int32)
                if self.ifLoadDepth:
                    self.all_scaledDepths = np.zeros((m_memory_bank,), dtype=np.float32)
                if self.ifLoadNormal:
                    self.all_normals = np.zeros((m_memory_bank, 3), dtype=np.float32)
            
            if max_workers > 0:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    threads = [
                        executor.submit(f_partial, j) for j in range(len(initial_indTrain))
                    ]
                    for thread in concurrent.futures.as_completed(threads):
                        # try:
                        if True:
                            j, img, valid_mask, hdr, dilated_mask, scaledDepth, normal, img_raw, hdr_raw = thread.result()
                            j = int(j)
                            if j % 100 == 0:
                                print("%d / %d (max_workers = %d)" % (j, len(initial_indTrain), max_workers))
                            if self.ifPreloadModeFgbg:
                                fg_ind_current = np.where(valid_mask > 0)[0]
                                bg_ind_current = np.where(valid_mask <= 0)[0]
                                fg_current = int(fg_ind_current.shape[0])
                                bg_current = int(bg_ind_current.shape[0])
                                assert fg_count + fg_current <= fg_upper_bound, (fg_count, fg_current, fg_upper_bound)
                                assert bg_count + bg_current <= bg_upper_bound, (bg_count, bg_current, bg_upper_bound)
    
                                self.fg_rgbs[fg_count:fg_count + fg_current, :] = img[fg_ind_current, :]
                                self.fg_masks[fg_count:fg_count + fg_current] = valid_mask[fg_ind_current]
                                self.fg_hdrs[fg_count:fg_count + fg_current, :] = hdr[fg_ind_current, :]
                                self.fg_dilates[fg_count:fg_count + fg_current] = dilated_mask[fg_ind_current]
                                self.bg_dilates[bg_count:bg_count + bg_current] = dilated_mask[bg_ind_current]
                                self.fg_indexID[fg_count:fg_count + fg_current] = int(initial_indTrain[j]) * np.ones(
                                    (fg_current,), dtype=np.int32
                                )
                                self.bg_indexID[bg_count:bg_count + bg_current] = int(initial_indTrain[j]) * np.ones(
                                    (bg_current,), dtype=np.int32
                                )
                                self.fg_pixelID[fg_count:fg_count + fg_current] = np.arange(
                                    img.shape[0])[fg_ind_current].astype(np.int32)
                                self.bg_pixelID[bg_count:bg_count + bg_current] = np.arange(
                                    img.shape[0])[bg_ind_current].astype(np.int32)
                                if self.ifLoadDepth:
                                    self.fg_scaledDepths[fg_count:fg_count + fg_current] = scaledDepth[fg_ind_current]
                                if self.ifLoadNormal:
                                    self.fg_normals[fg_count:fg_count + fg_current] = normal[fg_ind_current, :]
                                fg_count += fg_current
                                bg_count += bg_current
                            else:
                                head = j * (self.img_wh[0] * self.img_wh[1])
                                tail = head + (self.img_wh[0] * self.img_wh[1])
                                self.all_rgbs[head:tail, :] = img
                                self.all_masks[head:tail] = valid_mask
                                self.all_hdrs[head:tail, :] = hdr
                                self.all_dilates[head:tail] = dilated_mask
                                self.all_indexID[head:tail] = int(
                                    initial_indTrain[j]
                                ) * np.ones((img.shape[0],), dtype=np.int32)
                                self.all_pixelID[head:tail] = np.arange(
                                    img.shape[0]
                                ).astype(np.int32)
                                if self.ifLoadDepth:
                                    self.all_scaledDepths[head:tail] = scaledDepth
                                if self.ifLoadNormal:
                                    self.all_normals[head:tail] = normal
                        # except Exception as exc:
                        #     print(f"Multi-threading image loading error {exc}")
            else:
                for j in range(len(initial_indTrain)):
                    j, img, valid_mask, hdr, dilated_mask, scaledDepth, normal, img_raw, hdr_raw = f_partial(j)
                    j = int(j)
                    if j % 100 == 0:
                        print("%d / %d (max_workers = %d)" % (j, len(initial_indTrain), max_workers))
                    if self.ifPreloadModeFgbg:
                        fg_ind_current = np.where(valid_mask > 0)[0]
                        bg_ind_current = np.where(valid_mask <= 0)[0]
                        fg_current = int(fg_ind_current.shape[0])
                        bg_current = int(bg_ind_current.shape[0])
                        assert fg_count + fg_current <= fg_upper_bound 
                        assert bg_count + bg_current <= bg_upper_bound

                        self.fg_rgbs[fg_count:fg_count + fg_current, :] = img[fg_ind_current, :]
                        self.fg_masks[fg_count:fg_count + fg_current] = valid_mask[fg_ind_current]
                        self.fg_hdrs[fg_count:fg_count + fg_current, :] = hdr[fg_ind_current, :]
                        self.fg_dilates[fg_count:fg_count + fg_current] = dilated_mask[fg_ind_current]
                        self.bg_dilates[bg_count:bg_count + bg_current] = dilated_mask[bg_ind_current]
                        self.fg_indexID[fg_count:fg_count + fg_current] = int(initial_indTrain[j]) * np.ones(
                            (fg_current,), dtype=np.int32
                        )
                        self.bg_indexID[bg_count:bg_count + bg_current] = int(initial_indTrain[j]) * np.ones(
                            (bg_current,), dtype=np.int32
                        )
                        self.fg_pixelID[fg_count:fg_count + fg_current] = np.arange(
                            img.shape[0])[fg_ind_current].astype(np.int32)
                        self.bg_pixelID[bg_count:bg_count + bg_current] = np.arange(
                            img.shape[0])[bg_ind_current].astype(np.int32)
                        if self.ifLoadDepth:
                            self.fg_scaledDepths[fg_count:fg_count + fg_current] = scaledDepth[fg_ind_current]
                        if self.ifLoadNormal:
                            self.fg_normals[fg_count:fg_count + fg_current] = normal[fg_ind_current, :]
                        fg_count += fg_current
                        bg_count += bg_current
                    else:
                        head = j * (self.img_wh[0] * self.img_wh[1])
                        tail = head + (self.img_wh[0] * self.img_wh[1])
                        self.all_rgbs[head:tail, :] = img
                        self.all_masks[head:tail] = valid_mask
                        self.all_hdrs[head:tail, :] = hdr
                        self.all_dilates[head:tail] = dilated_mask
                        self.all_indexID[head:tail] = int(initial_indTrain[j]) * np.ones(
                            (img.shape[0],), dtype=np.int32
                        )
                        self.all_pixelID[head:tail] = np.arange(img.shape[0]).astype(
                            np.int32
                        )
                        if self.ifLoadDepth:
                            self.all_scaledDepths[head:tail] = scaledDepth
                        if self.ifLoadNormal:
                            self.all_normals[head:tail] = normal
            if self.ifPreloadModeFgbg:
                self.fg_rgbs = self.fg_rgbs[:fg_count, :]
                self.fg_masks = self.fg_masks[:fg_count]
                self.fg_hdrs = self.fg_hdrs[:fg_count, :]
                self.fg_dilates = self.fg_dilates[:fg_count]
                self.bg_dilates = self.bg_dilates[:bg_count]
                self.fg_indexID = self.fg_indexID[:fg_count]
                self.bg_indexID = self.bg_indexID[:bg_count]
                self.fg_pixelID = self.fg_pixelID[:fg_count]
                self.bg_pixelID = self.bg_pixelID[:bg_count]
                if self.ifLoadDepth:
                    self.fg_scaledDepths = self.fg_scaledDepths[:fg_count]
                if self.ifLoadNormal:
                    self.fg_normals = self.fg_normals[:fg_count, :]
                self.fg_count = fg_count
                self.bg_count = bg_count

            if self.datasetConf["lgtLoadingMode"] in ["SH", "OMEGA", "ENVMAP"]:
                for i, index in enumerate(self.indTrain.tolist()):  # This place is not relevant to memory bank, as you need to hot the cache for all the training cases
                    # only for the purpose of warm the cache
                    self.meta["lgtDatasetCache"].getCache(
                        self.A0["lgtDatasetList"][index]
                    )
                    # for envmap, we also wish to pre-load all of the envmap exrs
                    if self.datasetConf["lgtLoadingMode"] in ["ENVMAP"]:
                        print("    Preloading the envmaps. (rendering)Dataset: %s, i: %d, j: %d, len(indTrain): %d. lgtDataset: %s, index (j): %d" % (
                            self.datasetConf["dataset"], i, index, len(self.indTrain), self.A0["lgtDatasetList"][index], int(self.A0["lgtIDList"][index])
                        ))
                        hdr0 = self.meta["lgtDatasetCache"].getExr(
                            self.A0["lgtDatasetList"][index], int(self.A0["lgtIDList"][index]),
                        )
                        if self.meta["lgtEnvmapPreload"] is None:
                            self.meta["lgtEnvmapPreload"] = np.zeros((len(self.indTrain), hdr0.shape[0], hdr0.shape[1], 3), dtype=np.float32)
                        self.meta["lgtEnvmapPreload"][i, :, :, :] = hdr0
                        # Make very sure that the matrix self.meta["lgtenvmapPredload"] is indexed by i, not j!
                        # This is because int(A0_lgt["m"]) might be very large
                        # This complicates the indexing in Trainer.preprocessBatchTHGPU a little bit.
                        # TODO: so how? Remember also pass out an index that refers to which of indTrain is this sample, meaning: j = indTrain[this-scaler]
                        #   and the mapping from "this-scaler" to "j" should be the inverse mapping of indTrain.
                        #   So you may want to create this inverse mapping and store it in self.meta
                # We assume the whole training set contains only one lgtDataset for fast sh retrieving
                assert len(self.meta["lgtDatasetCache"].cache.keys()) == 1
            elif self.datasetConf["lgtLoadingMode"] == "QSPL":
                pass  # do nothing, no need for warming up the cache
            elif self.datasetConf["lgtLoadingMode"] == "NONE":
                pass
            else:
                raise NotImplementedError(
                    "Unknown datasetConf['lgtLoadingMode']: %s"
                    % self.datasetConf["lgtLoadingMode"]
                )

    def __len__(self):
        if self.split == "train":
            if not self.singleImageMode:
                return len(self.all_rgbs)
            else:
                return self.mTrain
        elif self.split == "val":
            return self.mVal
        elif self.split == "test":
            return self.mTest
        else:
            raise NotImplementedError("Unknown split: %s" % self.split)
        return len(self.meta["frames"])

    def __getitem__(self, indexTrainValTest):
        overfitIndexTvt = self.datasetConf.get("singleSampleOverfittingIndexTvt", 0)

        if self.split == "train":
            if not self.singleImageMode:
                index = indexTrainValTest
                overfitIndex = np.array(0, dtype=np.int32)
                # overfit to one ray...
            else:
                index = self.indTrain[indexTrainValTest]
                overfitIndex = self.indTrain[overfitIndexTvt]
        elif self.split == "val":
            index = self.indVal[indexTrainValTest]
            overfitIndex = self.indVal[overfitIndexTvt]
        elif self.split == "test":
            index = self.indTest[indexTrainValTest]
            overfitIndex = self.indTest[overfitIndexTvt]
        else:
            print(
                "Warning: Since your split is %s, you cannot call this function!"
                % self.split
            )
            raise ValueError

        if self.datasetConf.get("singleSampleOverfittingMode", False):
            index = int(overfitIndex)
        else:
            index = int(index)

        b0np = self.getOneNP(index)
        return b0np

    def getOneNP(self, idx):
        # if self.split == "train":  # use data in the buffers
        if not self.singleImageMode:
            raise ValueError("This part should not be called - do batchPrepceocessingTHGPU directly in the trainer")
        else:  # create data for each image separately
            # Load frame data.
            # img, valid_mask, hdr, dilated_mask = self._load_data(idx)
            _, img, valid_mask, hdr, dilated_mask, scaledDepth, normal, img_raw, hdr_raw = static_load_data_no_lgt(
                j=0,
                indList=[idx],
                projRoot=self.projRoot,
                nameList=self.A0.get("nameList", None),
                ifLoadImg=self.datasetConf.get("ifLoadImg", True),
                ifLoadHdr=self.datasetConf["ifLoadHdr"],
                flagSplit=self.A0["flagSplit"],
                nameListExr=self.A0.get("nameListExr", None),
                white_back=self.datasetConf["white_back"],
                img_wh=self.img_wh,
                dataset=self.datasetConf["dataset"],
                maskDilateKernel=self.maskDilateKernel,
                ifLoadDepth=self.datasetConf.get("ifLoadDepth", False),
                blenderTagDepth=self.blenderFrameTag,
                nameListDepth=self.A0.get(self.datasetConf.get("depthFileListA0Tag", None), None),
                ifLoadNormal=self.datasetConf.get("ifLoadNormal", False),
                blenderTagNormal=self.blenderFrameTag,
                nameListNormal=self.A0.get("nameListExrNormal", None),
                hdrScaling=self.hdrScaling,
                sceneScaleSecond=self.datasetConf["sceneScaleSecond"],
                loadDepthTag=self.datasetConf.get("loadDepthTag", None),
            )

            # Generate rays.
            sceneShiftFirst = self.datasetConf["sceneShiftFirst"]
            assert sceneShiftFirst.shape == (3,) and sceneShiftFirst.dtype == np.float32
            sceneScaleSecond = float(self.datasetConf["sceneScaleSecond"])
            assert sceneScaleSecond > 0
            # inside_cube_coordinates = (original_coordinates - shiftFirst) * scaleSecond
            if self.datasetConf["getRaysMeans"] == "ELU":
                E0 = (self.A0["E"][idx, :] - sceneShiftFirst) * sceneScaleSecond
                L0 = (self.A0["L"][idx, :] - sceneShiftFirst) * sceneScaleSecond
                U0 = self.A0["U"][idx, :]
                w2c = ELU02cam0(np.concatenate([E0, L0, U0], 0))
                c2w = np.linalg.inv(w2c)
                rays_o, rays_d = get_rays(self.directions, c2w[:3, :])
            elif self.datasetConf["getRaysMeans"] == "camInv64":
                camInv640 = self.A0["camInv64"][idx, :3, :].copy()
                camInv640[:, 3] = (camInv640[:, 3] - sceneShiftFirst) * sceneScaleSecond
                rays_o, rays_d = get_rays(self.directions, camInv640)
                rays_o = rays_o.astype(np.float32)
                rays_d = rays_d.astype(np.float32)
                c2w = camInv640.astype(np.float32)
            else:
                raise ValueError("Unknown getRaysMeans: %s" % self.datasetConf["getRaysMeans"])

            if self.readjustERadius:
                rays_o, rays_to_objCentroid0_dist = readjustE(rays_o, rays_d, self.A0["objCentroid"][idx, :], self.readjustERadius)
            elif self.partialPixelPred:  # we need to compute the rays_to_objCentroid0_dist though
                _, rays_to_objCentroid0_dist = readjustE(rays_o, rays_d, self.A0["objCentroid"][idx, :], 0)
            
            if self.partialPixelPred:
                partialPixelPredMaxDistToObjCentroid = self.datasetConf["partialPixelPredMaxDistToObjCentroid"]
                rays_flag = np.all(np.isfinite(rays_o), 1) & (rays_to_objCentroid0_dist < partialPixelPredMaxDistToObjCentroid)
            else:
                rays_flag = np.ones((rays_o.shape[0],), dtype=bool)
            if np.all(rays_flag == 0):
                rays_flag[0] = True  # always not let it to be empty...

            pixel_area = np.copy(self.pixel_area_flattened)
            zbuffer_to_euclidean_ratio = np.copy(self.zbuffer_to_euclidean_ratio)
            random_background_color = (
                1 if self.datasetConf["white_back"] else 0
            ) * np.ones((rays_o.shape[0], 3), dtype=np.float32)  # since it is testing, and we just give the expected background color (rather than the randomly generated ones)
            rays = np.concatenate(
                [
                    rays_o,
                    rays_d,
                    self.near * np.ones_like(rays_o[:, :1]),
                    self.far * np.ones_like(rays_o[:, :1]),
                    pixel_area[:, None],
                    zbuffer_to_euclidean_ratio[:, None],
                    random_background_color,
                ],
                1,
            )  # (H*W, 8)

            index = np.array(int(idx), dtype=np.int32)
            flagSplit = np.array(int(self.flagSplit[index]), dtype=np.int32)
            
            sample = {
                "caseSplitInsideIndex": self.A0["caseSplitInsideIndexList"][idx].astype(np.int64),
                # "viewID": self.A0["viewIDList"][idx].astype(np.int64),
                "rays": rays,
                "rays_flag": rays_flag,
                "rgbs": img.astype(np.float32) / 255.0,
                "c2w": c2w,
                "valid_mask": valid_mask,
                "dilated_mask": dilated_mask,
                "index": index,
                "flagSplit": flagSplit,
                "dataset": self.datasetConf["dataset"],
                "winWidth": np.array(self.img_wh[0], dtype=np.int32),
                "winHeight": np.array(self.img_wh[1], dtype=np.int32),
                "hdrs": hdr,  # alway put, put a zero map if ifLoadHdr is False
                "zbuffer_to_euclidean_ratio": zbuffer_to_euclidean_ratio,
                "scaledDepth": scaledDepth,
                "normal": normalize(normal, 1, eps=1e-12),
            }

            # highlight
            if self.ifPreloadHighlight:
                i_highlight = np.where(self.highlight["indexID_highlight"] == idx)[0]
                pixelID_highlight = self.highlight["pixelID_highlight"][i_highlight]
                i_sublight = np.where(self.highlight["indexID_sublight"] == idx)[0]
                pixelID_sublight = self.highlight["pixelID_sublight"][i_sublight]
                sample["pixelID_highlight"] = pixelID_highlight
                sample["pixelID_sublight"] = pixelID_sublight

            # lgt
            #   visualization purpose
            if (
                (("lgtDatasetList" in self.A0.keys()) or ("lgtIDList" in self.A0.keys())) and
                (self.datasetConf["lgtLoadingMode"] not in ["ENVMAPLFHYPER", "NONE"])
            ):
                assert "lgtDatasetList" in self.A0.keys()
                assert "lgtIDList" in self.A0.keys()
                A0_lgt = self.meta["lgtDatasetCache"].getCache(
                    self.A0["lgtDatasetList"][index]
                )["A0"]
                lgtDataset = self.A0["lgtDatasetList"][index]
                lgtID = int(self.A0["lgtIDList"][index])
                lgtType = A0_lgt.get("lgtType", "Unknown")
                if "nameList" in A0_lgt.keys():
                    R1_fn = self.projRoot + A0_lgt["nameList"][lgtID]
                else:
                    R1_fn = self.projRoot + "v/R/%s/R1/%08d.exr" % (self.A0["lgtDatasetList"][idx], self.A0["lgtIDList"][idx])
                assert os.path.isfile(R1_fn), R1_fn

                tmp = cv2.imread(R1_fn, flags=cv2.IMREAD_UNCHANGED)  # [:, :, ::-1]  # BGR to RGB
                assert len(tmp.shape) in [2, 3]
                if len(tmp.shape) == 2:
                    envmap = np.tile(tmp[:, :, None], (1, 1, 3))
                else:
                    envmap = tmp[:, :, ::-1]

                if (self.img_wh[0] == 1366) and (self.img_wh[1] == 2048):
                    envmapVisWidth, envmapVisHeight = 128 * 3, 64 * 3
                elif (self.img_wh[0] == 800) and (self.img_wh[1] == 1200):
                    envmapVisWidth, envmapVisHeight = 128 * 3, 64 * 3
                elif (self.img_wh[0] == 800) and (self.img_wh[1] == 800):
                    envmapVisWidth, envmapVisHeight = 128 * 3, 64 * 3
                else:
                    raise NotImplementedError("Unknown self.img_wh: %s" % str(self.img_wh))
                envmapResized = cv2.resize(envmap, (envmapVisWidth, envmapVisHeight), interpolation=cv2.INTER_NEAREST)
                envmapVis = np.clip(tonemap_srgb_to_rgb_np(envmapResized), a_min=0, a_max=1)
                if len(envmapVis.shape) == 2:
                    envmapVis = np.stack([envmapVis, envmapVis, envmapVis], 2)
                elif envmapVis.shape[2] == 1:
                    envmapVis = np.concatenate([envmapVis, envmapVis, envmapVis], 2)
                assert envmapVis.shape == (envmapVisHeight, envmapVisWidth, 3), (envmapVis.shape, envmapVisHeight, envmapVisWidth)
                sample["envmapVis"] = envmapVis
            if self.datasetConf.get("lgtLoadingMode", "SH") == "SH":
                lgtDataset = self.A0["lgtDatasetList"][idx]
                lgtID = int(self.A0["lgtIDList"][idx])
                sample["sh"] = self.meta["lgtDatasetCache"].getCache(lgtDataset)["A0"][
                    "sh"
                ][lgtID, :]  # no need to tile, it will do automatic broadcasting. This is late condition.
                sample["lgtType"] = self.meta["lgtDatasetCache"].getCache(lgtDataset)[
                    "A0"
                ]["lgtType"]
            elif self.datasetConf["lgtLoadingMode"] in ["OMEGA", "MEDIANCUT"]:
                lgtDataset = self.A0["lgtDatasetList"][idx]
                lgtID = int(self.A0["lgtIDList"][idx])
                sample["omegaInput"] = np.tile(self.meta["lgtDatasetCache"].getCache(lgtDataset)["A0"][
                    "omega"
                ][lgtID, :][None, :], (rays.shape[0], 1))  # tile: this is early condition
                sample["lgtType"] = self.meta["lgtDatasetCache"].getCache(lgtDataset)[
                    "A0"
                ]["lgtType"]
            elif self.datasetConf["lgtLoadingMode"] == "QSPL":
                # QSPL
                # quick single point light
                # quick: there is no need to create a lgt dataset if all the lighting is a single point light
                # there is no need to maintain a lgt cache for this type of the lighting
                # nor any need to do R1 rendering of the envmap for this type of the lighting
                lgtE0 = (self.A0["lgtE"][idx] - self.datasetConf["sceneShiftFirst"]) * self.datasetConf["sceneScaleSecond"]
                objCentroid0 = (self.A0["objCentroid"][idx] - self.datasetConf["sceneShiftFirst"]) * self.datasetConf["sceneScaleSecond"]
                B = int(sample["hdrs"].shape[0])
                sample["lgtE"] = np.tile(lgtE0[None, :], (B, 1))
                sample["objCentroid"] = np.tile(objCentroid0[None, :], (B, 1))
                # sample["omegaInput"] = np.tile(omegaInput0[None, :], (B, 1))
                sample["lgtType"] = self.datasetConf["lgtLoadingMode"]
            elif self.datasetConf["lgtLoadingMode"] == "ENVMAP":
                envmap0 = self.meta["lgtDatasetCache"].getExr(
                    self.A0["lgtDatasetList"][idx], int(self.A0["lgtIDList"][idx])
                )
                sample["envmap0"] = envmap0  # Note all the rays follow this single envmap0
                sample["lgtE0"] = (self.A0["lgtE"][idx] - self.datasetConf["sceneShiftFirst"]) * self.datasetConf["sceneScaleSecond"]
                sample["objCentroid0"] = (self.A0["objCentroid"][idx] - self.datasetConf["sceneShiftFirst"]) * self.datasetConf["sceneScaleSecond"]
                sample['lgtType'] = self.datasetConf["lgtLoadingMode"]
            elif self.datasetConf["lgtLoadingMode"] == "ENVMAPLFHYPER":
                envmap_full = self.meta["lgtDatasetCache"].queryEnvmap(
                    self.A0["lgtIDList"][idx:idx + 1],
                    if_augment=False,
                    if_return_all=True,
                )
                sample["envmapOriginalHighres"] = envmap_full["envmap_original_highres"][0, :, :, :].detach().cpu().numpy()
                envmap_to_record = envmap_full["envmap_normalized"][0, :, :, :].float().detach().cpu().numpy()  # (16, 32, 3)
                envmap_alphaed = envmap_to_record  # (16, 32, 3)
                assert envmap_alphaed.shape == (16, 32, 3)
                sample["envmapAlphaed"] = envmap_alphaed
                sample["envmapNormalizingFactor"] = envmap_full["envmap_normalizing_factor"][0].detach().cpu().numpy()
            elif self.datasetConf["lgtLoadingMode"] == "NONE":
                sample["lgtType"] = "NONE"
            else:
                raise NotImplementedError(
                    "Unknown datasetConf['lgtLoadingMode']: %s"
                    % self.datasetConf["lgtLoadingMode"]
                )

        assert "envmap" not in sample.keys()  # use either envmap0 or samples
        return sample

    def update_the_memory_bank_once(self):
        raise ValueError("This method has been disabled")
        # We assume that we just update one image, randomly kick out img_wh[0] * img_wh[1] from the current memory bank
        # If you wish to update two images, call this method twice

        # In the future you can precisly input an index to prescribe which training sample to load       
        # otherwise, we just use the randomly generated number to load

        index_to_load = int(self.indTrain[int(random.randint(0, self.mTrain - 1))])
        
        # ind_to_replace = np.random.permutation(self.m_memory_bank)[:(self.img_wh[0] * self.img_wh[1])]
        ind_to_replace = np.array([
            random.sample(range(self.m_memory_bank), self.img_wh[0] * self.img_wh[1])
        ], dtype=np.int32)

        _, img, valid_mask, hdr, dilated_mask, img_raw, hdr_raw = static_load_data_no_lgt(
            j=0,
            indList=[index_to_load],
            projRoot=self.projRoot,
            nameList=self.A0.get("nameList", None),
            ifLoadImg=self.datasetConf.get("ifLoadImg", True),
            ifLoadHdr=self.datasetConf["ifLoadHdr"],
            flagSplit=self.A0["flagSplit"],
            nameListExr=self.A0.get("nameListExr", None),
            white_back=self.datasetConf["white_back"],
            img_wh=self.img_wh,
            dataset=self.datasetConf["dataset"],
            maskDilateKernel=self.maskDilateKernel,
        )

        self.all_rgbs[ind_to_replace, :] = img
        self.all_masks[ind_to_replace] = valid_mask
        self.all_hdrs[ind_to_replace, :] = hdr
        self.all_dilates[ind_to_replace] = dilated_mask
        self.all_indexID[ind_to_replace] = index_to_load
        self.all_pixelID[ind_to_replace] = np.arange(img.shape[0]).astype(np.int32)

    def external_get_detailed_lgt_info(self, index):
        A0_lgt = self.meta["lgtDatasetCache"].getCache(
            self.A0["lgtDatasetList"][index]
        )["A0"]
        lgtDataset = self.A0["lgtDatasetList"][index]
        lgtID = int(self.A0["lgtIDList"][index])
        lgtType = A0_lgt.get("lgtType", "Unknown")

        with unio.open(
            self.unioRoot + "v/R/%s/R1a/%08d.npz" % (lgtDataset, lgtID),
            self.uconf,
            "rb",
        ) as f:
            hdr = np.load(f)["arr_0"].astype(np.float32)

        detailed_lgt_info = {
            "lgtDataset": lgtDataset,
            "lgtID": lgtID,
            "lgtType": lgtType,
            "lgtHdr": hdr,
        }

        return detailed_lgt_info
