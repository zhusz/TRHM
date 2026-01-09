import os
import pickle
import numpy as np
import math
import torch
import cv2
import time
from collections import OrderedDict
from easydict import EasyDict
from multiprocessing import Process, shared_memory
from .renderingLightStageLfhyper import OnlineAugmentableEnvmapDatasetCache
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb, tonemap_srgb_to_rgb_np
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper


class RenderingNerfBlenderDatasetDemoEnvmap(object):  # It only has testing mode and used for demos
    def __init__(self, datasetConf, **kwargs):
        self.datasetConf = datasetConf
        self.projRoot = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../") + "/"
        self.cudaDevice = kwargs["cudaDevice"]

        assert os.environ.get("RANK", 0) == 0
        assert os.environ.get("WORLD_SIZE", 0) == 0

        split = datasetConf["split"]  # train, val
        singleImageMode = datasetConf["singleImageMode"]
        # assert split in ["test", "alltest"]
        assert singleImageMode

        dataset = datasetConf["dataset"]
        with open(self.projRoot + "v/A/%s/A0_randomness.pkl" % dataset, "rb") as f:
            self.A0 = pickle.load(f)

        self.winWidth = datasetConf["winWidth"]
        self.winHeight = datasetConf["winHeight"]
        assert np.all(self.A0["winWidth"] == self.winWidth)
        assert np.all(self.A0["winHeight"] == self.winHeight)

        # lgtDataset
        self.lgtDatasetCache = OnlineAugmentableEnvmapDatasetCache(
            datasetConf["lgtDataset"], projRoot=self.projRoot,
            quantile_cut_min=datasetConf["lgt_quantile_cut_min"],
            quantile_cut_max=datasetConf["lgt_quantile_cut_max"],
            quantile_cut_fixed=datasetConf["lgt_quantile_cut_fixed"],
            cudaDevice=self.cudaDevice,
            if_lgt_load_highres=datasetConf["if_lgt_load_highres"],
        )

        # cache
        self.meta = {}
        self.meta["pixelMapCoordinates"] = self.setupPixelMapCoordinates()

        self.setupFlagSplit()

    def setupFlagSplit(self):
        # indexing rule:
        #   m = mView (which is A0["m"]) * mLgt (or say, mEnvmap)
        # m is only for one case

        # And also a critical difference compared to the training class below "enderingNerfBlenderDatasetLfhyper"
        #   here the flagSplit allows many 0(invalid), making m precisely equals to A0["m"] * mLgt (rather than extracting out only the valid ones, i.e. flagSplit has no 0.)

        # any training view will make it invalid, any training lgt will make it invalid, otherwise the flagSplit adheres to the view flagSplit
        # This demo class does not contain any flagSplit that equals to 1.

        datasetConf = self.datasetConf
        A0 = self.A0

        indCase_A0 = np.where(A0["caseIDList"] == datasetConf["caseID"])[0]
        mView = indCase_A0.shape[0]
        mLgt = self.lgtDatasetCache.m

        m = mView * mLgt
        flagSplit = np.tile(
            A0["flagSplit"][indCase_A0][:, None],
            (1, mLgt),
        )
        flagSplit[flagSplit == 1] = 0
        flagSplit[:, self.lgtDatasetCache.indTrain] = 0
        flagSplit = flagSplit.reshape((m,))

        self.mView = mView
        self.mLgt = mLgt
        self.m = m
        self.indCase_A0 = indCase_A0

        self.flagSplit = flagSplit
        self.mTrain = 0
        self.indVal = np.where(flagSplit == 2)[0]
        self.mVal =int(self.indVal.shape[0])
        self.indTest = np.where(flagSplit == 3)[0]
        self.mTest = int(self.indTest.shape[0])

    def setupPixelMapCoordinates(self):
        winWidth = self.winWidth
        winHeight = self.winHeight
        xi = np.arange(winWidth).astype(np.int32)
        yi = np.arange(winHeight).astype(np.int32)
        x, y = np.meshgrid(xi, yi)
        x = x.reshape(-1)
        y = y.reshape(-1)
        pixelIDFull = y * winWidth + x
        return {
            "x": x,
            "y": y,
            "pixelIDFull": pixelIDFull,
        }

    def getOneNP(self, index):
        ind_A0 = self.indCase_A0[index // self.mLgt]
        ind_lgt = index % self.mLgt
        flagSplit = self.flagSplit[index]
        # assert flagSplit in [2, 3]
        # assert flagSplit == self.A0["flagSplit"][ind_A0]
        # assert self.lgtDatasetCache.A0["flagSplit"][ind_lgt] > 1

        A0 = self.A0
        lgtDatasetCache = self.lgtDatasetCache
        datasetConf = self.datasetConf
        assert A0["caseIDList"][ind_A0] == datasetConf["caseID"]

        # load the envmap
        envmap_full = lgtDatasetCache.queryEnvmap(
            np.array([ind_lgt], dtype=np.int32),
            if_augment=False,
            if_return_all=True,
        )

        # load the view
        assert A0["winWidth"][ind_A0] == self.winWidth
        assert A0["winHeight"][ind_A0] == self.winHeight
        a = {}
        for k in ["focalLengthWidth", "focalLengthHeight", "winWidth", "winHeight", "E", "L", "U"]:
            a[k] = A0[k][ind_A0]
        w2c0 = ELU02cam0(np.concatenate([a["E"], a["L"], a["U"]], 0))
        c2w0 = np.linalg.inv(w2c0)
        focalLengthWidth, focalLengthHeight = a["focalLengthWidth"], a["focalLengthHeight"]
        winWidth, winHeight = self.winWidth, self.winHeight
        rowID = np.tile(np.arange(winHeight).astype(np.int32)[:, None], (1, winWidth)).reshape(-1)
        colID = np.tile(np.arange(winWidth).astype(np.int32), (winHeight,))
        assert datasetConf["rays_background_pad_width"] == 0
        assert datasetConf["rays_background_pad_height"] == 0
        directions_unnormalized_cam = np.stack([
            (colID.astype(np.float32) + 0.5 - winWidth / 2.0) / focalLengthWidth,
            (rowID.astype(np.float32) + 0.5 - winHeight / 2.0) / focalLengthHeight,
            np.ones((rowID.shape[0],), dtype=np.float32),
        ], 1)
        assert np.all(directions_unnormalized_cam[:, 2] == 1)
        zbuffer_to_euclidean_ratio = np.linalg.norm(directions_unnormalized_cam, ord=2, axis=1)
        directions_unnormalized = (c2w0[None, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
        directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
        directions_normalized = directions_unnormalized / directions_norm[:, None]
        sceneShiftFirst = datasetConf["sceneShiftFirst"]
        sceneScaleSecond = datasetConf["sceneScaleSecond"]
        rays = np.concatenate([
            np.tile(
                (c2w0[None, :3, 3] - sceneShiftFirst[None, :]) * sceneScaleSecond,
                (rowID.shape[0], 1)
            ),  # rays_o
            directions_normalized,  # rays_d
            datasetConf.get("ray_near", np.nan) * np.ones((rowID.shape[0], 1), dtype=np.float32),
            datasetConf.get("ray_far", np.nan) * np.ones((rowID.shape[0], 1), dtype=np.float32),
            np.nan * np.zeros((rowID.shape[0], 2), dtype=np.float32),
            np.zeros((rowID.shape[0], 3), dtype=np.float32),
        ], 1)

        # envmap processing
        envmap_to_record = envmap_full["envmap_normalized"][0, :, :, :].float()  # (16, 32, 3)
        envmap_to_apply = envmap_to_record / 16 / 32 * 3  # normalized by a constant factor, to make its sum to be exactly 1
        envmap_to_apply = envmap_to_apply * 1  # alpha is assumed to be 1
        envmap_to_apply = envmap_to_apply.reshape(16 * 32, 3)
        envmap_alphaed = envmap_to_record.reshape((16, 32, 3))

        # hdrs (not loading at all)
        hdrs = np.zeros((winHeight, winWidth, 3), dtype=np.float32)
        rgbs = np.zeros((winHeight, winWidth, 3), dtype=np.float32)
        valid_mask = np.zeros((winHeight, winWidth), dtype=bool)

        samples = {
            "rays": rays,
            "hdrs": hdrs.reshape((winHeight * winWidth, 3)),
            "rgbs": rgbs.reshape((winHeight * winWidth, 3)),
            "c2w": c2w0,
            "valid_mask": valid_mask.reshape((winHeight * winWidth,)),
            "index": np.array(index, dtype=np.int32),
            "flagSplit": np.array(flagSplit, dtype=np.int32),
            "dataset": datasetConf["dataset"],
            "winWidth": np.array(winWidth, dtype=np.int32),
            "winHeight": np.array(winHeight, dtype=np.int32),
            "lgtID": np.array(ind_lgt, dtype=np.int32),
            "envmapAlphaed": envmap_alphaed.detach().cpu().numpy(),
            "envmapToApply": envmap_to_apply.detach().cpu().numpy(),
            "envmapNormalized": envmap_full["envmap_normalized"][0, :, :, :].detach().cpu().numpy(),
            "envmapNormalizingFactor": envmap_full["envmap_normalizing_factor"][0].detach().cpu().numpy(),
            "envmapSined": envmap_full["envmap_sined"][0, :, :, :].detach().cpu().numpy(),
            "envmapResized": envmap_full["envmap_resized"][0, :, :, :].detach().cpu().numpy(),
            "envmapQuantiled": envmap_full["envmap_quantiled"][0, :, :, :].detach().cpu().numpy(),
            "envmapAugmented": envmap_full["envmap_augmented"][0, :, :, :].detach().cpu().numpy(),
            "envmapOriginalHighres": envmap_full["envmap_original_highres"][0, :, :, :].detach().cpu().numpy(),
            "envmapOriginal": envmap_full["envmap_original"][0, :, :, :].detach().cpu().numpy(),
            "envmapOriginalNormalized": envmap_full["envmap_original_normalized"][0, :, :, :].detach().cpu().numpy(),
            "envmapPointlightOmega": envmap_full["envmap_pointlight_omega"][0, :],
        }

        return samples


def load_nerfBlender_olats(shared_names, shapes, dtypes, startViewID, endViewID, projRoot, dataset, caseID, debugReadHowMany, nOlat, winWidth, winHeight, hyperHdrScaling, verbose):
    
    shmList = [shared_memory.SharedMemory(name=shared_name) for shared_name in shared_names]

    fg_cumsumRayCount = np.ndarray(shapes[0], dtype=dtypes[0], buffer=shmList[0].buf)
    fg_viewID = np.ndarray(shapes[1], dtype=dtypes[1], buffer=shmList[1].buf)
    fg_pixelID = np.ndarray(shapes[2], dtype=dtypes[2], buffer=shmList[2].buf)
    fg_hdrs = np.ndarray(shapes[3], dtype=dtypes[3], buffer=shmList[3].buf)
    indCaseTrain_A0 = np.ndarray(shapes[4], dtype=dtypes[4], buffer=shmList[4].buf)
    flagSplit = np.ndarray(shapes[5], dtype=dtypes[5], buffer=shmList[5].buf)
    caseSplitInsideIndexList = np.ndarray(shapes[6], dtype=dtypes[6], buffer=shmList[6].buf)

    for viewID in range(startViewID, endViewID):
        if verbose:
            print("load_nerfBlender_olats viewID %d" % viewID)
            t0 = time.time()
        ray_start = 0 if viewID == 0 else fg_cumsumRayCount[viewID - 1]
        ray_end = fg_cumsumRayCount[viewID]
        assert np.all(fg_viewID[ray_start:ray_end] == viewID)
        for olatID in range(nOlat):
            # print((viewID, olatID))

            index_A0 = indCaseTrain_A0[olatID * debugReadHowMany + viewID]
            # fn = projRoot + A0["nameListExr"][index_A0]
            assert flagSplit[index_A0] == 1  # split is train
            split = "train"
            fn = projRoot + "v/R/%s/R2exr/%sCase%d/%s/r_%d.exr" % (
                dataset, dataset, caseID, split, caseSplitInsideIndexList[index_A0]
            )

            assert os.path.isfile(fn), fn
            tmp = cv2.imread(fn, flags=cv2.IMREAD_UNCHANGED)
            assert tmp.shape == (winHeight, winWidth, 4)
            current_valid_mask = tmp[:, :, 3] > 1.0e-5
            assert current_valid_mask.sum() == ray_end - ray_start
            fg_hdrs[ray_start:ray_end, olatID, 0] = (tmp[:, :, 2].reshape((winHeight * winWidth,))[fg_pixelID[ray_start:ray_end]]) * hyperHdrScaling
            fg_hdrs[ray_start:ray_end, olatID, 1] = (tmp[:, :, 1].reshape((winHeight * winWidth,))[fg_pixelID[ray_start:ray_end]]) * hyperHdrScaling
            fg_hdrs[ray_start:ray_end, olatID, 2] = (tmp[:, :, 0].reshape((winHeight * winWidth,))[fg_pixelID[ray_start:ray_end]]) * hyperHdrScaling

        if verbose:
            print("    Elapsed Time is %.3fseconds." % (time.time() - t0))

    for x in shmList:
        x.close()


# Notes:
#   This dataset class apply the combination of a view dataset (renderingNerfBlenderXXX) and an lgt dataset to create the sample
#   for datasets(view) that use this dataset class:
#       These datasets' own A0 should organize the indexing in the (16*32lgts)-major and view(e.g. 100/100/200)-minor's way
#       Furthermore, they should have the train/val/test split comes in the strict 
#   for datasets(lgt) it is flexible
class RenderingNerfBlenderDatasetLfhyper(object):
    def __init__(self, datasetConf, **kwargs):
        self.datasetConf = datasetConf
        self.projRoot = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../") + "/"
        self.if_need_metaDataLoading = kwargs["if_need_metaDataLoading"]
        self.cudaDevice = kwargs["cudaDevice"]

        assert os.environ.get("RANK", 0) == 0
        assert os.environ.get("WORLD_SIZE", 0) == 0

        split = datasetConf["split"]  # train, val
        singleImageMode = datasetConf["singleImageMode"]
        if split in ["val", "test"]:
            assert singleImageMode

        self.split = split
        self.singleImageMode = singleImageMode

        dataset = datasetConf["dataset"]
        with open(self.projRoot + "v/A/%s/A0_randomness.pkl" % dataset, "rb") as f:
            self.A0 = pickle.load(f)
            # mTrain is (16 * 32 lgts) * (100 views)
            # mVal is (16 * 32 lgts) * (16 views)

        self.winWidth = datasetConf["winWidth"]
        self.winHeight = datasetConf["winHeight"]

        # lgtDataset
        self.lgtDatasetCache = OnlineAugmentableEnvmapDatasetCache(
            datasetConf["lgtDataset"], projRoot=self.projRoot,
            quantile_cut_min=datasetConf["lgt_quantile_cut_min"],
            quantile_cut_max=datasetConf["lgt_quantile_cut_max"],
            quantile_cut_fixed=datasetConf["lgt_quantile_cut_fixed"],
            cudaDevice=self.cudaDevice,
            if_lgt_load_highres=datasetConf["if_lgt_load_highres"],
        )

        # cache
        self.meta = {}
        self.meta["pixelMapCoordinates"] = self.setupPixelMapCoordinates()

        self.setupFlagSplit()

        # self.generateCamerasAndRays()
        # No need - A0 provides everything

        if (not self.singleImageMode) and (self.if_need_metaDataLoading):
            self.uniqueTrainingGroupViewIDList = self.initializeTheMemoryBank()

    def setupPixelMapCoordinates(self):
        winWidth = self.winWidth
        winHeight = self.winHeight
        xi = np.arange(winWidth).astype(np.int32)
        yi = np.arange(winHeight).astype(np.int32)
        x, y = np.meshgrid(xi, yi)
        x = x.reshape(-1)
        y = y.reshape(-1)
        pixelIDFull = y * winWidth + x
        return {
            "x": x,
            "y": y,
            "pixelIDFull": pixelIDFull,
        }

    def setupFlagSplit(self):
        # indexing rule:
        #   m = mView (which is A0["m"] / (16 * 32)) * mLgt (or say, mEnvmap)
        # For dataloader indexing, it is view-major and envmapID-minor
        # First train, then val, then test.
        #   i.e. [0, mTrain_view * mLgt) is train, [mTrain_view * mLgt, (mTrain_view + mVal_view) * mLgt] is val, etc.

        # For A0
        # In each case&split segment, it is pointlightlgt(16*32)-major and view-minor (This major-minor is good for blender efficient rendering between adjacent samples)

        datasetConf = self.datasetConf
        caseID = int(datasetConf["caseID"])
        A0 = self.A0

        # pick out all the indices that belongs to the current case/split (we only work with one case, but all the splits together)
        nOlat = 16 * 32
        indCaseTrain_A0 = np.where((A0["caseIDList"] == caseID) & (A0["flagSplit"] == 1))[0]
        indCaseVal_A0 = np.where((A0["caseIDList"] == caseID) & (A0["flagSplit"] == 2))[0]
        indCaseTest_A0 = np.where((A0["caseIDList"] == caseID) & (A0["flagSplit"] == 3))[0]
        assert indCaseTrain_A0.shape[0] % (nOlat) == 0
        assert indCaseVal_A0.shape[0] % (nOlat) == 0
        assert indCaseTest_A0.shape[0] % (nOlat) == 0
        mTrain_view = int(indCaseTrain_A0.shape[0] // (nOlat))
        mVal_view = int(indCaseVal_A0.shape[0] // (nOlat))
        mTest_view = int(indCaseTest_A0.shape[0] // (nOlat))

        if datasetConf["debugReadTwo"]:
            debugReadHowMany = int(self.datasetConf.get("debugReadHowMany"))  # this refers to how many views, which has not yet multipled with nOlat
            assert debugReadHowMany <= mTrain_view
            indCaseTrain_A0 = indCaseTrain_A0[:debugReadHowMany][None, :] + (
                np.arange(nOlat, dtype=np.int32)[:, None] * mTrain_view
            )
            indCaseTrain_A0 = indCaseTrain_A0.reshape((debugReadHowMany * nOlat,))
            # Note mTrain_view remains the same

        indCase_A0 = np.concatenate([indCaseTrain_A0, indCaseVal_A0, indCaseTest_A0], 0)
        assert np.all(A0["winWidth"][indCase_A0] == self.winWidth)
        assert np.all(A0["winHeight"][indCase_A0] == self.winHeight)
        
        self.indCaseTrain_A0 = indCaseTrain_A0.astype(np.int32)
        self.indCaseVal_A0 = indCaseVal_A0.astype(np.int32)
        self.indCaseTest_A0 = indCaseTest_A0.astype(np.int32)
        self.mTrain_view = mTrain_view
        self.mVal_view = mVal_view
        self.mTest_view = mTest_view

        self.indCaseTrain_lgt = np.where(self.lgtDatasetCache.A0["flagSplit"] == 1)[0]
        self.mTrain_lgt = int(self.indCaseTrain_lgt.shape[0])
        self.indCaseVal_lgt = np.where(self.lgtDatasetCache.A0["flagSplit"] == 2)[0]
        self.mVal_lgt = int(self.indCaseVal_lgt.shape[0])
        self.indCaseTest_lgt = np.where(self.lgtDatasetCache.A0["flagSplit"] == 3)[0]
        self.mTest_lgt = int(self.indCaseTest_lgt.shape[0])

        def makeFlagSplitSingleSplit(m_view, m_lgt, flagSplitScalar):
            for x in [m_view, m_lgt, flagSplitScalar]:
                assert type(x) == int
            flagSplitSingleSplit = flagSplitScalar * np.ones((m_view * m_lgt,), dtype=np.int32)
            return flagSplitSingleSplit

        self.flagSplit = np.concatenate([
            makeFlagSplitSingleSplit(self.mTrain_view, self.mTrain_lgt, 1),
            makeFlagSplitSingleSplit(
                self.mVal_view,
                (self.mVal_lgt + self.mTest_lgt) if (self.lgtDatasetCache.A0["flagSplit"].max() == 3) else self.mVal_lgt,
                2,
            ),
            makeFlagSplitSingleSplit(self.mTest_view, self.mTest_lgt, 3),
        ], 0)

    def initializeTheMemoryBank(self):
        projRoot = self.projRoot
        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        caseID = int(datasetConf["caseID"])
        indCaseTrain_A0 = self.indCaseTrain_A0
        A0 = self.A0
        flagSplit = A0["flagSplit"]
        caseSplitInsideIndexList = A0["caseSplitInsideIndexList"]

        winWidth, winHeight = self.winWidth, self.winHeight

        nOlat = 16 * 32
        nView = self.mTrain_view  # pre-loading only happens with the training split
        if datasetConf["debugReadTwo"]:
            debugReadHowMany = datasetConf["debugReadHowMany"]
        else:
            debugReadHowMany = self.mTrain_view

        # pre-evaluating the total memory cost
        bg_indexID = []
        bg_pixelID = []
        fg_indexID = []
        fg_pixelID = []
        fg_rayCount = np.zeros((debugReadHowMany,), dtype=np.int32)
        for viewID in range(debugReadHowMany):
            print("(pre-evaluating the total memory cost) Preloading initial memory bank (class: %s, dataset: %s) viewID %d" % (
                self.__class__.__name__, dataset, viewID
            ))
            olatID = 0
            index_A0 = indCaseTrain_A0[olatID * nView + viewID]
            fn = projRoot + A0["nameListExr"][index_A0]
            assert os.path.isfile(fn), fn
            tmp = cv2.imread(fn, flags=cv2.IMREAD_UNCHANGED)
            assert tmp.shape == (winHeight, winWidth, 4)
            valid_mask = (tmp[:, :, 3] > 1.0e-5).reshape((winWidth * winHeight,))
            
            current_bg_pixelID = np.where(valid_mask == 0)[0].astype(np.int32)
            current_bg_indexID = viewID * np.ones((current_bg_pixelID.shape[0],), dtype=np.int32)
            current_fg_pixelID = np.where(valid_mask)[0].astype(np.int32)
            current_fg_indexID = viewID * np.ones((current_fg_pixelID.shape[0],), dtype=np.int32)

            fg_rayCount[viewID] = current_fg_pixelID.shape[0]

            bg_indexID.append(current_bg_indexID)
            bg_pixelID.append(current_bg_pixelID)
            fg_indexID.append(current_fg_indexID)
            fg_pixelID.append(current_fg_pixelID)

        bg_indexID = np.concatenate(bg_indexID, 0)
        bg_pixelID = np.concatenate(bg_pixelID, 0)
        fg_indexID = np.concatenate(fg_indexID, 0)
        fg_pixelID = np.concatenate(fg_pixelID, 0)
        fg_cumsumRayCount = np.cumsum(fg_rayCount, axis=0, dtype=np.int32)
        fg_rays_tot = int(fg_cumsumRayCount[-1])

        shapes = [
            (debugReadHowMany,),  # fg_cumsumRayCount  # values already set
            (fg_rays_tot,),  # fg_indexID  # values already set
            (fg_rays_tot,),  # fg_pixelID  # values already set
            (fg_rays_tot, nOlat, 3),  # fg_hdrs  # to get the values
            (indCaseTrain_A0.shape[0],),  # indCaseTrain_A0 # values already set
            (flagSplit.shape[0],),  # flagSplit
            (caseSplitInsideIndexList.shape[0],),  # caseSplitInsideIndexList
        ]
        dtypes = [np.int32] * 3 + [np.float32] + [np.int32] * 3
        assert len(shapes) == len(dtypes)
        shared_memories = []
        count = 0
        for shape, dtype in zip(shapes, dtypes):
            # size = np.zeros(shape, dtype=dtype).nbytes
            size = math.prod(shape) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=size)
            shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            print((count, shape, dtype, size))
            if count == 0:  # fg_cumsumRayCount
                assert shared_array.shape == fg_cumsumRayCount.shape
                assert shared_array.dtype == fg_cumsumRayCount.dtype
                np.copyto(shared_array, fg_cumsumRayCount)
            elif count == 1:  # fg_indexID
                assert shared_array.shape == fg_indexID.shape
                assert shared_array.dtype == fg_indexID.dtype
                np.copyto(shared_array, fg_indexID)
            elif count == 2:  # fg_pixelID
                assert shared_array.shape == fg_pixelID.shape
                assert shared_array.dtype == fg_pixelID.dtype
                np.copyto(shared_array, fg_pixelID)
            elif count == 3:  # fg_hdrs, "output"
                # shared_memories[-1][1].fill(0)
                shared_array.fill(0)
            elif count == 4:  # indCaseTrain_A0
                assert shared_array.shape == indCaseTrain_A0.shape
                assert shared_array.dtype == indCaseTrain_A0.dtype
                np.copyto(shared_array, indCaseTrain_A0)
            elif count == 5:  # flagSplit
                assert shared_array.shape == flagSplit.shape
                assert shared_array.dtype == flagSplit.dtype
                np.copyto(shared_array, flagSplit)
            elif count == 6:  # caseSplitInsideIndexList
                assert shared_array.shape == caseSplitInsideIndexList.shape
                assert shared_array.dtype == caseSplitInsideIndexList.dtype
                np.copyto(shared_array, caseSplitInsideIndexList)
            else:
                raise ValueError("Invalid count: %d" % count)
            shared_memories.append((shm, shared_array))
            count += 1
        shared_names = [shm.name for shm, _ in shared_memories]
        num_processes = min(50, debugReadHowMany)
        chunk = int(math.ceil(float(debugReadHowMany) / num_processes))
        processes = []

        processes = []
        for chunkID in range(num_processes):
            startViewID = chunkID * chunk
            endViewID = min((chunkID + 1) * chunk, nView)
            p = Process(
                target=load_nerfBlender_olats,
                args=(
                    shared_names, shapes, dtypes, startViewID, endViewID, projRoot, dataset, caseID, debugReadHowMany, nOlat, winWidth, winHeight, datasetConf["hyperHdrScaling"], True
                ),
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        
        self.all_bg_indexID = bg_indexID  # view only, has not yet multiplied with nOlat
        self.all_bg_pixelID = bg_pixelID
        self.all_fg_indexID = fg_indexID  # view only, has not yet multiplied with nOlat
        self.all_fg_pixelID = fg_pixelID
        self.shared_memories = shared_memories  # So the buffer would persist even when getting out of this scope
        self.shared_shapes = shapes
        self.shared_dtypes = dtypes
 
    def getOneNP(self, index):
        datasetConf = self.datasetConf
        caseID = int(datasetConf["caseID"])
        lgtDatasetCache = self.lgtDatasetCache
        cudaDevice = self.cudaDevice

        indCaseTrain_A0 = self.indCaseTrain_A0
        indCaseVal_A0 = self.indCaseVal_A0
        indCaseTest_A0 = self.indCaseTest_A0
        mTrain_view = self.mTrain_view
        mVal_view = self.mVal_view
        mTest_view = self.mTest_view
        mTrain_lgt = self.mTrain_lgt
        mVal_lgt = self.mVal_lgt
        mTest_lgt = self.mTest_lgt
        nOlat = 16 * 32
        mLgt = self.lgtDatasetCache.m
        A0 = self.A0
        
        mTrain = mTrain_view * mTrain_lgt
        if self.lgtDatasetCache.A0["flagSplit"].max() == 3:  # demo mode
            mVal = mVal_view * (mVal_lgt + mTest_lgt)
        else:
            mVal = mVal_view * mVal_lgt
        mTest = mTest_view * mTest_lgt
        if 0 <= index < mTrain:
            index_split = index - 0
            index_view_split = index_split // mTrain_lgt
            index_lgt_split = index_split % mTrain_lgt
            ind_A0 = np.arange(nOlat).astype(np.int32) * mTrain_view + index_view_split + nOlat * (mTrain_view + mVal_view + mTest_view) * caseID
            assert np.all(A0["flagSplit"][ind_A0] == 1)
            ind_lgt = self.lgtDatasetCache.indTrain[index_lgt_split]
            assert self.lgtDatasetCache.A0["flagSplit"][ind_lgt] == 1 
            flagSplit = 1
        elif mTrain <= index < mTrain + mVal:
            index_split = index - mTrain
            if self.lgtDatasetCache.A0["flagSplit"].max() == 3:  # demo mdoe
                index_view_split = index_split // (mVal_lgt + mTest_lgt)
                index_lgt_split = index_split % (mVal_lgt + mTest_lgt)
                ind_A0 = np.arange(nOlat).astype(np.int32) * mVal_view + index_view_split + nOlat * mTrain_view + nOlat * (mTrain_view + mVal_view + mTest_view) * caseID
                assert np.all(A0["flagSplit"][ind_A0] == 2)
                if index_lgt_split < mVal_lgt:
                    ind_lgt = self.lgtDatasetCache.indVal[index_lgt_split]
                    assert self.lgtDatasetCache.A0["flagSplit"][ind_lgt] == 2
                else:
                    ind_lgt = self.lgtDatasetCache.indTest[index_lgt_split - mVal_lgt]
                    assert self.lgtDatasetCache.A0["flagSplit"][ind_lgt] == 3
            else:
                index_view_split = index_split // mVal_lgt
                index_lgt_split = index_split % mVal_lgt
                ind_A0 = np.arange(nOlat).astype(np.int32) * mVal_view + index_view_split + nOlat * mTrain_view + nOlat * (mTrain_view + mVal_view + mTest_view) * caseID
                assert np.all(A0["flagSplit"][ind_A0] == 2)
                ind_lgt = self.lgtDatasetCache.indVal[index_lgt_split]
                assert self.lgtDatasetCache.A0["flagSplit"][ind_lgt] == 2
            flagSplit = 2
        elif mTrain + mVal <= index < mTrain + mVal + mTest:
            index_split = index - mTrain - mVal
            index_view_split = index_split // mTest_lgt
            index_lgt_split = index_split % mTest_lgt
            ind_A0 = np.arange(nOlat).astype(np.int32) * mTest_view + index_view_split + nOlat * (mTrain_view + mVal_view)
            assert np.all(A0["flagSplit"][ind_A0] == 3)
            ind_lgt = self.lgtDatasetCache.indTest[index_lgt_split]
            assert self.lgtDatasetCache.A0["flagSplit"][ind_lgt] == 3
            flagSplit = 3
        else:
            raise ValueError("Invalid ind_A0: (%d, %d, %d, %d, %d, %d, %d)" % (index, mTrain_view, mVal_view, mTest_view, mTrain_lgt, mVal_lgt, mTest_lgt))

        assert np.all(A0["caseIDList"][ind_A0] == datasetConf["caseID"])
        
        # load the envmap
        envmap_full = lgtDatasetCache.queryEnvmap(
            np.array([ind_lgt], dtype=np.int32),
            if_augment=False,
            if_return_all=True,
        )

        # load the view
        a = {}
        for k in ["focalLengthWidth", "focalLengthHeight", "winWidth", "winHeight", "E", "L", "U"]:
            x = A0[k][ind_A0]
            assert np.all(x.min(0) == x.max(0))
            a[k] = x[0]
        assert a["winWidth"] == self.winWidth
        assert a["winHeight"] == self.winHeight
        w2c0 = ELU02cam0(np.concatenate([a["E"], a["L"], a["U"]], 0))
        c2w0 = np.linalg.inv(w2c0)
        focalLengthWidth, focalLengthHeight = a["focalLengthWidth"], a["focalLengthHeight"]
        winWidth, winHeight = self.winWidth, self.winHeight
        rowID = np.tile(np.arange(winHeight).astype(np.int32)[:, None], (1, winWidth)).reshape(-1)
        colID = np.tile(np.arange(winWidth).astype(np.int32), (winHeight,))
        assert datasetConf["rays_background_pad_width"] == 0
        assert datasetConf["rays_background_pad_height"] == 0
        directions_unnormalized_cam = np.stack([
            (colID.astype(np.float32) + 0.5 - winWidth / 2.0) / focalLengthWidth,
            (rowID.astype(np.float32) + 0.5 - winHeight / 2.0) / focalLengthHeight,
            np.ones((rowID.shape[0],), dtype=np.float32),
        ], 1)
        assert np.all(directions_unnormalized_cam[:, 2] == 1)
        zbuffer_to_euclidean_ratio = np.linalg.norm(directions_unnormalized_cam, ord=2, axis=1)
        directions_unnormalized = (c2w0[None, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
        directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
        directions_normalized = directions_unnormalized / directions_norm[:, None]
        sceneShiftFirst = datasetConf["sceneShiftFirst"]
        sceneScaleSecond = datasetConf["sceneScaleSecond"]
        rays = np.concatenate([
            np.tile(
                (c2w0[None, :3, 3] - sceneShiftFirst[None, :]) * sceneScaleSecond,
                (rowID.shape[0], 1)
            ),  # rays_o
            directions_normalized,  # rays_d
            datasetConf.get("ray_near", np.nan) * np.ones((rowID.shape[0], 1), dtype=np.float32),
            datasetConf.get("ray_far", np.nan) * np.ones((rowID.shape[0], 1), dtype=np.float32),
            np.nan * np.zeros((rowID.shape[0], 2), dtype=np.float32),
            np.zeros((rowID.shape[0], 3), dtype=np.float32),
        ], 1)

        # load the hdr (if required)
        if datasetConf["ifLoadHdr"]:
            hdrOlats = np.zeros((nOlat, winHeight, winWidth, 3), dtype=np.float32)
            valid_mask = None
            for i, j_A0 in enumerate(ind_A0.tolist()):
                fn_A0 = self.projRoot + A0["nameListExr"][j_A0]
                assert os.path.isfile(fn_A0), fn_A0
                bgra0 = cv2.imread(fn_A0, flags=cv2.IMREAD_UNCHANGED)
                assert bgra0.shape == (winHeight, winWidth, 4)
                rgb0 = np.stack([bgra0[:, :, 2], bgra0[:, :, 1], bgra0[:, :, 0]], 2)
                hdrOlats[i, :, :, :] = rgb0 * datasetConf["hyperHdrScaling"]
                if valid_mask is None:
                    valid_mask = bgra0[:, :, 3] > 1.0e-5
                else:
                    assert np.all(valid_mask == (bgra0[:, :, 3] > 1.0e-5))
            hdrOlats_thgpu = torch.from_numpy(hdrOlats).to(cudaDevice)

            envmap_to_record = envmap_full["envmap_normalized"][0, :, :, :].float()  # (16, 32, 3)
            envmap_to_apply = envmap_to_record / 16 / 32 * 3  # normalized by a constant factor, to make its sum to be exactly 1
            envmap_to_apply = envmap_to_apply * 1  # alpha is assumed to be 1
            envmap_to_apply = envmap_to_apply.reshape(16 * 32, 3)
            envmap_alphaed = envmap_to_record.reshape((16, 32, 3))
            
            hdrOlats_thgpu = torch.from_numpy(hdrOlats).to(cudaDevice)
            hdrs_thgpu = (hdrOlats_thgpu * envmap_to_apply[:, None, None, :]).sum(0).detach() * datasetConf["hyperHdrScaling"]
            rgbs_thgpu = torch.clamp(tonemap_srgb_to_rgb(hdrs_thgpu), min=0, max=1)
            hdrs = hdrs_thgpu.cpu().numpy()
            rgbs = rgbs_thgpu.cpu().numpy()

        else:
            hdrs = np.zeros((winHeight, winWidth, 3), dtype=np.float32)
            rgbs = np.zeros((winHeight, winWidth, 3), dtype=np.float32)
            valid_mask = np.zeros((winHeight, winWidth), dtype=bool)

        samples = {
            "rays": rays,
            "hdrs": hdrs.reshape((winHeight * winWidth, 3)),
            "rgbs": rgbs.reshape((winHeight * winWidth, 3)),
            "c2w": c2w0,
            "valid_mask": valid_mask.reshape((winHeight * winWidth,)),
            "index": np.array(index, dtype=np.int32),
            "flagSplit": np.array(flagSplit, dtype=np.int32),
            "dataset": datasetConf["dataset"],
            "winWidth": np.array(winWidth, dtype=np.int32),
            "winHeight": np.array(winHeight, dtype=np.int32),
            "lgtID": np.array(ind_lgt, dtype=np.int32),
            "envmapAlphaed": envmap_alphaed.detach().cpu().numpy(),
            "envmapToApply": envmap_to_apply.detach().cpu().numpy(),
            "envmapNormalized": envmap_full["envmap_normalized"][0, :, :, :].detach().cpu().numpy(),
            "envmapNormalizingFactor": envmap_full["envmap_normalizing_factor"][0].detach().cpu().numpy(),
            "envmapSined": envmap_full["envmap_sined"][0, :, :, :].detach().cpu().numpy(),
            "envmapResized": envmap_full["envmap_resized"][0, :, :, :].detach().cpu().numpy(),
            "envmapQuantiled": envmap_full["envmap_quantiled"][0, :, :, :].detach().cpu().numpy(),
            "envmapAugmented": envmap_full["envmap_augmented"][0, :, :, :].detach().cpu().numpy(),
            "envmapOriginalHighres": envmap_full["envmap_original_highres"][0, :, :, :].detach().cpu().numpy(),
            "envmapOriginal": envmap_full["envmap_original"][0, :, :, :].detach().cpu().numpy(),
            "envmapOriginalNormalized": envmap_full["envmap_original_normalized"][0, :, :, :].detach().cpu().numpy(),
        }
        return samples
