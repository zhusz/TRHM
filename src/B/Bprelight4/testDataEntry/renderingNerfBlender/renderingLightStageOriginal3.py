# Conventions about when to do sceneShiftFirst and sceneScaleSecond (apply to both synthetic and real datasets)
#   - training time preloaded meta data, such as datasetObj.A0["E"] or datasetObj.A0["lgtE"] are storing the raw data (not-yet-scaled), and their scaling time happens during trainer.batchPreProcesingTHGPU()
#       - Exception: highlight preloaded data has already incorporated the shift/scale transforms before pre-storing, as it requires highlightRays being produced for pre-storing
#   - training time + test time zbufferDepth: zbufferDepth is the only R-type dataset component (R-type: one file per sample, in constrast to A-type: meta data), and its rescaling happends immediately after it is being loaded from the file, regardless of whether it is training time or test time
#   - test time bsv0: the returned bsv0 contains all the data that is already rescaled. This applies to both E/lgtE and zbufferDepth.
# One more thing: zbufferDepth is only processed by sceneScaleSecond, but not processed by sceneShiftFirst, because the camera is also moving along with sceneShiftFirst

import os
import pickle
import numpy as np
import math
import cv2
from .utils import map_olatIDList_to_frameIDList
projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../"
import sys
sys.path.append(projRoot + "src/versions/")
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from codes_py.np_ext.data_processing_utils_v1 import normalize


class ObjCentroidCache(object):
    def __init__(self):
        # self.dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
        self.dataset = "capture7jf"
        self.cache = {}

    def getCache(self, sceneID):
        if sceneID not in self.cache.keys():
            v0, f0 = load_obj_np(projRoot + "v/R/%s/R0obj/%08d.obj" % (self.dataset, sceneID))
            objCentroid0 = (v0.min(0) + v0.max(0)) / 2.0
            self.cache[sceneID] = objCentroid0
        return self.cache[sceneID]


class EnvmapLgtDatasetCache(object):
    def __init__(self):
        self.cache = {}  # self.cache[dataset]["A0"]

    @staticmethod
    def getLgtA0(lgtDataset):
        lgtA0_fn = projRoot + "v/A/%s/A0_main.pkl" % (lgtDataset)
        assert os.path.isfile(lgtA0_fn), lgtA0_fn
        with open(lgtA0_fn, "rb") as f:
            A0_lgt = pickle.load(f)
        return A0_lgt

    def getCache(self, lgtDataset):
        if lgtDataset not in self.cache.keys():
            A0_lgt = self.getLgtA0(lgtDataset)
            self.cache[lgtDataset] = {
                "A0": A0_lgt,
            }
        return self.cache[lgtDataset]

    def readEnvmap(self, lgtDataset, index):  # do resize outside
        A0_lgt = self.getCache(lgtDataset)["A0"]
        fn = projRoot + A0_lgt["nameList"][index]
        assert os.path.isfile(fn), fn
        tmp = cv2.imread(fn, flags=cv2.IMREAD_UNCHANGED)
        assert tmp.dtype == np.float32
        rgb = np.stack([tmp[:, :, 2], tmp[:, :, 1], tmp[:, :, 0]], 2)
        return rgb

    def ensureHotCache(self, lgtDataset, indList, verbose=True):
        A0 = self.getCache(lgtDataset)["A0"]
        if "envmaps" not in self.cache[lgtDataset].keys():
            m = int(A0["m"])
            height = int(A0["winHeight"])
            width = int(A0["winWidth"])
            envmaps = np.zeros((m, height, width, 3), dtype=np.float32)
            self.cache[lgtDataset]["envmaps"] = envmaps
            flagAlreadyRead = np.zeros((m,), dtype=bool)
            self.cache[lgtDataset]["flagAlreadyRead"] = flagAlreadyRead
        else:
            envmaps = self.cache[lgtDataset]["envmaps"]
            flagAlreadyRead = self.cache[lgtDataset]["flagAlreadyRead"]
        nameList = A0["nameList"]
        for j in indList.tolist():
            if flagAlreadyRead[j] == False:
                fn = projRoot + nameList[j]
                assert os.path.isfile(fn), fn
                print("    [lgtEnvmapDataset Cache] Loading from %s" % fn)
                envmaps[j, :, :, :] = self.readEnvmap(lgtDataset, j)
                flagAlreadyRead[j] = True


class RenderingLightStageOriginal3(object):
    def __init__(self, datasetConf, **kwargs):
        self.datasetConf = datasetConf

        self.rank = int(os.environ.get("RANK", 0))
        self.numMpProcess = int(os.environ.get("WORLD_SIZE", 0))

        split = datasetConf["split"]  # train, val
        singleImageMode = datasetConf["singleImageMode"]
        if split in ["val", "test"]:
            assert singleImageMode

        self.split = split
        self.singleImageMode = singleImageMode
        self.ifPreloadHighlight = datasetConf.get("ifPreloadHighlight", False)

        # preset value
        self.winWidth = 1366
        self.winHeight = 2048

        # default from datasetConf
        pass

        # At the moment we assume a fixed bucket "nerf_dataset".
        self.projRoot = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../") + "/"

        # A0 and A0b
        dataset = datasetConf["dataset"]
        with open(self.projRoot + "v/A/%s/A0_fromBlender.pkl" % "capture7jf", "rb") as f:
            self.A0 = pickle.load(f)
        with open(self.projRoot + "v/A/%s/A0b_precomputation.pkl" % "capture7jf", "rb") as f:
            self.A0b = pickle.load(f)

        # cache
        self.meta = {}
        self.objCentroidCache = ObjCentroidCache()
        self.envmapLgtDatasetCache = EnvmapLgtDatasetCache()

        self.meta["pixelMapCoordinates"] = self.setupPixelMapCoordinates()

        self.setupFlagSplit()

        self.generateCamerasAndRays()

        if not self.singleImageMode:
            if self.ifPreloadHighlight:
                self.preload_highlight()
            self.preload_original_buckets()

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

    def furtherReadjustFlagSplit(self):  # Can be overrided to fit your particular need (e.g. overfit to a subset)
        pass

    def setupFlagSplit(self):
        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        bucket = datasetConf["bucket"]
        caseID = datasetConf["caseID"]
        sceneID = caseID
        if bucket == "nerf":
            flagSplit = np.zeros((180,), dtype=np.int32)  # nerf only has one lighting, 180 images
            for j in range(180):
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if flagRemove:
                    flagSplit[j] = 0
                else:
                    flagSplit[j] = 1 if viewID <= 18 else 2
            self.flagSplit = flagSplit
        elif bucket.startswith("lfdistill"):  # lens-flare-corrupted frames testing
            assert bucket == "lfdistill"
            caseID = datasetConf["caseID"]
            R0q_fn = self.projRoot + "v/R/capture7jf/R0q/%08d.pkl" % caseID
            assert os.path.isfile(R0q_fn), R0q_fn
            with open(R0q_fn, "rb") as f:
                q_all = pickle.load(f)["q_all"]  # (9, 20, 331)
            m = 9 * 20 * 331
            flagSplit = (q_all.reshape(m,) > 0.005).astype(np.int32) * 3  # 0: not lens-flare-corrupted, and hence no need to run. 3: need to test because of lens-flare-corrupted
            for j in range(180):
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if flagRemove:
                    flagSplit[j * 331:(j + 1) * 331] = 0
            self.nOlat = 331
            self.flagSplit = flagSplit
        # elif bucket.startswith("olat") and (not bucket.startswith("olatFull")):
        # elif bucket.startswith("ol3t") and (not bucket.startswith("ol3tFree")):
        elif bucket.startswith("distill"):  # indexing is ranged in [0, 92610) (92610 is 180 * 512) for distill512
            nOlat = int(bucket[len("distill"):])
            # flagSplit = 3 * np.ones((180 * nOlat), dtype=np.int32)
            flagSplit = np.tile(np.array([1] * 18 + [2] * 2, dtype=np.int32)[None, :, None], (9, 1, nOlat)).reshape((180 * nOlat))
            self.nOlat = nOlat
            self.flagSplit = flagSplit
        elif bucket.startswith("envmap"):
            if bucket == "envmapLavalAug2":
                nEnvmap = 80
            else:
                raise NotImplemented("Unknown bucket: %s" % bucket)
            self.nOlat = nEnvmap  # for code compatibility, although it is envmap, we use self.nOlat. There is no ambiguity here.
            flagSplit = np.zeros((180 * nEnvmap,), dtype=np.int32)
            for j in range(180):  # 180-major, cc(olat)-minor
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if flagRemove:
                    flagSplit[j * nEnvmap : (j + 1) * nEnvmap] = 0
                else:
                    flagSplit[j * nEnvmap : (j + 1) * nEnvmap] = 1 if viewID <= 18 else (3 if bucket in ["ol3tFree3"] else 2)
            self.flagSplit = flagSplit
        elif (bucket.startswith("ol3t")) and ("Envmap" not in bucket):
            if bucket == "ol3tFree3":
                olatIDchosen = np.array([43, 73, 105, 137, 169, 203, 233, 265, 297, 329], dtype=np.int32)  # 1-based
                nOlat = int(olatIDchosen.shape[0])
                assert nOlat == 10
            elif bucket == "ol3t75":
                nOlat = 75
                olatIDchosen = np.linspace(1, 331, nOlat).astype(np.int32)
            elif bucket == "ol3t331":
                nOlat = 331
                olatIDchosen = np.arange(331).astype(np.int32)
            else:
                raise NotImplementedError("Unknown bucket: %s" % bucket)
                # nOlat = int(bucket[4:])
            self.nOlat = nOlat
            self.olatIDchosen = olatIDchosen
            flagSplit = np.zeros((180 * nOlat,), dtype=np.int32)
            for j in range(180):  # 180-major, cc(olat)-minor
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if flagRemove:
                    flagSplit[j * nOlat : (j + 1) * nOlat] = 0
                else:
                    flagSplit[j * nOlat : (j + 1) * nOlat] = 1 if viewID <= 18 else (3 if bucket in ["ol3tFree3"] else 2)
            self.flagSplit = flagSplit
        elif (bucket.startswith("ol3t")) and ("Envmap" in bucket):
            # the total # of samples becomes: 180 * nOlat * nEnvmap
            # It still loads from bucket_ol3tXXX
            # the returned olat lighting and the envmap lighting does not have any relation with each other - just for implementation-wise convenience
            # If you do not use the olat lighting, then the indexing is nOlat-time redundent
            # If you do not use the envmap lighting, then the indexing is nEnvmap-time redundent (and probably you should instead use the bucket=="ol3tXXX" case above)
            if bucket.startswith("ol3tFree3Envmap"):
                # olatIDchosen = np.array([43, 73, 105, 137, 169, 203, 233, 265, 297, 329], dtype=np.int32)  # 1-based
                # nOlat = int(olatIDchosen.shape[0])
                # assert nOlat == 10
                olatIDchosen = np.array([43], dtype=np.int32)
                nOlat = int(olatIDchosen.shape[0])
            else:
                raise NotImplementedError("Unknown bucket: %s" % bucket)
            self.nOlat = nOlat
            self.olatIDchosen = olatIDchosen
            # envmap from a lgtDataset
            def getLgtDataset(bucket):
                id_left = bucket.find("Envmap")
                id_right = bucket.rfind("Envmap")
                assert id_left == id_right
                t = bucket[id_left + len("envmap"):]
                return t[0].lower() + t[1:]
            lgtDataset = getLgtDataset(bucket)

            # A0_lgt = getLgtA0(lgtDataset)
            A0_lgt = self.envmapLgtDatasetCache.getCache(lgtDataset)["A0"]
            nEnvmap = int(A0_lgt["m"])
            self.nEnvmap = nEnvmap
            self.envmapLgtDataset = lgtDataset
            flagSplit = np.zeros((180 * nOlat * nEnvmap,), dtype=np.int32)
            for j in range(180):  # 180-major, cc(olat)-minor
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if flagRemove:
                    flagSplit[j * nOlat * nEnvmap : (j + 1) * nOlat * nEnvmap] = 0
                else:
                    flagSplit[j * nOlat * nEnvmap : (j + 1) * nOlat * nEnvmap] = 1 if viewID <= 18 else 2  # (3 if bucket.startswith("ol3tFree3") else 2)
            self.flagSplit = flagSplit

        else:
            raise NotImplementedError("Unknown bucket: %s" % bucket)

        self.furtherReadjustFlagSplit()

        # indTrain only use the part (of the current gpus) for multi-GPU training settings
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
        elif self.datasetConf.get("debugReadTwo", False):
            self.flagSplit[
                (np.arange(self.flagSplit.shape[0]) // self.nOlat >= self.datasetConf.get("debugReadTwoHowMany", 2))
                & (self.flagSplit == 1)
            ] = 0
        elif self.datasetConf.get("debugReadEvery", False):
            raise NotImplementedError("TODO here when you need it")
        self.indTrain = np.where(self.flagSplit == 1)[0]

        # Other commons
        self.indVal = np.where(self.flagSplit == 2)[0]
        self.indTest = np.where(self.flagSplit == 3)[0]
        self.ind = np.where(self.flagSplit > 0)[0]
        self.mTrain = (self.flagSplit == 1).sum()
        self.mVal = (self.flagSplit == 2).sum()
        self.mTest = (self.flagSplit == 3).sum()
        assert len(self.ind) > 0

    def generateCamerasAndRays(self):
        datasetConf = self.datasetConf
        caseID = datasetConf["caseID"]
        sceneID = caseID
        winWidth = self.winWidth
        winHeight = self.winHeight

        self.E = np.nan * np.zeros((180, 3), dtype=np.float32)
        self.L = np.nan * np.zeros((180, 3), dtype=np.float32)
        self.U = np.nan * np.zeros((180, 3), dtype=np.float32)
        self.c2w = np.nan * np.zeros((180, 4, 4), dtype=np.float32)
        self.focalLengthWidth = np.nan * np.zeros((180,), dtype=np.float32)
        self.focalLengthHeight = np.nan * np.zeros((180,), dtype=np.float32)
        for groupID in range(9):
            for viewID in range(1, 1 + 20):
                j = groupID * 20 + viewID - 1
                # if self.flagSplit[j] > 0:  # This is nerf condition
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if not flagRemove:
                    cameraInfo = self.A0b[sceneID]["leaf_camera_collectors"][groupID][viewID]
                    E0 = cameraInfo["EWorld0"]
                    L0 = cameraInfo["LWorld0"]
                    U0 = cameraInfo["UWorld0"]
                    w2c0 = ELU02cam0(np.concatenate([E0, L0, U0], 0))
                    c2w0 = np.linalg.inv(w2c0)
                    self.E[j, :] = E0
                    self.L[j, :] = L0
                    self.U[j, :] = U0
                    self.c2w[j, :, :] = c2w0
                    self.focalLengthWidth[j] = cameraInfo["lens"] / cameraInfo["sensor_width"] * winWidth
                    self.focalLengthHeight[j] = cameraInfo["lens"] / cameraInfo["sensor_height"] * winHeight

    def preload_highlight(self):
        datasetConf = self.datasetConf
        caseID = datasetConf["caseID"]
        sceneID = caseID
        sceneTag = self.A0[sceneID]["sceneTag"]
        x = self.meta["pixelMapCoordinates"]["x"]
        y = self.meta["pixelMapCoordinates"]["y"]
        pixelIDFull = self.meta["pixelMapCoordinates"]["pixelIDFull"]
        winWidth = self.winWidth
        winHeight = self.winHeight
        sceneShiftFirst = datasetConf["sceneShiftFirst"]
        sceneScaleSecond = datasetConf["sceneScaleSecond"]

        # for groupViewID in range(180):
        record_IDs_highlight = []
        record_hdr_highlight = []
        record_rays_highlight = []
        record_lgtE_highlight = []
        # record_tag_highlight = []  # bool: True for highlight, and False for sublight
        record_IDs_sublight = []
        record_hdr_sublight = []
        record_rays_sublight = []
        record_lgtE_sublight = []
        for groupID in range(9):
            lgtE_of_the_group = np.stack([
                self.A0b[sceneID]["leaf_lgt_collectors"][groupID][o]["EWorld0"]
                for o in range(1, 1 + 331)
            ], 0).astype(np.float32)
            for viewID in range(1, 1 + 20):
                groupViewID = groupID * 20 + viewID - 1
                if not np.any(self.flagSplit[
                    (groupID * 20 + viewID - 1) * self.nOlat + 0 : (groupID * 20 + viewID - 1) * self.nOlat + self.nOlat
                ] == 1):
                    continue
                print("Preloading highlight: %d / %d" % (groupViewID, 180))
                fn_highlight = self.projRoot +     "v/misc/capture7jf/%s/bucket_%s/%03d/%03d_%03d.pkl" % (
                    sceneTag, datasetConf["highlightBucket"], 40 * groupID, 40 * groupID, viewID,
                )
                assert os.path.isfile(fn_highlight), fn_highlight
                with open(fn_highlight, "rb") as f:
                    pkl_highlight = pickle.load(f)
                    pkl_highlight["hdr_highlight"] *= datasetConf["hdrScaling"]
                    pkl_highlight["hdr_sublight"] *= datasetConf["hdrScaling"]
                fn_mask = self.projRoot + "v/misc/capture7jf/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                    sceneTag, 40 * groupID, 40 * groupID, viewID
                )
                assert os.path.isfile(fn_mask), fn_mask
                with open(fn_mask, "rb") as f:
                    pkl_mask = pickle.load(f)

                assert pkl_mask["zonemap"].min() == 1
                zonemap1234 = (pkl_mask["zonemap"] < 5).reshape(-1)
                pixelID = pixelIDFull[zonemap1234]
                pixelID_highlight = pixelID[pkl_highlight["aa_highlight"]]
                olatID_highlight = pkl_highlight["bb_highlight"] + 1
                hdr_highlight = pkl_highlight["hdr_highlight"]
                pixelID_sublight = pixelID[pkl_highlight["aa_sublight"]]
                olatID_sublight = pkl_highlight["bb_sublight"] + 1
                hdr_sublight = pkl_highlight["hdr_sublight"]
                tag_highlight = np.concatenate([
                    np.ones((pixelID_highlight.shape[0],), dtype=bool),
                    np.zeros((pixelID_sublight.shape[0],), dtype=bool),
                ], 0)
                pixelID_highlight = np.concatenate([pixelID_highlight, pixelID_sublight], 0)
                olatID_highlight = np.concatenate([olatID_highlight, olatID_sublight], 0)
                hdr_highlight = np.concatenate([hdr_highlight, hdr_sublight], 0)

                # select olatIDs that do not belong to the Free3 test set
                olatID_testOnly = np.array(
                    [43, 73, 105, 137, 169, 203, 233, 265, 297, 329], dtype=np.int32)
                mask = np.isin(olatID_highlight, olatID_testOnly) == 0
                pixelID_highlight = pixelID_highlight[mask]
                olatID_highlight = olatID_highlight[mask]
                hdr_highlight = hdr_highlight[mask, :]
                tag_highlight = tag_highlight[mask]
                del mask  # this mask is not betterMask3...

                # construct the camera info now
                c2w0 = self.c2w[groupViewID]
                focalLengthWidth0 = self.focalLengthWidth[groupViewID]
                focalLengthHeight0 = self.focalLengthHeight[groupViewID]
                assert datasetConf["rays_background_pad_width"] == 0
                assert datasetConf["rays_background_pad_height"] == 0
                rowID = pixelID_highlight // winWidth
                colID = pixelID_highlight % winWidth
                directions_unormalized_cam = np.stack([
                    (colID.astype(np.float32) + 0.5 - winWidth / 2.0) / focalLengthWidth0,
                    (rowID.astype(np.float32) + 0.5 - winHeight / 2.0) / focalLengthHeight0,
                    np.ones((rowID.shape[0],), dtype=np.float32),
                ], 1)
                directions_unormalized = (
                    c2w0[None, :3, :3] * directions_unormalized_cam[:, None, :]
                ).sum(2)
                directions_normalized = normalize(directions_unormalized, dimAxis=1)
                rays_highlight = np.concatenate([
                    np.tile(
                        (c2w0[None, :3, 3] - sceneShiftFirst[None, :]) * sceneScaleSecond,
                        (rowID.shape[0], 1),
                    ),
                    directions_normalized,
                ], 1)

                # construct the lgt info now
                lgtE_highlight = (lgtE_of_the_group[olatID_highlight - 1, :] - sceneShiftFirst[None, :]) * sceneScaleSecond
                # We won't export objCentroid here, as objCentroid is the same throughout the program

                # IDs
                IDs = np.stack([
                    # sceneID * np.ones((B,), dtype=np.int32),
                    groupViewID * np.ones((pixelID_highlight.shape[0], ), dtype=np.int32),  # 0
                    olatID_highlight.astype(np.int32),  # 1
                    pixelID_highlight.astype(np.int32),  # 2
                ], 1)

                record_IDs_highlight.append(IDs[tag_highlight])
                record_hdr_highlight.append(hdr_highlight[tag_highlight])
                record_rays_highlight.append(rays_highlight[tag_highlight])
                record_lgtE_highlight.append(lgtE_highlight[tag_highlight])
                tag_sublight = tag_highlight == False
                record_IDs_sublight.append(IDs[tag_sublight])
                record_hdr_sublight.append(hdr_highlight[tag_sublight])
                record_rays_sublight.append(rays_highlight[tag_sublight])
                record_lgtE_sublight.append(lgtE_highlight[tag_sublight])

        record_IDs_highlight = np.concatenate(record_IDs_highlight, 0)
        record_hdr_highlight = np.concatenate(record_hdr_highlight, 0)
        record_rays_highlight = np.concatenate(record_rays_highlight, 0)
        record_lgtE_highlight = np.concatenate(record_lgtE_highlight, 0)
        record_IDs_sublight = np.concatenate(record_IDs_sublight, 0)
        record_hdr_sublight = np.concatenate(record_hdr_sublight, 0)
        record_rays_sublight = np.concatenate(record_rays_sublight, 0)
        record_lgtE_sublight = np.concatenate(record_lgtE_sublight, 0)

        self.highlight_IDs = record_IDs_highlight
        self.highlight_hdrs = record_hdr_highlight
        self.highlight_rays = record_rays_highlight
        self.highlight_lgtEs = record_lgtE_highlight

        self.sublight_IDs = record_IDs_sublight
        self.sublight_hdrs = record_hdr_sublight
        self.sublight_rays = record_rays_sublight
        self.sublight_lgtEs = record_lgtE_sublight

    def preload_original_buckets(self):
        datasetConf = self.datasetConf
        caseID = datasetConf["caseID"]
        sceneID = caseID
        sceneTag = self.A0[sceneID]["sceneTag"]
        bucket = datasetConf["bucket"]
        dataset = datasetConf["dataset"]
        winWidth = self.winWidth
        winHeight = self.winHeight

        if bucket == "nerf":  # we only care about ldr-nerf
            rays_tot = int(self.indTrain.shape[0]) * self.winWidth * self.winHeight

            rays_fg_tot = int(0.5 * rays_tot)
            self.all_fg_rgbs = np.zeros((rays_fg_tot, 3), dtype=np.float32)
            self.all_fg_indexID = np.zeros((rays_fg_tot,), dtype=np.int32)  # to get extrinsics
            self.all_fg_pixelID = np.zeros((rays_fg_tot,), dtype=np.int32)  # to get intrinsics
            rays_fg_count = 0

            rays_bg_tot = 180 * self.winWidth * self.winHeight
            self.all_bg_indexID = np.zeros((rays_bg_tot,), dtype=np.int32)
            self.all_bg_pixelID = np.zeros((rays_bg_tot,), dtype=np.int32)
            rays_bg_count = 0

            if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                rays_mg_tot = 180 * self.winWidth * self.winHeight
                self.all_mg_indexID = np.zeros((rays_mg_tot,), dtype=np.int32)
                self.all_mg_pixelID = np.zeros((rays_mg_tot,), dtype=np.int32)
                rays_mg_count = 0

            readInds = self.indTrain
            for j in readInds.tolist():
                print("    Preloading bucket nerf for %s: %d / %d" % (sceneTag, j, 180))
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if self.flagSplit[j] == 1:
                    assert not flagRemove
                    with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                            dataset, sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                        tmp = pickle.load(f)
                        zonemap0 = tmp["zonemap"]
                        betterMask3 = tmp["betterMask3"]
                        del tmp
                    if datasetConf["preloadingForegroundZoneNumberUpperBound"] >= 1:
                        mask0_foreground = (zonemap0 <= datasetConf["preloadingForegroundZoneNumberUpperBound"])
                    elif datasetConf["preloadingForegroundZoneNumberUpperBound"] == -1:
                        assert datasetConf["preloadingBackgroundZoneNumberLowerBound"] == -1
                        mask0_foreground = betterMask3
                    else:
                        raise NotImplementedError("Unknown preloadingForegroundZoneNumberUpperBound: %s" % datasetConf["preloadingForegroundZoneNumberUpperBound"])
                    if datasetConf["preloadingBackgroundZoneNumberLowerBound"] >= 1:
                        mask0_background = (zonemap0 >= datasetConf["preloadingBackgroundZoneNumberLowerBound"])
                    elif datasetConf["preloadingBackgroundZoneNumberLowerBound"] == -1:
                        assert datasetConf["preloadingForegroundZoneNumberUpperBound"] == -1
                        mask0_background = (betterMask3 == 0)
                    else:
                        raise NotImplementedError("Unknown preloadingBackgroundZoneNumberLowerBound: %s" % datasetConf["preloadingBackgroundZoneNumberLowerBound"])
                    if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                        # dilate the positive betterMask3, and then remove the original positive betterMask3 (only remains the newly dilated part)
                        L = datasetConf["mgPixDilation"]
                        kernel = np.ones((L, L), dtype=np.float32)
                        dilated = (cv2.dilate(betterMask3.astype(np.float32), kernel, iterations=1) > 0)
                        dilated[betterMask3] = False
                        mgMask0 = dilated
                        del dilated
                    with open(self.projRoot + "v/misc/%s/%s/bucket_nerf3/%03d/%03d_%03d.pkl" % (
                            dataset, sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                        rgbs = pickle.load(f)["rgbs"]

                    m_fg_rays_now = int(mask0_foreground.sum())
                    #
                    self.all_fg_rgbs[rays_fg_count : rays_fg_count + m_fg_rays_now] = rgbs[np.where(mask0_foreground[zonemap0 <= 4])[0], :]
                    #
                    self.all_fg_indexID[rays_fg_count : rays_fg_count + m_fg_rays_now] = j
                    yy, xx = np.where(mask0_foreground)
                    assert datasetConf.get("rays_background_pad_width", 0) == 0
                    self.all_fg_pixelID[rays_fg_count : rays_fg_count + m_fg_rays_now] = yy * self.winWidth + xx  # There is no background padding
                    del yy, xx
                    rays_fg_count += m_fg_rays_now

                    m_bg_rays_now = int(mask0_background.sum())
                    self.all_bg_indexID[rays_bg_count : rays_bg_count + m_bg_rays_now] = j
                    yy, xx = np.where(mask0_background)
                    self.all_bg_pixelID[rays_bg_count : rays_bg_count + m_bg_rays_now] = yy * self.winWidth + xx
                    del yy, xx
                    rays_bg_count += m_bg_rays_now

                    if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                        m_mg_rays_now = int(mgMask0.sum())
                        self.all_mg_indexID[rays_mg_count : rays_mg_count + m_mg_rays_now] = j
                        yy, xx = np.where(mgMask0)
                        assert datasetConf.get("rays_background_pad_width", 0) == 0
                        self.all_mg_pixelID[rays_mg_count : rays_mg_count + m_mg_rays_now] = yy * self.winWidth + xx  # There is no background padding
                        del yy, xx
                        rays_mg_count += m_mg_rays_now

            self.all_fg_rgbs = self.all_fg_rgbs[:rays_fg_count]
            self.all_fg_indexID = self.all_fg_indexID[:rays_fg_count]
            self.all_fg_pixelID = self.all_fg_pixelID[:rays_fg_count]

            self.all_bg_indexID = self.all_bg_indexID[:rays_bg_count]
            self.all_bg_pixelID = self.all_bg_pixelID[:rays_bg_count]

            if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                self.all_mg_indexID = self.all_mg_indexID[:rays_mg_count]
                self.all_mg_pixelID = self.all_mg_pixelID[:rays_mg_count]

        elif bucket.startswith("ol3t"):

            rays_tot = (self.indTrain.shape[0]) * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])

            rays_fg_tot = int(0.5 * rays_tot)
            self.all_fg_hdrs = np.zeros((rays_fg_tot, 3), dtype=np.float32)
            self.all_fg_indexID = np.zeros((rays_fg_tot,), dtype=np.int32)  # You can get groupViewID and ccID from this single number (groupViewID 0~179, ccID 0~74)

            self.all_fg_pixelID = np.zeros((rays_fg_tot,), dtype=np.int32)
            if datasetConf.get("ifLoadDepth", False):  # This depth is zbuffer depth
                self.all_fg_zbuffer = np.zeros((rays_fg_tot,), dtype=np.float32)
            rays_fg_count = 0

            rays_bg_tot = 180 * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])
            # for different lightings, we only store one lighting, as they are all the same
            self.all_bg_groupViewID = np.zeros((rays_bg_tot,), dtype=np.int32)  # [0, 180)
            self.all_bg_pixelID = np.zeros((rays_bg_tot,), dtype=np.int32)
            rays_bg_count = 0

            if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                rays_mg_tot = 180 * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])
                self.all_mg_groupViewID = np.zeros((rays_mg_tot,), dtype=np.int32)
                self.all_mg_pixelID = np.zeros((rays_mg_tot,), dtype=np.int32)
                rays_mg_count = 0

            self.olat_lgtEs = np.zeros((180, self.nOlat, 3), dtype=np.float32)
            self.objCentroid0 = None

            readInds = np.arange(180).astype(np.int32)  # buckets that does not belong to the current training set (including not the right GPU) will be filtered inside the for-loop
            for j in readInds.tolist():
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if np.any(self.flagSplit[
                    (groupID * 20 + viewID - 1) * self.nOlat + 0 : (groupID * 20 + viewID - 1) * self.nOlat + self.nOlat
                ] == 1):  # As long as there is one on, then you need to load this bucket
                    assert not flagRemove
                    print("     Preloading bucket %s for %s: %d / %d" % (bucket, sceneTag, j, 180))
                    with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                            dataset, sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                        tmp = pickle.load(f)
                        zonemap0 = tmp["zonemap"]
                        betterMask3 = tmp["betterMask3"]
                        del tmp
                    if datasetConf["preloadingForegroundZoneNumberUpperBound"] >= 1:
                        mask0_foreground = (zonemap0 <= datasetConf["preloadingForegroundZoneNumberUpperBound"])
                    elif datasetConf["preloadingForegroundZoneNumberUpperBound"] == -1:
                        assert datasetConf["preloadingBackgroundZoneNumberLowerBound"] == -1
                        mask0_foreground = betterMask3
                    else:
                        raise NotImplementedError("Unknown preloadingForegroundZoneNumberUpperBound: %s" % datasetConf["preloadingForegroundZoneNumberUpperBound"])
                    if datasetConf["preloadingBackgroundZoneNumberLowerBound"] >= 1:
                        mask0_background = (zonemap0 >= datasetConf["preloadingBackgroundZoneNumberLowerBound"])
                    elif datasetConf["preloadingBackgroundZoneNumberLowerBound"] == -1:
                        assert datasetConf["preloadingForegroundZoneNumberUpperBound"] == -1
                        mask0_background = (betterMask3 == 0)
                    else:
                        raise NotImplementedError("Unknown preloadingBackgroundZoneNumberLowerBound: %s" % datasetConf["preloadingBackgroundZoneNumberLowerBound"])
                    if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                        # dilate the positive betterMask3, and then remove the original positive betterMask3 (only remains the newly dilated part)
                        L = datasetConf["mgPixDilation"]
                        kernel = np.ones((L, L), dtype=np.float32)
                        dilated = (cv2.dilate(betterMask3.astype(np.float32), kernel, iterations=1) > 0)
                        dilated[betterMask3] = False
                        mgMask0 = dilated
                        del dilated
                    with open(self.projRoot + "v/misc/%s/%s/bucket_%s/%03d/%03d_%03d.pkl" % (
                            dataset, sceneTag, bucket, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                        # rgbs = pickle.load(f)["rgbs"]
                        pkl = pickle.load(f)
                        pkl["ol3ts"] *= datasetConf["hdrScaling"]

                        lgtEs = pkl["lgtEs"]
                        objCentroid0 = pkl["objCentroid0"]
                        olats = pkl["ol3ts"]
                        q = pkl["q"]

                        # q[:] = 0  # debug Let us keep this first

                        lens_flares_free_flag = q <= 0.005  # this is the judgement of whether or not it is corrupted by lens flare, and it is pretty accurate
                        lens_flares_free_ind = np.where(lens_flares_free_flag)[0]
                        ens_flares_free_tot = int(lens_flares_free_ind.shape[0])
                        self.olat_lgtEs[j, :, :] = lgtEs  # seems apparently a bug  # not necessarily, take a look at the convention at the top of this code file
                        if self.objCentroid0 is None:
                            self.objCentroid0 = objCentroid0
                        else:
                            assert np.all(self.objCentroid0 == objCentroid0)
                    if datasetConf.get("ifLoadDepth", False):
                        if datasetConf["loadDepthTag"] == "neurecon":
                            raise NotImplementedError("Have not yet tried its correctness")
                            depth_fn = self.projRoot + "v/misc/real_capture_surfacenerf_results/%s/capture7jf_scene_%d/groupID_%d/sceneID_%d_groupID_%d_viewID_%d_winSize_2048.pkl" % (
                                datasetConf["depthFileListDirTag"], sceneID, groupID, sceneID, groupID, viewID
                            )
                            assert os.path.isfile(depth_fn), depth_fn
                            with open(depth_fn, "rb") as f:
                                zbuffer0 = pickle.load(f)["depth0"] * datasetConf["sceneScaleSecond"]  # wait, this is euclidean depth, right?? Seems not using zbuffer depth would be a bug??
                        elif datasetConf["loadDepthTag"] == "NeuS":
                            depth_fn = self.projRoot + "v/misc/neusdepth/capture7jf/case%d/%03d/%03d_%03d.pkl" % (
                                caseID, groupID * 40, groupID * 40, viewID,
                            )
                            assert os.path.isfile(depth_fn), depth_fn
                            with open(depth_fn, "rb") as f:
                                zbuffer0 = pickle.load(f)["zbufferDepth0"] * datasetConf["sceneScaleSecond"]
                            zbuffer0[np.isfinite(zbuffer0) == 0] = 0
                        else:
                            raise ValueError("Unknown datasetConf['loadDepthTag']: %s" % datasetConf["loadDepthTag"])

                    m_fg_rays_now = int(mask0_foreground.sum())
                    lens_flares_free_tot = int(lens_flares_free_ind.shape[0])
                    self.all_fg_hdrs[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot, :] = (
                        olats[np.where(mask0_foreground[zonemap0 <= 4])[0], :, :][:, lens_flares_free_ind, :]
                            .reshape((m_fg_rays_now * lens_flares_free_tot, 3))
                    )
                    self.all_fg_indexID[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                        np.tile(
                            j * self.nOlat + lens_flares_free_ind,
                            (m_fg_rays_now,)
                        )
                    )
                    yy, xx = np.where(mask0_foreground)
                    self.all_fg_pixelID[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                        np.tile(
                            ((yy + datasetConf["rays_background_pad_height"]) * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + (xx + datasetConf["rays_background_pad_width"]))[:, None],
                            (1, lens_flares_free_tot,),
                        ).reshape((m_fg_rays_now * lens_flares_free_tot,))
                    )
                    del yy, xx
                    if datasetConf.get("ifLoadDepth", False):

                        self.all_fg_zbuffer[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                            np.tile(
                                zbuffer0[mask0_foreground][:, None],
                                (1, lens_flares_free_tot),
                            ).reshape((m_fg_rays_now * lens_flares_free_tot,))
                        )
                    rays_fg_count += m_fg_rays_now * lens_flares_free_tot

                    # background: only store under one lighting condition --> which stores exactly the same as the nerf
                    mask0_background_padded = np.ones((
                        self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                        self.winWidth + 2 * datasetConf["rays_background_pad_width"],
                    ), dtype=bool)
                    mask0_background_padded[
                        datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                        datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
                    ] = mask0_background
                    m_bg_rays_now = int(mask0_background_padded.sum())
                    self.all_bg_groupViewID[rays_bg_count : rays_bg_count + m_bg_rays_now] = j
                    yy, xx = np.where(mask0_background_padded)
                    self.all_bg_pixelID[rays_bg_count : rays_bg_count + m_bg_rays_now] = yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx
                    del yy, xx
                    rays_bg_count += m_bg_rays_now

                    # middleground: (still a subset of background) the same as the background regarding formatting
                    if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                        mask0_middleground_padded = np.zeros((  # the padded ones are never put to the middle ground
                            self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                            self.winWidth + 2 * datasetConf["rays_background_pad_width"],
                        ), dtype=bool)
                        mask0_middleground_padded[
                            datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                            datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
                        ] = mgMask0
                        m_mg_rays_now = int(mask0_middleground_padded.sum())
                        self.all_mg_groupViewID[rays_mg_count : rays_mg_count + m_mg_rays_now] = j
                        yy, xx = np.where(mask0_middleground_padded)
                        self.all_mg_pixelID[rays_mg_count : rays_mg_count + m_mg_rays_now] = yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx
                        del yy, xx
                        rays_mg_count += m_mg_rays_now

            self.all_fg_hdrs = self.all_fg_hdrs[:rays_fg_count]
            self.all_fg_indexID = self.all_fg_indexID[:rays_fg_count]
            self.all_fg_pixelID = self.all_fg_pixelID[:rays_fg_count]
            if datasetConf.get("ifLoadDepth", False):
                self.all_fg_zbuffer = self.all_fg_zbuffer[:rays_fg_count]
            self.all_bg_groupViewID = self.all_bg_groupViewID[:rays_bg_count]
            self.all_bg_pixelID = self.all_bg_pixelID[:rays_bg_count]
            if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                self.all_mg_groupViewID = self.all_mg_groupViewID[:rays_mg_count]
                self.all_mg_pixelID = self.all_mg_pixelID[:rays_mg_count]

        elif bucket.startswith("envmap"):

            rays_tot = (self.indTrain.shape[0]) * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])

            rays_fg_tot = int(0.5 * rays_tot)
            self.all_fg_hdrs = np.zeros((rays_fg_tot, 3), dtype=np.float32)
            self.all_fg_indexID = np.zeros((rays_fg_tot,), dtype=np.int32)  # You can get groupViewID and ccID from this single number (groupViewID 0~179, ccID 0~74)
            self.all_fg_lgtID = np.zeros((rays_fg_tot,), dtype=np.int32)

            self.all_fg_pixelID = np.zeros((rays_fg_tot,), dtype=np.int32)
            if datasetConf.get("ifLoadDepth", False):  # This depth is zbuffer depth
                self.all_fg_zbuffer = np.zeros((rays_fg_tot,), dtype=np.float32)
            rays_fg_count = 0

            rays_bg_tot = 180 * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])
            # for different lightings, we only store one lighting, as they are all the same
            self.all_bg_groupViewID = np.zeros((rays_bg_tot,), dtype=np.int32)  # [0, 180)
            self.all_bg_pixelID = np.zeros((rays_bg_tot,), dtype=np.int32)
            rays_bg_count = 0

            if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                rays_mg_tot = 180 * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])
                self.all_mg_groupViewID = np.zeros((rays_mg_tot,), dtype=np.int32)
                self.all_mg_pixelID = np.zeros((rays_mg_tot,), dtype=np.int32)
                rays_mg_count = 0

            # self.olat_lgtEs = np.zeros((180, self.nOlat, 3), dtype=np.float32)
            self.objCentroid0 = None

            readInds = np.arange(180).astype(np.int32)  # buckets that does not belong to the current training set (including not the right GPU) will be filtered inside the for-loop
            for j in readInds.tolist():
                groupID = j // 20
                viewID = j % 20 + 1
                flagRemove1 = (sceneID == 1) and (
                    (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
                )
                flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                flagRemove = flagRemove1 or flagRemove10
                if np.any(self.flagSplit[
                    (groupID * 20 + viewID - 1) * self.nOlat + 0 : (groupID * 20 + viewID - 1) * self.nOlat + self.nOlat
                ] == 1):  # As long as there is one on, then you need to load this bucket
                    assert not flagRemove
                    print("     Preloading bucket %s for %s: %d / %d" % (bucket, sceneTag, j, 180))
                    with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                            dataset, sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                        tmp = pickle.load(f)
                        zonemap0 = tmp["zonemap"]
                        betterMask3 = tmp["betterMask3"]
                        del tmp
                    if datasetConf["preloadingForegroundZoneNumberUpperBound"] >= 1:
                        mask0_foreground = (zonemap0 <= datasetConf["preloadingForegroundZoneNumberUpperBound"])
                    elif datasetConf["preloadingForegroundZoneNumberUpperBound"] == -1:
                        assert datasetConf["preloadingBackgroundZoneNumberLowerBound"] == -1
                        mask0_foreground = betterMask3
                    else:
                        raise NotImplementedError("Unknown preloadingForegroundZoneNumberUpperBound: %s" % datasetConf["preloadingForegroundZoneNumberUpperBound"])
                    if datasetConf["preloadingBackgroundZoneNumberLowerBound"] >= 1:
                        mask0_background = (zonemap0 >= datasetConf["preloadingBackgroundZoneNumberLowerBound"])
                    elif datasetConf["preloadingBackgroundZoneNumberLowerBound"] == -1:
                        assert datasetConf["preloadingForegroundZoneNumberUpperBound"] == -1
                        mask0_background = (betterMask3 == 0)
                    else:
                        raise NotImplementedError("Unknown preloadingBackgroundZoneNumberLowerBound: %s" % datasetConf["preloadingBackgroundZoneNumberLowerBound"])
                    if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                        # dilate the positive betterMask3, and then remove the original positive betterMask3 (only remains the newly dilated part)
                        L = datasetConf["mgPixDilation"]
                        kernel = np.ones((L, L), dtype=np.float32)
                        dilated = (cv2.dilate(betterMask3.astype(np.float32), kernel, iterations=1) > 0)
                        dilated[betterMask3] = False
                        mgMask0 = dilated
                        del dilated
                    with open(self.projRoot + "v/misc/%s/%s/bucket_%s/%03d/%03d_%03d.pkl" % (
                            dataset, sceneTag, bucket, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                        # rgbs = pickle.load(f)["rgbs"]
                        pkl = pickle.load(f)
                        raise NotImplementedError("Have not yet conducted hdrScaling")
                        assert pkl["lgtDataset"] == datasetConf["envmapDataset"]
                        hdrs = pkl["hdrs"] * datasetConf["envmapHdrScaling"]
                        envmaps = pkl["envmaps"]
                        lgtID = pkl["lgtID"]
                        self.envmapLgtDatasetCache.ensureHotCache(pkl["lgtDataset"], lgtID)

                    if datasetConf.get("ifLoadDepth", False):
                        depth_fn = self.projRoot + "v/misc/real_capture_surfacenerf_results/%s/capture7jf_scene_%d/groupID_%d/sceneID_%d_groupID_%d_viewID_%d_winSize_2048.pkl" % (
                            datasetConf["depthFileListDirTag"], sceneID, groupID, sceneID, groupID, viewID
                        )
                        assert os.path.isfile(depth_fn), depth_fn
                        with open(depth_fn, "rb") as f:
                            zbuffer0 = pickle.load(f)["depth0"]

                    m_fg_rays_now = int(mask0_foreground.sum())
                    lens_flares_free_tot = int(envmaps.shape[0])
                    assert lens_flares_free_tot == self.nOlat
                    self.all_fg_hdrs[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot, :] = (
                        hdrs[np.where(mask0_foreground[zonemap0 <= 4])[0], :, :]
                            .reshape((m_fg_rays_now * lens_flares_free_tot, 3))
                    )
                    self.all_fg_indexID[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                        np.tile(
                            j * self.nOlat + np.arange(self.nOlat).astype(np.int32),
                            (m_fg_rays_now,)
                        )
                    )
                    self.all_fg_lgtID[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                        np.tile(
                            lgtID, (m_fg_rays_now,),
                        )
                    )
                    yy, xx = np.where(mask0_foreground)
                    self.all_fg_pixelID[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                        np.tile(
                            ((yy + datasetConf["rays_background_pad_height"]) * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + (xx + datasetConf["rays_background_pad_width"]))[:, None],
                            (1, lens_flares_free_tot,),
                        ).reshape((m_fg_rays_now * lens_flares_free_tot,))
                    )
                    del yy, xx
                    if datasetConf.get("ifLoadDepth", False):
                        raise NotImplementedError("I am pretty sure that you have not rescale the zbuffer depth with sceneScaleSecond yet.")
                        raise NotImplementedError("According to the convention, depth rescale should happen immediately after loading the depth.")
                        raise NotImplementedError("Although the following is newly implemented, its correctness has not been validated so far.")
                        self.all_fg_zbuffer[rays_fg_count : rays_fg_count + m_fg_rays_now * lens_flares_free_tot] = (
                            np.tile(
                                zbuffer0[mask0_foreground][:, None],
                                (1, lens_flares_free_tot),
                            ).reshape((m_fg_rays_now * lens_flares_free_tot,))
                        )
                    rays_fg_count += m_fg_rays_now * lens_flares_free_tot

                    # background: only store under one lighting condition --> which stores exactly the same as the nerf
                    mask0_background_padded = np.ones((
                        self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                        self.winWidth + 2 * datasetConf["rays_background_pad_width"],
                    ), dtype=bool)
                    mask0_background_padded[
                        datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                        datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
                    ] = mask0_background
                    m_bg_rays_now = int(mask0_background_padded.sum())
                    self.all_bg_groupViewID[rays_bg_count : rays_bg_count + m_bg_rays_now] = j
                    yy, xx = np.where(mask0_background_padded)
                    self.all_bg_pixelID[rays_bg_count : rays_bg_count + m_bg_rays_now] = yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx
                    del yy, xx
                    rays_bg_count += m_bg_rays_now

                    # middleground: (still a subset of background) the same as the background regarding formatting
                    if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                        mask0_middleground_padded = np.zeros((  # the padded ones are never put to the middle ground
                            self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                            self.winWidth + 2 * datasetConf["rays_background_pad_width"],
                        ), dtype=bool)
                        mask0_middleground_padded[
                            datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                            datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
                        ] = mgMask0
                        m_mg_rays_now = int(mask0_middleground_padded.sum())
                        self.all_mg_groupViewID[rays_mg_count : rays_mg_count + m_mg_rays_now] = j
                        yy, xx = np.where(mask0_middleground_padded)
                        self.all_mg_pixelID[rays_mg_count : rays_mg_count + m_mg_rays_now] = yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx
                        del yy, xx
                        rays_mg_count += m_mg_rays_now

            self.all_fg_hdrs = self.all_fg_hdrs[:rays_fg_count]
            self.all_fg_indexID = self.all_fg_indexID[:rays_fg_count]
            if bucket.startswith("envmap"):
                self.all_fg_lgtID = self.all_fg_lgtID[:rays_fg_count]
            self.all_fg_pixelID = self.all_fg_pixelID[:rays_fg_count]
            if datasetConf.get("ifLoadDepth", False):
                self.all_fg_zbuffer = self.all_fg_zbuffer[:rays_fg_count]
            self.all_bg_groupViewID = self.all_bg_groupViewID[:rays_bg_count]
            self.all_bg_pixelID = self.all_bg_pixelID[:rays_bg_count]
            if datasetConf.get("batchSizeMgPerProcess", 0) > 0:
                self.all_mg_groupViewID = self.all_mg_groupViewID[:rays_mg_count]
                self.all_mg_pixelID = self.all_mg_pixelID[:rays_mg_count]

        else:
            raise NotImplementedError("Unknown bucket: %s" % bucket)

        print("[Capture7jf Dataset Loader] Preloading Main finished!")

    def getOneNP(self, idx):
        assert self.singleImageMode
        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        caseID = datasetConf["caseID"]
        sceneID = caseID
        sceneTag = self.A0[sceneID]["sceneTag"]
        winWidth = self.winWidth
        winHeight = self.winHeight

        if datasetConf["bucket"] == "nerf":
            groupID = idx // 20
            viewID = idx % 20 + 1
            groupViewID = idx

            # rays
            focalLengthWidth = self.focalLengthWidth[idx]
            focalLengthHeight = self.focalLengthHeight[idx]
            c2w0 = self.c2w[idx]
            rowID = np.tile(np.arange(winHeight).astype(np.int32)[:, None], (1, winWidth)).reshape(-1)
            colID = np.tile(np.arange(winWidth).astype(np.int32), (winHeight,))
            directions_unnormalized_cam = np.stack([
                (colID.astype(np.float32) + 0.5 - winWidth / 2.0) / focalLengthWidth,
                (rowID.astype(np.float32) + 0.5 - winHeight / 2.0) / focalLengthHeight,
                np.ones((rowID.shape[0],), dtype=np.float32),
            ], 1)
            directions_unnormalized = (c2w0[None, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
            directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
            directions_normalized = directions_unnormalized / directions_norm[:, None]
            assert datasetConf["white_back"] == False  # this is always the case for real data
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

            with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                    dataset, sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                tmp = pickle.load(f)
                betterMask3 = tmp["betterMask3"]
                zonemap = tmp["zonemap"]
            if datasetConf["singleImageForegroundZoneNumberUpperBound"] == -1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == -1
                mask0 = betterMask3
            elif datasetConf["singleImageForegroundZoneNumberUpperBound"] >= 1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == datasetConf["singleImageForegroundZoneNumberUpperBound"] + 1
                mask0 = zonemap <= datasetConf["singleImageForegroundZoneNumberUpperBound"]
            else:
                raise ValueError("Value of singleImageForegroundZoneNumberUpperBound can only be -1, 1, 2, 3, 4, but here is %d" % (
                    datasetConf["singleImageForegroundZoneNumberUpperBound"]
                ))
            sum_up_fn = self.projRoot + "v/misc/%s/%s/sum_up/%03d/%03d_%03d.png" % (
                dataset, sceneTag, 40 * groupID, 40 * groupID, viewID)
            assert os.path.isfile(sum_up_fn), sum_up_fn
            tmpbgr0 = cv2.imread(sum_up_fn, flags=cv2.IMREAD_UNCHANGED)
            assert tmpbgr0.shape == (winHeight, winWidth, 3) and tmpbgr0.dtype == np.uint8
            rgb0 = np.ascontiguousarray(tmpbgr0.astype(np.float32)[:, :, ::-1] / 255.0)
            mask3 = np.tile(mask0[:, :, None], (1, 1, 3))
            rgb0[mask3 == 0] = 0

            sample = {
                "rays": rays,
                "rgbs": rgb0.reshape((-1, 3)),
                "c2w": c2w0,
                "valid_mask": mask0.reshape((-1,)),
                "index": np.array(idx, dtype=np.int32),
                "flagSplit": self.flagSplit[idx],
                "dataset": dataset,
                "winWidth": np.array(winWidth, dtype=np.int32),
                "winHeight": np.array(winHeight, dtype=np.int32),
                "focalLengthWidth": np.array(focalLengthWidth, dtype=np.float32),
                "focalLengthHeight": np.array(focalLengthHeight, dtype=np.float32),
                "groupViewID": groupViewID,
                "groupID": groupID,
                "viewID": viewID,
                "ccID": -1,
                "olatID": -1,
            }

            assert "envmap" not in sample.keys()  # use either envmap0 or samples
            return sample

        elif datasetConf["bucket"].startswith("envmap"):
            # sometimes having the label, sometimes not, depending on whether there are bucket files there
            bucket = datasetConf["bucket"]
            samples = {}

            groupViewID = idx // self.nOlat
            groupID = groupViewID // 20
            viewID = groupViewID % 20 + 1
            ccID = idx % self.nOlat

            # whether or not have the bucket file
            probe_bucket_file = self.projRoot + "v/misc/%s/%s/bucket_%s/%03d/%03d_%03d.pkl" % (
                "capture7jf", sceneTag, bucket, 40 * groupID, 40 * groupID, viewID,
            )
            if os.path.isfile(probe_bucket_file):
                with open(probe_bucket_file, "rb") as f:
                    bucketPkl = pickle.load(f)
                    raise NotImplementedError("Not yet conducted hdrScaling")
                assert datasetConf["envmapDataset"] == bucketPkl["lgtDataset"]
                lgtDataset = datasetConf["envmapDataset"]
                lgtID = bucketPkl["lgtID"]
                self.envmapLgtDatasetCache.ensureHotCache(bucketPkl["lgtDataset"], lgtID)
                assert np.all(
                    self.envmapLgtDatasetCache.cache[bucketPkl["lgtDataset"]]["envmaps"][lgtID] ==
                    bucketPkl["envmaps"]
                )
                samples["lgtDataset"] = lgtDataset
                samples["lgtID"] = lgtID[ccID]
                samples["envmaps"] = np.tile(
                    bucketPkl["envmaps"][ccID, :, :, :][None, :, :, :],
                    (winWidth * winHeight, 1, 1, 1,),
                )
            else:
                print(probe_bucket_file)
                raise NotImplementedError("You need to set up your rule for mapping and loading the envmaps. This file is does not exist: %s" % probe_bucket_file)
                samples["XXX"] = XXX
                bucketPkl = None

            # rays (Notes in single image mode, we do not do background pad, even if datasetConf specifies pad)
            focalLengthWidth = self.focalLengthWidth[groupViewID]
            focalLengthHeight = self.focalLengthHeight[groupViewID]
            c2w0 = self.c2w[groupViewID]
            rowID = np.tile(np.arange(winHeight).astype(np.int32)[:, None], (1, winWidth)).reshape(-1)
            colID = np.tile(np.arange(winWidth).astype(np.int32), (winHeight,))
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

            # load
            with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                    "capture7jf", sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                tmp = pickle.load(f)
                betterMask3 = tmp["betterMask3"]
                zonemap = tmp["zonemap"]
            if datasetConf["singleImageForegroundZoneNumberUpperBound"] == -1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == -1
                mask0 = betterMask3
            elif datasetConf["singleImageForegroundZoneNumberUpperBound"] >= 1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == datasetConf["singleImageForegroundZoneNumberUpperBound"] + 1
                mask0 = zonemap <= datasetConf["singleImageForegroundZoneNumberUpperBound"]
            else:
                raise ValueError("Value of singleImageForegroundZoneNumberUpperBound can only be -1, 1, 2, 3, 4, but here is %d" % (
                    datasetConf["singleImageForegroundZoneNumberUpperBound"]
                ))

            if bucketPkl:
                r = np.zeros((winHeight, winWidth), dtype=np.float32)
                g = np.zeros((winHeight, winWidth), dtype=np.float32)
                b = np.zeros((winHeight, winWidth), dtype=np.float32)
                r[zonemap <= 4] = bucketPkl["hdrs"][:, ccID, 0]
                g[zonemap <= 4] = bucketPkl["hdrs"][:, ccID, 1]
                b[zonemap <= 4] = bucketPkl["hdrs"][:, ccID, 2]
                r[mask0 == 0] = 0
                g[mask0 == 0] = 0
                b[mask0 == 0] = 0
                hdr0 = np.stack([r, g, b], 2)
                rgb0 = np.zeros_like(hdr0)
                B = rays.shape
            else:
                hdr0 = np.zeros((winHeight, winWidth, 3), dtype=np.float32)
                rgb0 = np.zeros((winHeight, winWidth, 3), dtype=np.float32)

            # depth
            # if datasetConf.get("ifLoadDepth", False):
            if True:  # always load, as this is the validaton phase
                depth_fn = projRoot + "v/misc/real_capture_surfacenerf_results/%s/capture7jf_scene_%d/groupID_%d/sceneID_%d_groupID_%d_viewID_%d_winSize_2048.pkl" % (
                    "real_capture_neureconwm_results", sceneID, groupID, sceneID, groupID, viewID,
                )
                assert os.path.isfile(depth_fn), depth_fn
                with open(depth_fn, "rb") as f:
                    depth0 = pickle.load(f)["depth0"]  # Make sure to remember that this depth is zbuffer depth
                assert depth0.shape == (self.winHeight, self.winWidth)
                assert depth0.dtype == np.float32
                depth0 *= sceneScaleSecond

            samples.update({
                "rays": rays,
                "hdrs": hdr0.reshape((-1, 3)),
                "rgbs": rgb0.reshape((-1, 3)),
                "c2w": c2w0,
                "valid_mask": mask0.reshape((-1,)),
                "index": np.array(idx, dtype=np.int32),
                "flagSplit": self.flagSplit[(groupID * 20 + viewID - 1) * self.nOlat + 0],  # "+0": only check for the first light flag
                "dataset": dataset,
                "winWidth": np.array(winWidth, dtype=np.int32),
                "winHeight": np.array(winHeight, dtype=np.int32),
                "depth": depth0.reshape((-1,)),
                "zbuffer_to_euclidean_ratio": zbuffer_to_euclidean_ratio,
                "groupViewID": groupViewID,
                "groupID": groupID,
                "viewID": viewID,
                "ccID": ccID,
            })

            assert "envmap" not in samples.keys()  # use either envmap0 or samples
            return samples

        # no need for ground truth
        elif datasetConf["bucket"].startswith("distill") or datasetConf["bucket"].startswith("lfdistill"):
            # For sure not to have the label

            bucket = datasetConf["bucket"]
            samples = {}
            groupViewID = idx // self.nOlat
            groupID = groupViewID // 20
            viewID = groupViewID % 20 + 1
            ccID = idx % self.nOlat

            # rays (Notes in single image mode, we do not do background pad, even if datasetConf specifies pad)
            focalLengthWidth = self.focalLengthWidth[groupViewID]
            focalLengthHeight = self.focalLengthHeight[groupViewID]
            c2w0 = self.c2w[groupViewID]
            rowID = np.tile(np.arange(winHeight).astype(np.int32)[:, None], (1, winWidth)).reshape(-1)
            colID = np.tile(np.arange(winWidth).astype(np.int32), (winHeight,))
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

            # get the mean lgtE radius (before shift and scale)
            objCentroid0 = (self.objCentroidCache.getCache(caseID) - sceneShiftFirst) * sceneScaleSecond

            if caseID in [1]:  # , 10]:
                flagRemoves = np.zeros((9,), dtype=bool)
                for _ in range(9):
                    flagRemove1 = (caseID == 1) and (
                        (_ in [7, 8])  # or ((groupID == 6) and (viewID == 1))
                    )
                    # flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
                    # flagRemove = flagRemove1 or flagRemove10
                    flagRemoves[_] = flagRemove1
            else:
                flagRemoves = np.zeros((9,), dtype=bool)

            lgtEs = (np.stack([
                (self.A0b[caseID]["leaf_lgt_collectors"][groupOlatID // 331][groupOlatID % 331 + 1]["EWorld0"]
                    if (flagRemoves[groupOlatID // 331] == 0) else np.nan * np.zeros((3,), dtype=np.float32))
                for groupOlatID in range(331 * 9)
            ], 0) - sceneShiftFirst[None, :]) * sceneScaleSecond
            lgtEsRelative = lgtEs - objCentroid0[None, :]
            lgtERadiusMean = np.linalg.norm(lgtEsRelative, ord=2, axis=1).mean()

            if bucket == "distill512":
                thetas = np.arange(16).astype(np.float32) * (np.pi / 16) + np.pi / 16 / 2
                phis = np.pi - (  # according to blender's convention
                    np.arange(32).astype(np.float32) * (2 * np.pi / 32) + 2 * np.pi / 32 / 2
                )
                ccRowID = ccID // 32
                ccColID = ccID % 32
                assert 0 <= ccRowID < 16
                theta = thetas[ccRowID]
                assert 0 <= ccColID < 32
                phi = phis[ccColID]
                omegaGlobal0 = np.array([
                    np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta),
                ], dtype=np.float32)
                lgtE0 = omegaGlobal0 * lgtERadiusMean + objCentroid0
                # raise ValueError("I have not checked throughly for this setting, but it seems you did not rescale lgtE0 or LgtEs with sceneShiftFirst and sceneScaleSecond")
                # raise ValueError("I made this bug fix 23 lines above this line. Hope it is correct.")
            elif bucket == "lfdistill":
                lgtE0 = lgtEs[ccID, :]
            else:
                raise NotImplementedError("Unknown bucket: %s" % bucket)

            # load
            with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                    "capture7jf", sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                tmp = pickle.load(f)
                betterMask3 = tmp["betterMask3"]
                zonemap = tmp["zonemap"]
            if datasetConf["singleImageForegroundZoneNumberUpperBound"] == -1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == -1
                mask0 = betterMask3
            elif datasetConf["singleImageForegroundZoneNumberUpperBound"] >= 1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == datasetConf["singleImageForegroundZoneNumberUpperBound"] + 1
                mask0 = zonemap <= datasetConf["singleImageForegroundZoneNumberUpperBound"]
            else:
                raise ValueError("Value of singleImageForegroundZoneNumberUpperBound can only be -1, 1, 2, 3, 4, but here is %d" % (
                    datasetConf["singleImageForegroundZoneNumberUpperBound"]
                ))

            hdr0 = np.zeros((winHeight, winWidth, 3), dtype=np.float32)
            rgb0 = np.zeros((winHeight, winWidth, 3), dtype=np.float32)

            B = rays.shape[0]

            if datasetConf.get("ifLoadDepth", False):
                if datasetConf["loadDepthTag"] == "neurecon":
                    raise NotImplementedError("Have not yet tried its correctness")
                    depth_fn = projRoot + "v/misc/real_capture_surfacenerf_results/%s/capture7jf_scene_%d/groupID_%d/sceneID_%d_groupID_%d_viewID_%d_winSize_2048.pkl" % (
                        "real_capture_neureconwm_results", sceneID, groupID, sceneID, groupID, viewID,
                    )
                    assert os.path.isfile(depth_fn), depth_fn
                    with open(depth_fn, "rb") as f:
                        depth0 = pickle.load(f)["depth0"]  # Make sure to remember that this depth is zbuffer depth
                    assert depth0.shape == (self.winHeight, self.winWidth)
                    assert depth0.dtype == np.float32
                    depth0 *= sceneScaleSecond
                    raise ValueError("Please make sure whether or not depth0 is zbuffer or euclidean depth, and then set the value of zbuffer0")
                elif datasetConf["loadDepthTag"] == "NeuS":
                    depth_fn = self.projRoot + "v/misc/neusdepth/capture7jf/case%d/%03d/%03d_%03d.pkl" % (
                        caseID, groupID * 40, groupID * 40, viewID,
                    )
                    assert os.path.isfile(depth_fn), depth_fn
                    with open(depth_fn, "rb") as f:
                        zbuffer0 = pickle.load(f)["zbufferDepth0"] * datasetConf["sceneScaleSecond"]
                    zbuffer0[np.isfinite(zbuffer0) == 0] = 0
                else:
                    raise ValueError("Unknown datasetConf['loadDepthTag']: %s" % datasetConf["loadDepthTag"])

            samples.update({
                "rays": rays,
                "hdrs": hdr0.reshape((-1, 3)),
                "rgbs": rgb0.reshape((-1, 3)),
                "c2w": c2w0,
                "valid_mask": mask0.reshape((-1,)),
                "index": np.array(idx, dtype=np.int32),
                "flagSplit": self.flagSplit[(groupID * 20 + viewID - 1) * self.nOlat + 0],  # "+0": only check for the first light flag
                "dataset": dataset,
                "winWidth": np.array(winWidth, dtype=np.int32),
                "winHeight": np.array(winHeight, dtype=np.int32),
                "lgtE": np.tile(lgtE0[None, :], (B, 1)),
                "objCentroid": np.tile(objCentroid0[None, :], (B, 1)),
                # "depth": depth0.reshape((-1,)),
                "zbuffer_to_euclidean_ratio": zbuffer_to_euclidean_ratio,
                # all the IDs, for visual purpose
                "groupViewID": groupViewID,
                "groupID": groupID,
                "viewID": viewID,
                "ccID": ccID,
            })
            if datasetConf.get("ifLoadDepth", False):
                samples["scaledDepth"] = zbuffer0.reshape((-1,))

            # distill-special readjustment
            if bucket == "lfdistill":  # Now "distill" is used for benchmarking
                samples["valid_mask_original"] = np.copy(samples["valid_mask"])
                valid_mask = samples["valid_mask_original"]
                for k in ["rays", "lgtE", "objCentroid"]:
                    samples[k] = samples[k][valid_mask, :]
                for k in ["zbuffer_to_euclidean_ratio", "depth", "valid_mask"]:
                    if k in samples.keys():
                        samples[k] = samples[k][valid_mask]

            assert "envmap" not in samples.keys()  # use either envmap0 or samples
            return samples

        # need ground truth
        elif datasetConf["bucket"].startswith("ol3t"):
            # This branch is implemented for both "ol3t" and "ol3tenvmap"
            # Assumes that there are labels and the bucket file exists

            bucket = datasetConf["bucket"]
            samples = {}
            if ("Envmap" not in bucket):  # the "ol3t" case
                groupViewID = idx // self.nOlat
                ccID = idx % self.nOlat
            else:  # the "ol3tenvmap" case - envmap is major, olat is minor
                groupViewID = idx // (self.nOlat * self.nEnvmap)
                envmapccID = idx % (self.nOlat * self.nEnvmap)
                envmapID = envmapccID // self.nOlat
                ccID = envmapccID % self.nOlat
                # load the envmap
                envmap0 = self.envmapLgtDatasetCache.readEnvmap(
                    self.envmapLgtDataset, envmapID)
                assert envmap0.shape == (512, 1024, 3), envmap0.shape
                samples["envmap0"] = envmap0
            groupID = groupViewID // 20
            viewID = groupViewID % 20 + 1
            olatID_forVisualPurposeOnly = self.olatIDchosen[ccID]

            # rays (Notes in single image mode, we do not do background pad, even if datasetConf specifies pad)
            focalLengthWidth = self.focalLengthWidth[groupViewID]
            focalLengthHeight = self.focalLengthHeight[groupViewID]
            c2w0 = self.c2w[groupViewID]
            rowID = np.tile(np.arange(winHeight).astype(np.int32)[:, None], (1, winWidth)).reshape(-1)
            colID = np.tile(np.arange(winWidth).astype(np.int32), (winHeight,))
            directions_unnormalized_cam = np.stack([
                (colID.astype(np.float32) + 0.5 - winWidth / 2.0) / focalLengthWidth,
                (rowID.astype(np.float32) + 0.5 - winHeight / 2.0) / focalLengthHeight,
                np.ones((rowID.shape[0],), dtype=np.float32),
            ], 1)
            assert np.all(directions_unnormalized_cam[:, 2] == 1)
            zbuffer_to_euclidean_ratio = np.linalg.norm(directions_unnormalized_cam, ord=2, axis=1)
            directions_unnormalized = (c2w0[None, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
            directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
            zbuffer_to_euclidean_ratio = directions_norm
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
                np.nan * np.zeros((rowID.shape[0], 1), dtype=np.float32),
                zbuffer_to_euclidean_ratio[:, None],
                np.zeros((rowID.shape[0], 3), dtype=np.float32),
            ], 1)

            # load
            with open(self.projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                    "capture7jf", sceneTag, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                tmp = pickle.load(f)
                betterMask3 = tmp["betterMask3"]
                zonemap = tmp["zonemap"]
            if datasetConf["singleImageForegroundZoneNumberUpperBound"] == -1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == -1
                mask0 = betterMask3
            elif datasetConf["singleImageForegroundZoneNumberUpperBound"] >= 1:
                assert datasetConf["singleImageBackgroundZoneNumberLowerBound"] == datasetConf["singleImageForegroundZoneNumberUpperBound"] + 1
                mask0 = zonemap <= datasetConf["singleImageForegroundZoneNumberUpperBound"]
            else:
                raise ValueError("Value of singleImageForegroundZoneNumberUpperBound can only be -1, 1, 2, 3, 4, but here is %d" % (
                    datasetConf["singleImageForegroundZoneNumberUpperBound"]
                ))
            bucket_loading = bucket if not ("Envmap" in bucket) else (
                bucket[:int(bucket.find("Envmap"))]
            )
            with open(self.projRoot + "v/misc/%s/%s/bucket_%s/%03d/%03d_%03d.pkl" % (
                    "capture7jf", sceneTag, bucket_loading, 40 * groupID, 40 * groupID, viewID), "rb") as f:
                tmp = pickle.load(f)
                tmp["ol3ts"] *= datasetConf["hdrScaling"]
                olats = tmp["ol3ts"]
                lgtEs = (tmp["lgtEs"] - sceneShiftFirst[None, :]) * sceneScaleSecond
                objCentroid0 = (tmp["objCentroid0"] - sceneShiftFirst) * sceneScaleSecond
            r = np.zeros((winHeight, winWidth), dtype=np.float32)
            g = np.zeros((winHeight, winWidth), dtype=np.float32)
            b = np.zeros((winHeight, winWidth), dtype=np.float32)
            r[zonemap <= 4] = olats[:, ccID, 0]
            g[zonemap <= 4] = olats[:, ccID, 1]
            b[zonemap <= 4] = olats[:, ccID, 2]
            r[mask0 == 0] = 0
            g[mask0 == 0] = 0
            b[mask0 == 0] = 0
            hdr0 = np.stack([r, g, b], 2)
            rgb0 = np.zeros_like(hdr0)
            lgtE0 = lgtEs[ccID, :]
            omegaInput0 = objCentroid0 - lgtE0
            omegaInput0_norm = float((omegaInput0**2).sum() ** 0.5)
            assert omegaInput0_norm > 1.0e-4
            omegaInput0 = omegaInput0 / omegaInput0_norm
            B = rays.shape[0]

            # depth
            if datasetConf.get("ifLoadDepth", False):
                if datasetConf["loadDepthTag"] == "neurecon":
                    raise NotImplementedError("Have not yet checked.")
                    depth_fn = projRoot + "v/misc/real_capture_surfacenerf_results/%s/capture7jf_scene_%d/groupID_%d/sceneID_%d_groupID_%d_viewID_%d_winSize_2048.pkl" % (
                        "real_capture_neureconwm_results", sceneID, groupID, sceneID, groupID, viewID,
                    )
                    assert os.path.isfile(depth_fn), depth_fn
                    with open(depth_fn, "rb") as f:
                        depth0 = pickle.load(f)["depth0"]  # Make sure to remember that this depth is zbuffer depth
                elif datasetConf["loadDepthTag"] == "NeuS":
                    depth_fn = self.projRoot + "v/misc/neusdepth/capture7jf/case%d/%03d/%03d_%03d.pkl" % (
                        caseID, groupID * 40, groupID * 40, viewID,
                    )
                    assert os.path.isfile(depth_fn), depth_fn
                    with open(depth_fn, "rb") as f:
                        zbuffer0 = pickle.load(f)["zbufferDepth0"] * datasetConf["sceneScaleSecond"]
                    zbuffer0[np.isfinite(zbuffer0) == 0] = 0
                    depth0 = zbuffer0
                else:
                    raise ValueError("Unknown datasetConf['loadDepthTag']: %s" % datasetConf["loadDepthTag"])
                assert depth0.shape == (self.winHeight, self.winWidth)
                assert depth0.dtype == np.float32
                depth0 *= sceneScaleSecond
            else:
                depth0 = np.zeros_like(rgb0[:, :, 0])

            # highlight and sublight
            fn_highlight = self.projRoot + "v/misc/capture7jf/%s/bucket_%s/%03d/%03d_%03d.pkl" % (
                sceneTag, "highlight331b", 40 * groupID, 40 * groupID, viewID,
            )
            assert os.path.isfile(fn_highlight), fn_highlight
            with open(fn_highlight, "rb") as f:
                pkl_highlight = pickle.load(f)
            zonemap1234 = (zonemap < 5).reshape(-1)
            pixelID1234 = self.meta["pixelMapCoordinates"]["pixelIDFull"][zonemap1234]
            i_highlight = np.where(pkl_highlight["bb_highlight"] + 1 == olatID_forVisualPurposeOnly)[0]
            pixelID_highlight = pixelID1234[pkl_highlight["aa_highlight"][i_highlight]]
            i_sublight = np.where(pkl_highlight["bb_sublight"] + 1 == olatID_forVisualPurposeOnly)[0]
            pixelID_sublight = pixelID1234[pkl_highlight["aa_sublight"][i_sublight]]

            samples.update({
                "rays": rays,
                "hdrs": hdr0.reshape((-1, 3)),
                "rgbs": rgb0.reshape((-1, 3)),
                "c2w": c2w0,
                "valid_mask": mask0.reshape((-1,)),
                "index": np.array(idx, dtype=np.int32),
                "flagSplit": self.flagSplit[(groupID * 20 + viewID - 1) * self.nOlat + 0],  # "+0": only check for the first light flag
                "dataset": dataset,
                "winWidth": np.array(winWidth, dtype=np.int32),
                "winHeight": np.array(winHeight, dtype=np.int32),
                "lgtE": np.tile(lgtE0[None, :], (B, 1)),
                "objCentroid": np.tile(objCentroid0[None, :], (B, 1)),
                "omegaInput": np.tile(omegaInput0[None, :], (B, 1)),
                "depth": depth0.reshape((-1,)),
                "zbuffer_to_euclidean_ratio": zbuffer_to_euclidean_ratio,
                # all the IDs, for visual purpose
                "groupViewID": groupViewID,
                "groupID": groupID,
                "viewID": viewID,
                "ccID": ccID,
                "olatID": olatID_forVisualPurposeOnly,

                "pixelID_highlight": pixelID_highlight,
                "pixelID_sublight": pixelID_sublight,
            })
            samples["scaledDepth"] = samples["depth"]

            assert "envmap" not in samples.keys()  # use either envmap0 or samples
            return samples

        else:
            raise NotImplementedError("Unknown bucket: %s" % datasetConf["bucket"])
