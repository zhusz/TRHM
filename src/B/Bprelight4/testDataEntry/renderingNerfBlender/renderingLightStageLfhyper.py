import torch
import torchvision
import os
import numpy as np
import pickle
import sys
import math
import cv2
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../"
sys.path.append(projRoot + "src/versions/")
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb_np


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


class OnlineAugmentableEnvmapDatasetCache(object):
    # Note: this datasetCache is a part of the rendering dataset below (for easier management when benchmarking)
    def __init__(self, lgtDataset, **kwargs):
        self.lgtDataset = lgtDataset
        self.projRoot = kwargs["projRoot"]
        self.cudaDevice = kwargs["cudaDevice"]
        self.if_lgt_load_highres = kwargs["if_lgt_load_highres"]
        A0_fn = self.projRoot + "v/A/%s/A0_main.pkl" % self.lgtDataset
        assert os.path.isfile(A0_fn), A0_fn
        with open(A0_fn, "rb") as f:
            self.A0 = pickle.load(f)

        # assert np.all(self.A0["winWidth"] == 512)
        # assert np.all(self.A0["winHeight"] == 256)
        # self.winWidth, self.winHeight = 512, 256
        assert np.all(self.A0["winHeight"] * 2 == self.A0["winWidth"])
        assert len(np.unique(self.A0["winWidth"])) == 1
        assert len(np.unique(self.A0["winHeight"])) == 1
        self.winWidth, self.winHeight = int(self.A0["winWidth"][0]), int(self.A0["winHeight"][0])

        self.m = int(self.A0["m"])

        self.indTrain = np.where(self.A0["flagSplit"] == 1)[0]
        self.mTrain = int(self.indTrain.shape[0])
        self.indVal = np.where(self.A0["flagSplit"] == 2)[0]
        self.mVal = int(self.indVal.shape[0])
        self.indTest = np.where(self.A0["flagSplit"] == 3)[0]
        self.mTest = int(self.indTest.shape[0])
        self.envmap_hdrs = np.zeros((self.m, self.winHeight, self.winWidth, 3), dtype=np.float32)
        self.envmap_flags = np.zeros((self.m, ), dtype=bool)
        self.envmap_total_flag = False

        # quantile_cut
        self.quantile_cut_min = kwargs["quantile_cut_min"]
        self.quantile_cut_max = kwargs["quantile_cut_max"]
        self.quantile_cut_fixed = kwargs["quantile_cut_fixed"]
        
        self.sin_theta_thgpu = torch.from_numpy(self._getSinTheta()).float().to(self.cudaDevice)

    def _getSinTheta(self):
        delta = np.pi / 16
        theta_i = np.linspace(delta / 2, np.pi - delta / 2, 16)
        sin_theta_i = np.sin(theta_i).astype(np.float32)
        return sin_theta_i

    def _readRawEnvmap(self, index, if_highres=False):  # this class method does not include cache
        A0_lgt = self.A0
        if if_highres:  # this generally happens during the demo phase
            fn = projRoot + A0_lgt["nameListOriginalHighRes"][index]
        else:
            fn = projRoot + A0_lgt["nameList"][index]
        assert os.path.isfile(fn), fn
        tmp = cv2.imread(fn, flags=cv2.IMREAD_UNCHANGED)
        assert tmp.dtype == np.float32
        rgb = np.stack([tmp[:, :, 2], tmp[:, :, 1], tmp[:, :, 0]], 2)
        return rgb

    def _doAugment(self, hdrs):
        assert (len(hdrs.shape) == 4) and (hdrs.shape[1] == self.winHeight) and (
            hdrs.shape[2] == self.winWidth) and (hdrs.shape[3] == 3), hdrs.shape
        assert type(hdrs) == torch.Tensor

        """ alpha should be done only after the resizing
        # alpha values
        alpha_min, alpha_max = [0.4, 1.6]
        alpha = np.random.rand(hdrs.shape[0],).astype(np.float32) * (alpha_max - alpha_min) + alpha_min
        hdrs *= alpha[:, None, None, None]
        """

        # flip (for the sake of speed, here we do either all flip or all not-flipped)
        flip = float(np.random.rand(1,)) < 0.5
        if flip:
            # hdrs = hdrs[:, :, ::-1, :]
            assert len(hdrs.shape) == 4
            hdrs = torch.flip(hdrs, dims=(2,))

        # phi_shift (for the sake of speed, here we phi-shift the same number of pixels for all the samples)
        phi_shift = int(float(np.random.rand(1,)) * self.winWidth)
        phi_shift = max(0, phi_shift)
        phi_shift = min(self.winWidth - 1, phi_shift)
        if phi_shift > 0:
            hdrs = torch.cat([
                hdrs[:, :, phi_shift:, :],
                hdrs[:, :, :phi_shift, :],
            ], 2)

        # theta_shift (for the sake of speed, here we theta-shift the same number of pixels for all the samples)
        theta_shift_bound = int(self.winHeight * (2.0 / 16.0))
        theta_shift = int(np.random.randint(2 * theta_shift_bound + 1, size=(1,))) - theta_shift_bound
        assert -theta_shift_bound <= theta_shift <= theta_shift_bound
        if theta_shift > 0:  # going down
            top_row_hdr_mean = hdrs[:, 0, :, :].mean(1)
            hdrs[:, theta_shift:, :, :] = hdrs[:, :-theta_shift, :, :].detach().clone()
            hdrs[:, :theta_shift, :, :] = top_row_hdr_mean[:, None, None, :]
        elif theta_shift < 0:
            bot_row_hdr_mean = hdrs[:, -1, :, :].mean(1)
            hdrs[:, :-(-theta_shift), :, :] = hdrs[:, (-theta_shift):, :, :].detach().clone()
            hdrs[:, -(-theta_shift):, :, :] = bot_row_hdr_mean[:, None, None, :]
        else:
            assert theta_shift == 0

        augment_info = {
            # "alpha": alpha,
            "flip": flip,
            "phi_shift": phi_shift,
            "theta_shift": theta_shift,
        }

        return hdrs, augment_info

    def queryEnvmap(self, ind, if_augment, **kwargs):
        if_return_all = kwargs["if_return_all"]
        cudaDevice = self.cudaDevice

        if not self.envmap_total_flag:
            for j in ind.tolist():
                if not self.envmap_flags[j]:
                    """
                    print("Cache Reading lgtDataset %s %d (%d / %d)" % (
                        self.lgtDataset,
                        j,
                        self.envmap_flags.sum(),
                        self.A0["m"],
                    ))
                    """
                    self.envmap_hdrs[j, :, :, :] = self._readRawEnvmap(j)
                    self.envmap_flags[j] = True
            if np.all(self.envmap_flags):
                self.envmap_total_flag = True

        """ What the hell is this ...
        envmap = np.zeros((ind.shape[0], self.winHeight, self.winWidth, 3), dtype=np.float32)
        for i, j in enumerate(ind.tolist()):
            envmap[i, :, :, :] = self.envmap_hdrs[j, :, :, :]
        """
        # And also we are changing to all thgpu for queryEnvmap
        envmap = torch.from_numpy(self.envmap_hdrs[ind, :, :, :]).float().to(cudaDevice)
        
        if if_augment:
            envmap_augmented, augment_info = self._doAugment(envmap)
        else:
            envmap_augmented = envmap
            augment_info = {
                "flip": False,
                "phi_shift": 0,
                "theta_shift": 0,
            }

        # do quantile cut
        if if_augment:
            q_min, q_max = self.quantile_cut_min, self.quantile_cut_max
            q = float(np.random.rand(1)) * (q_max - q_min) + q_min  # for the sake of speed, we use the same q for all the samples in the same batch
            augment_info["q"] = q
        else:
            # assert self.quantile_cut_min == self.quantile_cut_max, (self.quantile_cut_min, self.quantile_cut_max)
            # q = (self.quantile_cut_min + self.quantile_cut_max) / 2.0
            q = self.quantile_cut_fixed
            augment_info["q"] = q
        quantile = torch.quantile(
            envmap_augmented.reshape((ind.shape[0], self.winWidth * self.winHeight * 3)), q, dim=1)
        msk = envmap_augmented > quantile[:, None, None, None]
        envmap_quantiled = envmap_augmented.detach().clone()
        envmap_quantiled[msk] = quantile[:, None, None, None].repeat(1, self.winHeight, self.winWidth, 3)[msk]

        # resize to 16x32
        envmap_resized = torchvision.transforms.functional.resize(
            envmap_quantiled.permute(0, 3, 1, 2),
            size=(16, 32),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=False,
        ).permute(0, 2, 3, 1)

        # sin_theta apply
        envmap_sined = envmap_resized * self.sin_theta_thgpu[None, :, None, None]

        # normalize to make uniform sum
        # During testing, you must feed in the normalized hdr map.
        # You then multiple directly to the prediction result with the normalizing factor.
        envmap_sined_sum = (envmap_sined.reshape((ind.shape[0], 16 * 32 * 3)).sum(1)).float()
        envmap_normalized = (envmap_sined * (16 * 32 / envmap_sined_sum[:, None, None, None])).float()

        if if_return_all:

            # compute envmap_original_normalized, that will be directly useful (used by) the hdr3 predictions.
            envmap_original = envmap
            envmap_original_normalized = (envmap_original * (16 * 32 / envmap_sined_sum[:, None, None, None])).float()
            # envmap_original_normalized.mean() is slightly larger than envmap_normalized.mean() because
            #   it did not undergo the *self.sin_theta_thgpu process
            # Since the normalizing factor of envmap_original_normalized is not related to envmap_original at all,
            #   it does not matter whether or not to clip envmap_original's large value (no need to quantile)

            print("RenderingLightStageLfhyper-OnlineAugmentableEnvmapDatasetCache: Reading from the highres envmap (This only happens during testing!) - %s" % (
                projRoot + self.A0["nameListOriginalHighRes"][ind[0]]
            ))
            if self.if_lgt_load_highres:
                assert ind.shape == (1,)
                envmap_original_highres = torch.from_numpy(self._readRawEnvmap(ind[0], if_highres=True)[None, :, :, :]).float().to(envmap_original.device)
            else:
                envmap_original_highres = envmap_original.detach().clone()

            if self.lgtDataset in ["lgtEnvolatGray1k", "lgtEnvolatGray64x128", "lgtEnvolatGray128x256"]:
                omega = self.A0["omega"][ind, :]
                omega = np.stack([-omega[:, 0], -omega[:, 1], omega[:, 2]], 1)
            else:
                omega = np.nan * np.zeros((ind.shape[0], 3), dtype=np.float32)

            return {  # all thgpu except for augment_info
                "envmap_normalized": envmap_normalized,
                "envmap_normalizing_factor": envmap_sined_sum,
                "envmap_sined": envmap_sined,
                "envmap_resized": envmap_resized,
                "envmap_quantiled": envmap_quantiled,
                "envmap_augmented": envmap_augmented,
                "augment_info": augment_info,
                "envmap_original_highres": envmap_original_highres,
                "envmap_original": envmap_original,
                "envmap_original_normalized": envmap_original_normalized,
                "envmap_pointlight_omega": omega,
            }
        else:
            return envmap_normalized  # thgpu


class RenderingLightStageLfhyper(object):
    def __init__(self, datasetConf, **kwargs):
        self.datasetConf = datasetConf
        self.projRoot = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../../") + "/"
        self.cudaDevice = kwargs["cudaDevice"]

        # maintain the multi-processing nature, as we are likely to use multi-gpu for training this
        self.rank = int(os.environ.get("RANK", 0))
        self.numMpProcess = int(os.environ.get("WORLD_SIZE", 0))

        split = datasetConf["split"]  # train, val
        singleImageMode = datasetConf["singleImageMode"]
        if split in ["val", "test"]:
            assert singleImageMode

        self.split = split
        self.singleImageMode = singleImageMode

        # preset value
        self.winWidth = 1366
        self.winHeight = 2048

        # # default from datasetConf
        pass

        # A0 and A0b
        dataset = datasetConf["dataset"]
        with open(self.projRoot + "v/A/%s/A0_fromBlender.pkl" % "capture7jf", "rb") as f:
            self.A0 = pickle.load(f)
        with open(self.projRoot + "v/A/%s/A0b_precomputation.pkl" % "capture7jf", "rb") as f:
            self.A0b = pickle.load(f)

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
        self.objCentroidCache = ObjCentroidCache()

        self.setupFlagSplit()

        self.generateCamerasAndRays()

        if not self.singleImageMode:
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
        # indexing rules:   
        # m = 180 * (nEnvmap) (nEnvmap determined by the lgtDataset, whose m_lgt is 2330)
        # groupViewID major, envmapID minor
        # Strong augmentation is applied on the fly for training samples
        # meaning for the same indexing during training, very different samples (envmap) can be generated

        # split:
        # Only when both flagSplit_groupView and flagSplit_lgt is 1, then flagSplit is 1.
        # Only when both flagSplit_groupView and flagSplit_lgt is 2, then flagSplit is 2.
        # All other cases would have flagSplit to be 0
        # Neither flagSplit_groupView nor flagSplit_lgt should have any values to be 3.
        datasetConf = self.datasetConf
        sceneID = int(datasetConf["caseID"])

        flagSplit_lgt = np.tile(
            self.lgtDatasetCache.A0["flagSplit"], (180,)
        )
        m_lgt = int(self.lgtDatasetCache.A0["m"])
        assert self.lgtDatasetCache.A0["flagSplit"].shape == (m_lgt,)
        flagSplit_groupView = np.zeros((180,), dtype=np.int32)
        for j in range(180):
            groupID = j // 20
            viewID = j % 20 + 1
            flagRemove1 = (sceneID == 1) and (
                (groupID in [7, 8]) or ((groupID == 6) and (viewID == 1))
            )
            flagRemove10 = (sceneID == 10) and (groupID == 1) and (viewID == 6)
            flagRemove = flagRemove1 or flagRemove10
            if flagRemove:
                flagSplit_groupView[j] = 0
            else:
                flagSplit_groupView[j] = 1 if viewID <= 18 else 2
        flagSplit_groupView = np.tile(flagSplit_groupView[:, None], (1, m_lgt)).reshape((m_lgt * 180,))

        if flagSplit_lgt.max() == 3:  # we assume this is the demo mode only
            assert self.singleImageMode == True  # only demo mode
            assert np.all(np.isin(flagSplit_lgt, np.array([0, 1, 2, 3], dtype=np.int32)))
            assert np.all(np.isin(flagSplit_groupView, np.array([0, 1, 2], dtype=np.float32)))
            m = 180 * m_lgt
            flagSplit = np.zeros((m,), dtype=np.int32)
            flagSplit[(flagSplit_groupView == 1) & (flagSplit_lgt == 1)] = 1
            flagSplit[(flagSplit_groupView == 2) & (flagSplit_lgt == 2)] = 2
            flagSplit[(flagSplit_groupView == 2) & (flagSplit_lgt == 3)] = 3
        else:
            assert np.all(np.isin(flagSplit_lgt, np.array([0, 1, 2], dtype=np.int32)))
            assert np.all(np.isin(flagSplit_groupView, np.array([0, 1, 2], dtype=np.float32)))
            m = 180 * m_lgt
            flagSplit = np.zeros((m,), dtype=np.int32)
            flagSplit[(flagSplit_groupView == 1) & (flagSplit_lgt == 1)] = 1
            flagSplit[(flagSplit_groupView == 2) & (flagSplit_lgt == 2)] = 2
        
        self.flagSplit = flagSplit
        
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
                (np.arange(self.flagSplit.shape[0]) // m_lgt >= self.datasetConf.get("debugReadTwoHowMany", 2))
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
        self.m = 180 * m_lgt
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

    def getOneNP(self, index, cudaDevice="cuda:0"):
        assert self.singleImageMode
        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        caseID = datasetConf["caseID"]
        sceneID = caseID
        sceneTag = self.A0[sceneID]["sceneTag"]
        winWidth = self.winWidth
        winHeight = self.winHeight

        lgtDatasetCache = self.lgtDatasetCache
        m_lgt = lgtDatasetCache.m
        m_groupView = 180
        m = self.m
        """
        nOlat = 16 * 32
        """
        nOlat = 331
        """"""
        assert m_lgt * m_groupView == m
        groupViewID = int(index // m_lgt)
        lgtID = int(index % m_lgt)
        groupID = groupViewID // 20
        viewID = groupViewID % 20 + 1

        # load the envmap
        envmap_full = lgtDatasetCache.queryEnvmap(
            np.array([lgtID], dtype=np.int32),
            if_augment=False,
            if_return_all=True,
        )

        # load the rays (no need to consider backgorund_padding)
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

        # combine to the label hdr
        betterMask3_fn = projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
            "capture7jf", sceneTag, 40 * groupID, 40 * groupID, viewID,
        )
        assert os.path.isfile(betterMask3_fn), betterMask3_fn
        print("BetterMask3 Testtime Loading: groupViewID = %d / %d" % (groupViewID, 180))
        with open(betterMask3_fn, "rb") as f:
            # betterMask3 = pickle.load(f)["betterMask3"]
            tmp = pickle.load(f)
            betterMask3 = tmp["betterMask3"]
            zonemap = tmp["zonemap"]
        del tmp

        # ol3t331
        ol3t331_fn = projRoot + "v/misc/capture7jf/%s/bucket_ol3t331/%03d/%03d_%03d.pkl" % (
            sceneTag, 40 * groupID, 40 * groupID, viewID,
        )
        assert os.path.isfile(ol3t331_fn), ol3t331_fn
        print("        Lfhyper Testing Loading: %s" % ol3t331_fn)
        with open(ol3t331_fn, "rb") as f:
            pkl = pickle.load(f)
        lgtEs = (pkl["lgtEs"] - sceneShiftFirst[None, :]) * sceneScaleSecond
        objCentroid0 = (pkl["objCentroid0"] - sceneShiftFirst) * sceneScaleSecond
        omegas_unnormalized = lgtEs - objCentroid0[None, :]
        omegas = omegas_unnormalized / np.linalg.norm(omegas_unnormalized, ord=2, axis=1)[:, None]
        del lgtEs, objCentroid0, omegas_unnormalized
        ol3t331 = pkl["ol3ts"]  # with lens-flare corruption  # not yet timed with the datasetConf["hyperHdrScaling"]
        q = pkl["q"]
        del pkl
        Q = ((q > 0.005) & np.isfinite(q)).sum()
        if Q > 0:
            lfdistill_fn = projRoot + "v/misc/lfdistill/case%d/%03d/%03d_%03d.pkl" % (
                caseID, 40 * groupID, 40 * groupID, viewID,
            )
            assert os.path.isfile(lfdistill_fn), lfdistill_fn
            with open(lfdistill_fn, "rb") as f:
                pkl = pickle.load(f)
            assert np.all(pkl["valid_mask"] == betterMask3.reshape(-1,))
            hdr_masked = pkl["hdr2_masked"] + pkl["hdr3_masked"]
            del pkl
        
            assert hdr_masked.shape == (betterMask3.sum(), Q, 3)
            tmp = np.tile(betterMask3.reshape((winWidth * winHeight,))[:, None, None], (1, Q, 3))
            hdr_12345 = np.zeros((winWidth * winHeight, Q, 3), dtype=np.float32)
            hdr_12345[tmp] = hdr_masked.reshape(-1)
            del tmp
            hdr_1234 = hdr_12345[np.where(zonemap.reshape(-1) < 5)[0], :, :]
            ol3t331[:, np.where(q > 0.005)[0], :] = hdr_1234
            del ol3t331_fn, q, lfdistill_fn, hdr_masked, Q, hdr_12345, hdr_1234
        ol3t331 *= datasetConf["hyperHdrScaling"]
        """"""

        assert envmap_full["envmap_normalized"].shape == (1, 16, 32, 3)

        envmap_to_record = envmap_full["envmap_normalized"][0, :, :, :].float()  # (16, 32, 3)
        envmap_to_apply = envmap_to_record / 16 / 32 * 3  # normalized by a constant factor, to make its sum to be exactly 1
        envmap_to_apply = envmap_to_apply * 1  # alpha is assumed to be 1
        envmap_to_apply = envmap_to_apply.reshape(16 * 32, 3)

        # Use omegas to compute the phi and theta, and grid_sample the envmap response
        phi = np.arctan2(omegas[:, 1], omegas[:, 0])  # (-np.pi, np.pi)
        phi = np.pi - phi  # [0, 2 * np.pi]
        phi_ = phi / np.pi - 1  # [-1, 1]
        assert np.all(np.abs(omegas[:, 2]) < 1.00001)
        theta = np.arccos(np.clip(omegas[:, 2], a_min=-1, a_max=1))  # [0, np.pi]
        theta_ = theta / np.pi * 2 - 1  # [-1, 1]
        tmp = F.grid_sample(
            envmap_to_apply.permute(1, 0).view(1, 3, 16, 32),
            torch.from_numpy(
                np.stack([phi_, theta_], 1)[None, :, None, :]
            ).float().to(envmap_to_apply.device),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, :, :, 0].permute(1, 0)  # (331, 3)  # Now this (331, 3)-tmp shall replace the (512, 3)-envmap_to_apply
        ol3t331_thgpu = torch.from_numpy(ol3t331).float().to(cudaDevice)
        olat_envmap_applied = (ol3t331_thgpu * tmp[None, :, :]).sum(1).detach().cpu().numpy()

        olat_envmap_applied = olat_envmap_applied * 1  # alpha is 1 (if_augment is False)
        envmap_alphaed = envmap_to_record.reshape((16, 32, 3))
        hdrs = np.zeros((winHeight * winWidth, 3), dtype=np.float32)
        hdrs[(zonemap < 5).reshape(winWidth * winHeight), :] = olat_envmap_applied
        rgbs = np.clip(tonemap_srgb_to_rgb_np(hdrs), a_min=0, a_max=1)

        samples = {
            "rays": rays,
            "hdrs": hdrs,
            "rgbs": rgbs,
            "c2w": c2w0,
            "valid_mask": betterMask3.reshape(winWidth * winHeight),
            "index": np.array(index, dtype=np.int32),
            "flagSplit": self.flagSplit[index],
            "dataset": dataset,
            "winWidth": np.array(winWidth, dtype=np.int32),
            "winHeight": np.array(winHeight, dtype=np.int32),
            "groupViewID": groupViewID,
            "groupID": groupID,
            "viewID": viewID,
            "lgtDataset": datasetConf["lgtDataset"],
            "lgtID": lgtID,
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

    def doMemoryBankReadOutOnce(self, specifiedGroupViewID):
        if specifiedGroupViewID is None:  # pick the most wanted groupViewID
            unique, counts = np.unique(self.all_fg_groupViewID, return_counts=True)
            zeroOccuranceGroupViewID = np.setdiff1d(self.uniqueTrainingGroupViewIDList, unique)
            if zeroOccuranceGroupViewID.shape[0] > 0:
                t = int(np.random.randint(zeroOccuranceGroupViewID.shape[0], size=(1,)))
                specifiedGroupViewID = zeroOccuranceGroupViewID[t]
            else:
                specifiedGroupViewID = int(unique[int(np.argsort(counts)[0])])
        else:
            assert type(specifiedGroupViewID) is int
            assert specifiedGroupViewID in self.uniqueTrainingGroupViewIDList, (
                self.uniqueTrainingGroupViewIDList, specifiedGroupViewID
            )
        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        caseID = datasetConf["caseID"]
        sceneTag = self.A0[caseID]["sceneTag"]
        groupViewID = specifiedGroupViewID
        groupID = groupViewID // 20
        viewID = groupViewID % 20 + 1

        winWidth, winHeight = self.winWidth, self.winHeight

        betterMask3_fn = projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
            "capture7jf", sceneTag, 40 * groupID, 40 * groupID, viewID,
        )
        assert os.path.isfile(betterMask3_fn), betterMask3_fn
        print("BetterMask3 Distill Readouts Loading: groupViewID = %d / %d" % (groupViewID, 180))
        with open(betterMask3_fn, "rb") as f:
            tmp = pickle.load(f)
            betterMask3 = tmp["betterMask3"]
            zonemap = tmp["zonemap"]
            del tmp

        # Direct copy
        ol3t331_fn = projRoot + "v/misc/capture7jf/%s/bucket_ol3t331/%03d/%03d_%03d.pkl" % (
            sceneTag, 40 * groupID, 40 * groupID, viewID,
        )
        assert os.path.isfile(ol3t331_fn), ol3t331_fn
        print("        initializeMemoryBank Testing Loading: %s" % ol3t331_fn)
        with open(ol3t331_fn, "rb") as f:
            pkl = pickle.load(f)
        lgtEs = (pkl["lgtEs"] - datasetConf["sceneShiftFirst"][None, :]) * datasetConf["sceneScaleSecond"]
        objCentroid0 = (pkl["objCentroid0"] - datasetConf["sceneShiftFirst"]) * datasetConf["sceneScaleSecond"]
        omegas_unnormalized = lgtEs - objCentroid0[None, :]
        omegas = omegas_unnormalized / np.linalg.norm(omegas_unnormalized, ord=2, axis=1)[:, None]
        del lgtEs, objCentroid0, omegas_unnormalized
        ol3t331 = pkl["ol3ts"]  # with lens-flare corruption  # not yet timed with the datasetConf["hyperHdrScaling"]
        q = pkl["q"]
        del pkl
        Q = ((q > 0.005) & np.isfinite(q)).sum()
        if Q > 0:
            lfdistill_fn = projRoot + "v/misc/lfdistill/case%d/%03d/%03d_%03d.pkl" % (
                caseID, 40 * groupID, 40 * groupID, viewID,
            )
            assert os.path.isfile(lfdistill_fn), lfdistill_fn
            with open(lfdistill_fn, "rb") as f:
                pkl = pickle.load(f)
            assert np.all(pkl["valid_mask"] == betterMask3.reshape(-1,))
            hdr_masked = pkl["hdr2_masked"] + pkl["hdr3_masked"]
            del pkl
        
            assert hdr_masked.shape == (betterMask3.sum(), Q, 3)
            tmp = np.tile(betterMask3.reshape((winWidth * winHeight,))[:, None, None], (1, Q, 3))
            hdr_12345 = np.zeros((winWidth * winHeight, Q, 3), dtype=np.float32)
            hdr_12345[tmp] = hdr_masked.reshape(-1)
            del tmp
            hdr_1234 = hdr_12345[np.where(zonemap.reshape(-1) < 5)[0], :, :]
            ol3t331[:, np.where(q > 0.005)[0], :] = hdr_1234
            del ol3t331_fn, q, lfdistill_fn, hdr_masked, Q, hdr_12345, hdr_1234
        ol3t331 *= datasetConf["hyperHdrScaling"]

        betterMask3_in_zonemap1234 = np.where(betterMask3.reshape(-1)[zonemap.reshape(-1) < 5])[0]
        ol3t331 = ol3t331[betterMask3_in_zonemap1234, :, :]

        mask0_foreground = betterMask3
        mask0_foreground_padded = np.zeros((  # the padded ones are never put to the fore-ground
            self.winHeight + 2 * datasetConf["rays_background_pad_height"],
            self.winWidth + 2 * datasetConf["rays_background_pad_width"],
        ), dtype=bool)
        mask0_foreground_padded[
            datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
            datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
        ] = mask0_foreground
        yy, xx = np.where(mask0_foreground_padded)
        m_fg_rays_now = int(mask0_foreground_padded.sum())

        assert ol3t331.shape == (m_fg_rays_now, 331, 3)
        nOlat = 331

        mbtot_per_process = int(math.ceil(float(datasetConf["mbtot"]) / max(self.numMpProcess, 1)))
        assert m_fg_rays_now < mbtot_per_process
        assert self.all_fg_hdrs.shape == (mbtot_per_process, nOlat, 3)
        assert self.all_fg_groupViewID.shape == (mbtot_per_process,)
        assert self.all_fg_pixelID.shape == (mbtot_per_process,)
        ind_replace = np.random.choice(mbtot_per_process, m_fg_rays_now, replace=False).astype(np.int32)

        self.all_fg_hdrs[ind_replace, :, :] = ol3t331
        self.all_fg_omegas[ind_replace, :, :] = omegas

        self.all_fg_groupViewID[ind_replace] = groupViewID
        self.all_fg_pixelID[ind_replace] = yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx

    def initializeTheMemoryBank(self):
        projRoot = self.projRoot
        datasetConf = self.datasetConf
        dataset = datasetConf["dataset"]
        caseID = int(datasetConf["caseID"])
        sceneTag = self.A0[caseID]["sceneTag"]

        winWidth, winHeight = self.winWidth, self.winHeight

        mbtot_per_process = int(math.ceil(float(datasetConf["mbtot"]) / max(self.numMpProcess, 1)))

        nOlat = 331

        m_lgt = int(self.lgtDatasetCache.A0["m"])

        trainingGroupViewID = np.tile(np.arange(180).astype(np.int32)[:, None], (1, m_lgt)).reshape(self.m)[self.indTrain]
        uniqueTrainingGroupViewIDList = np.unique(trainingGroupViewID)

        # bg mg and fg (fg is in the memory bank mode, which is different from vanilla olat)
        rays_bg_tot = 180 * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])  
        self.all_bg_groupViewID = np.zeros((rays_bg_tot,), dtype=np.int32)  # [0, 180)
        self.all_bg_pixelID = np.zeros((rays_bg_tot,), dtype=np.int32)
        rays_bg_count = 0

        rays_mg_tot = 180 * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) * (self.winHeight + 2 * datasetConf["rays_background_pad_height"])
        self.all_mg_groupViewID = np.zeros((rays_mg_tot,), dtype=np.int32)
        self.all_mg_pixelID = np.zeros((rays_mg_tot,), dtype=np.int32)
        rays_mg_count = 0

        self.all_fg_hdrs = np.zeros((mbtot_per_process, nOlat, 3), dtype=np.float32)

        self.all_fg_omegas = np.zeros((mbtot_per_process, nOlat, 3), dtype=np.float32)
        
        self.all_fg_groupViewID = np.zeros((mbtot_per_process,), dtype=np.int32)
        self.all_fg_pixelID = np.zeros((mbtot_per_process,), dtype=np.int32)
        rays_fg_count = 0

        perm = np.random.permutation(uniqueTrainingGroupViewIDList.shape[0]).astype(np.int32)

        # reshuffled groupViewID (for randomness for the fg memory bank mode)
        for groupViewID in uniqueTrainingGroupViewIDList[perm].tolist():
            groupID = groupViewID // 20
            viewID = groupViewID % 20 + 1
            betterMask3_fn = projRoot + "v/misc/%s/%s/betterMask3/%03d/%03d_%03d.pkl" % (
                "capture7jf", sceneTag, 40 * groupID, 40 * groupID, viewID,
            )
            assert os.path.isfile(betterMask3_fn), betterMask3_fn
            print("BetterMask3 Preloading: groupViewID = %d / %d" % (groupViewID, 180))
            with open(betterMask3_fn, "rb") as f:
                tmp = pickle.load(f)
                betterMask3 = tmp["betterMask3"]
                zonemap = tmp["zonemap"]
                del tmp
            L = datasetConf["mgPixDilation"]
            kernel = np.ones((L, L), dtype=np.float32)
            dilated = (cv2.dilate(betterMask3.astype(np.float32), kernel, iterations=1) > 0)
            dilated[betterMask3] = False

            mask0_background = betterMask3 == 0
            mask0_background_padded = np.ones((
                self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                self.winWidth + 2 * datasetConf["rays_background_pad_width"],
            ), dtype=bool)
            mask0_background_padded[
                datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
            ] = mask0_background
            yy, xx = np.where(mask0_background_padded)
            m_bg_rays_now = int(mask0_background_padded.sum())
            self.all_bg_groupViewID[rays_bg_count : rays_bg_count + m_bg_rays_now] = groupViewID
            self.all_bg_pixelID[rays_bg_count : rays_bg_count + m_bg_rays_now] =  yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx
            rays_bg_count += m_bg_rays_now
            
            mask0_middleground = dilated
            mask0_middleground_padded = np.zeros((  # the padded ones are never put to the middle ground
                self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                self.winWidth + 2 * datasetConf["rays_background_pad_width"],
            ), dtype=bool)
            mask0_middleground_padded[
                datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
            ] = mask0_middleground
            yy, xx = np.where(mask0_middleground_padded)
            m_mg_rays_now = int(mask0_middleground_padded.sum())
            self.all_mg_groupViewID[rays_mg_count : rays_mg_count + m_mg_rays_now] = groupViewID
            self.all_mg_pixelID[rays_mg_count : rays_mg_count + m_mg_rays_now] = yy * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx
            rays_mg_count += m_mg_rays_now

            if rays_fg_count < mbtot_per_process:
                # Direct copy
                ol3t331_fn = projRoot + "v/misc/capture7jf/%s/bucket_ol3t331/%03d/%03d_%03d.pkl" % (
                    sceneTag, 40 * groupID, 40 * groupID, viewID,
                )
                assert os.path.isfile(ol3t331_fn), ol3t331_fn
                print("        initializeMemoryBank Testing Loading: %s" % ol3t331_fn)
                with open(ol3t331_fn, "rb") as f:
                    pkl = pickle.load(f)
                lgtEs = (pkl["lgtEs"] - datasetConf["sceneShiftFirst"][None, :]) * datasetConf["sceneScaleSecond"]
                objCentroid0 = (pkl["objCentroid0"] - datasetConf["sceneShiftFirst"]) * datasetConf["sceneScaleSecond"]
                omegas_unnormalized = lgtEs - objCentroid0[None, :]
                omegas = omegas_unnormalized / np.linalg.norm(omegas_unnormalized, ord=2, axis=1)[:, None]
                del lgtEs, objCentroid0, omegas_unnormalized
                ol3t331 = pkl["ol3ts"]  # with lens-flare corruption  # not yet timed with the datasetConf["hyperHdrScaling"]
                q = pkl["q"]
                del pkl
                Q = ((q > 0.005) & np.isfinite(q)).sum()
                if Q > 0:
                    lfdistill_fn = projRoot + "v/misc/lfdistill/case%d/%03d/%03d_%03d.pkl" % (
                        caseID, 40 * groupID, 40 * groupID, viewID,
                    )
                    assert os.path.isfile(lfdistill_fn), lfdistill_fn
                    with open(lfdistill_fn, "rb") as f:
                        pkl = pickle.load(f)
                    assert np.all(pkl["valid_mask"] == betterMask3.reshape(-1,))
                    hdr_masked = pkl["hdr2_masked"] + pkl["hdr3_masked"]
                    del pkl
                
                    assert hdr_masked.shape == (betterMask3.sum(), Q, 3)
                    tmp = np.tile(betterMask3.reshape((winWidth * winHeight,))[:, None, None], (1, Q, 3))
                    hdr_12345 = np.zeros((winWidth * winHeight, Q, 3), dtype=np.float32)
                    hdr_12345[tmp] = hdr_masked.reshape(-1)
                    del tmp
                    hdr_1234 = hdr_12345[np.where(zonemap.reshape(-1) < 5)[0], :, :]
                    ol3t331[:, np.where(q > 0.005)[0], :] = hdr_1234
                    del ol3t331_fn, q, lfdistill_fn, hdr_masked, Q, hdr_12345, hdr_1234
                ol3t331 *= datasetConf["hyperHdrScaling"]

                betterMask3_in_zonemap1234 = np.where(betterMask3.reshape(-1)[zonemap.reshape(-1) < 5])[0]
                ol3t331 = ol3t331[betterMask3_in_zonemap1234, :, :]
                
                mask0_foreground = betterMask3
                mask0_foreground_padded = np.zeros((  # the padded ones are never put to the fore-ground
                    self.winHeight + 2 * datasetConf["rays_background_pad_height"],
                    self.winWidth + 2 * datasetConf["rays_background_pad_width"],
                ), dtype=bool)
                mask0_foreground_padded[
                    datasetConf["rays_background_pad_height"] : self.winHeight + datasetConf["rays_background_pad_height"],
                    datasetConf["rays_background_pad_width"] : self.winWidth + datasetConf["rays_background_pad_width"],
                ] = mask0_foreground
                yy, xx = np.where(mask0_foreground_padded)
                m_fg_rays_now = int(mask0_foreground_padded.sum())

                assert ol3t331.shape == (m_fg_rays_now, 331, 3)

                if m_fg_rays_now > mbtot_per_process - rays_fg_count:
                    m_fg_rays_now = mbtot_per_process - rays_fg_count

                self.all_fg_hdrs[rays_fg_count : rays_fg_count + m_fg_rays_now, :, :] = ol3t331[:m_fg_rays_now, :, :]
                self.all_fg_omegas[rays_fg_count : rays_fg_count + m_fg_rays_now, :, :] = omegas[None, :, :]

                self.all_fg_groupViewID[rays_fg_count : rays_fg_count + m_fg_rays_now] = groupViewID
                self.all_fg_pixelID[rays_fg_count : rays_fg_count + m_fg_rays_now] = yy[:m_fg_rays_now] * (self.winWidth + 2 * datasetConf["rays_background_pad_width"]) + xx[:m_fg_rays_now]
                rays_fg_count += m_fg_rays_now

        self.all_mg_groupViewID = self.all_mg_groupViewID[:rays_mg_count]
        self.all_mg_pixelID = self.all_mg_pixelID[:rays_mg_count:]
        self.all_bg_groupViewID = self.all_bg_groupViewID[:rays_bg_count]
        self.all_bg_pixelID = self.all_bg_pixelID[:rays_bg_count]

        return uniqueTrainingGroupViewIDList
