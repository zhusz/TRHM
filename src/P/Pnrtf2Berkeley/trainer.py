import copy
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict, OrderedDict
from socket import gethostname

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from torch.distributed import ReduceOp

from Bprelight4.csvPsnrEntry.benchmarkingPsnrLdrRenderingNerfBlender import (
    benchmarkingPsnrLdrRenderingNerfBlenderFunc,
)
from Bprelight4.csvPsnrEntry.dumpHtmlForPrepickRelightFine import (
    addToSummary0Txt0BrInds,
)

from Bprelight4.testDataEntry.testDataEntryPool import getTestDataEntryDict
from codes_py.toolbox_3D.mesh_io_v1 import dump_obj_np_fileObj
from codes_py.toolbox_3D.nerf_metrics_v1 import psnr
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc
from codes_py.toolbox_3D.rotations_v1 import ELU2cam

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
from codes_py.toolbox_framework.logger_v1 import LoggerDummy, Logger
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb

from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_torch.hook_v1 import PyTorchForwardHook

# from .dataset import PRenderingNerfBlenderDataset, PRenderingLightStageOrigianl2
from .losses import ColorLoss, HdrLoss, MaskLoss

# from .models.nerf import Embedding, NeRF
# from .models.rendering import render_rays
from .models.model import PRTNetwork1
from .models.hash_encoding import HashEmbedder, SHEncoder


def nanmean(x):
    return x[torch.isfinite(x)].mean()


def bt(s):
    return s[0].upper() + s[1:]


class Trainer(object):
    @staticmethod
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__

        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    @staticmethod
    def get_trailing_number(s):
        m = re.search(r"\d+$", s)
        return int(m.group()) if m else 0

    def __init__(self, config, **kwargs):
        self.rank = kwargs["rank"]
        self.numMpProcess = kwargs["numMpProcess"]
        self.ifDumpLogging = kwargs["ifDumpLogging"]
        self.trackedRank = 0 if self.numMpProcess <= 0 else ((self.numMpProcess // 2) - 1)
        self.projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"

        # set device.
        self.cudaDeviceForAll = "cuda:%d" % self.rank
        torch.cuda.set_device(self.cudaDeviceForAll)

        if self.numMpProcess <= 0:  # single threaded
            # assert self.rank == 0
            self.logDir = checkPDSRLogDirNew(
                config,
                projRoot=self.projRoot,
                ifInitial=config.R.startswith("Rtmp"),
            )
        else:  # DDP
            dist.init_process_group(
                backend="nccl", rank=self.rank, world_size=self.numMpProcess
            )
            dist.barrier(device_ids=[self.rank])
            self.setup_for_distributed(self.rank == self.trackedRank)
            dist.barrier(device_ids=[self.rank])
            if self.rank == self.trackedRank:
                self.logDir = checkPDSRLogDirNew(
                    config,
                    projRoot=self.projRoot,
                    ifInitial=config.R.startswith("Rtmp"),
                )
            else:
                self.logDir = self.projRoot + "v/P/%s/%s/%s/%s/" % (
                    config.P,
                    config.D,
                    config.S,
                    config.R,
                )
            dist.barrier(device_ids=[self.rank])

        # logger
        if (
            self.ifDumpLogging
        ):  # Future multi-GPU you may also want to modify this a little bit.
            self.logger = Logger(
                self.logDir + "trainingLog/",
                update_count=1000,
                refresh_count=10000,
            )
        else:
            self.logger = LoggerDummy()
        self.logger.info(
            "====== Starting training: Host %s GPU %s CPU %d Rank %d numMpProcess %d "
            "%s %s %s %s ======"
            % (
                gethostname(),
                os.environ["CUDA_VISIBLE_DEVICES"],
                os.getpid(),
                self.rank,
                self.numMpProcess,
                config.P,
                config.D,
                config.S,
                config.R,
            )
        )

        # cuda backend
        torch.backends.cudnn.benchmark = True

        # meta params (not influencing model training)
        self.printFreq = 10
        self.minSnapshotFreq = 2000
        self.samplingVerbose = False
        self.monitorTrainFreq = 2000
        self.monitorValFreq = 2000
        self.monitorDumpFreq = 2000
        self.monitorMode = 960  # which iteration is the first to step the monitors

        # runnerComplementTheConfig
        self.config = self._runnerComplementTheConfig(
            config, numMpProcess=self.numMpProcess
        )

    @staticmethod
    def _runnerComplementTheConfig(config, **kwargs):
        # This function cannot be called during demo time. Demo does not allow multiple GPU.
        numMpProcess = kwargs["numMpProcess"]
        datasetConfDict = config.datasetConfDict
        for datasetConf in datasetConfDict.values():
            if numMpProcess > 0:
                keys = list(datasetConf.keys())
                for k in keys:
                    if k.startswith("batchSize"):
                        datasetConf[k + "PerProcess"] = int(
                            datasetConf[k] / numMpProcess
                        )
                    elif k == "memory_bank_size_total":
                        datasetConf["memory_bank_size_per_process"] = int(
                            datasetConf["memory_bank_size_total"] / numMpProcess
                        )
            else:
                keys = list(datasetConf.keys())
                for k in keys:
                    if k.startswith("batchSize"):
                        datasetConf[k + "PerProcess"] = datasetConf[k]
                    elif k == "memory_bank_size_total":
                        datasetConf["memory_bank_size_per_process"] = datasetConf["memory_bank_size_total"]

        return config

    def metaDataLoading(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaDataLoading")

        datasetConfDict = config.datasetConfDict

        datasetObjDict = OrderedDict([])

        for datasetConf in datasetConfDict.values():
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                datasetObj = PRenderingNerfBlenderDataset(datasetConf)

                assert "memory_bank_size_per_process" not in datasetConf.keys()  # does not allow memory bank
                assert np.all(datasetObj.fg_masks > 0)
                datasetObj.preInd = np.where(datasetObj.fg_depths < 100)[0]
                datasetObj.preIndBgInFg = np.where(datasetObj.fg_depths > 100)[0]
                datasetObj.augbg_indexID = np.concatenate([
                    datasetObj.bg_indexID,
                    datasetObj.fg_indexID[datasetObj.preIndBgInFg],
                ], 0)               
                datasetObj.augbg_pixelID = np.concatenate([
                    datasetObj.bg_pixelID,
                    datasetObj.fg_pixelID[datasetObj.preIndBgInFg],
                ], 0)

                # to accelerate, put part of things into thgpu
                datasetObj.fg_hdrs_thgpu = torch.from_numpy(datasetObj.fg_hdrs).float().to(self.cudaDeviceForAll)
                datasetObj.fg_depths_thgpu = torch.from_numpy(
                    datasetObj.fg_depths
                ).float().to(self.cudaDeviceForAll)

                datasetObjDict[datasetConf["dataset"]] = datasetObj
            elif datasetConf["class"] == "RenderingLightStageOriginal2":
                datasetObj = PRenderingLightStageOrigianl2(datasetConf)
                datasetObjDict[datasetConf["dataset"]] = datasetObj
            else:
                raise NotImplementedError(
                    "Unknown dataset class: %s" % datasetConf["class"]
                )

        self.datasetObjDict = datasetObjDict

    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netConstruct")

        models = {}

        # all fixed settings
        with open(
            self.projRoot + 
            "v/P/PnrtfBerkeley/hashEmbedInfo_renderingNerfBlender58Pointk4L112V100.pkl"
        , "rb") as f:
            tmp = pickle.load(f)
        bounding_box = [
            torch.from_numpy(tmp["bounding_box_min"]).float().to(self.cudaDeviceForAll),
            torch.from_numpy(tmp["bounding_box_max"]).float().to(self.cudaDeviceForAll),
        ]
        vertices = torch.from_numpy(tmp["vertices"]).float().to(self.cudaDeviceForAll)
        assert config.R.endswith("3")  # must be the hotdog scene
        hashEmbed = HashEmbedder(
            bounding_box=bounding_box,
            n_features_per_level=16,  # won't listen to the config at this point
            base_resolution=16.,
            n_levels=19,
            finest_resolution=256.,
            sparse=True,
            vertices=vertices,
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

    def _netRandomInitialization(self, iter_label):
        # random initialization
        # (Different threads might result in different model weights. Only Rank 0 is used)
        self.logger.info("[Trainer] MetaModelLoading - _netRandomInitialization")

        if iter_label is not None:
            # So by definition these nan should be replaced with normal numbers later on.
            print("    NaN Init Robust Test... As iter_label is %s" % iter_label)
            for k_model, model in self.models.items():
                if k_model not in self.models["meta"][
                    "nonLearningModelNameList"
                ] and k_model not in ["meta"]:
                    for k, v in model.state_dict().items():
                        if "num_batches_tracked" not in k:
                            # print('%s - %s' % (k_model, k))
                            v.fill_(float("nan"))
        pass

    def _netResumingInitialization(self, **kwargs):
        iter_label = kwargs["iter_label"]
        self.logger.info(
            "[Trainer] MetaModelLoading - _netResumingInitialization (iter_label = %s)"
            % str(iter_label)
        )
        if iter_label is None:
            resumeIter = probe_load_network(self.logDir)
        elif type(iter_label) is str:
            if iter_label == "latest":
                resumeIter = "latest"
            else:
                resumeIter = int(iter_label)
        else:
            assert type(iter_label) is int
            resumeIter = iter_label
        if resumeIter == "latest" or resumeIter >= 0:
            for k in self.models.keys():
                if (
                    k != "meta"
                    and k not in self.models["meta"]["nonLearningModelNameList"]
                ):
                    _, self.models[k] = load_network(
                        self.logDir,
                        self.models[k],
                        k,
                        resumeIter,
                        map_location=self.cudaDeviceForAll,
                    )
        if resumeIter == "latest":
            resumeIter = -1
        return resumeIter

    def _netFinetuningInitialization(self):
        self.logger.info("[Trainer] MetaModelLoading - _netFinetuningInitialization")
        if hasattr(self.config, "finetuningPDSRI"):
            raise ValueError("This should not happen here. For volume rendering, re-define this function in extension.py")
            for k in self.models.keys():
                finetuningPDSRI = self.config.finetuningPDSRI
                if (k != "meta") and (
                    k not in self.models["meta"]["nonLearningModelNameList"]
                ):
                    fn = (
                        self.projRoot
                        + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth"
                        % (
                            finetuningPDSRI["P"],
                            finetuningPDSRI["D"],
                            finetuningPDSRI["S"],
                            finetuningPDSRI["R"],
                            k,
                            finetuningPDSRI["I"],
                        )
                    )
                    with open(fn, "rb") as f:
                        self.models[k].load_state_dict(
                            torch.load(f, map_location=self.cudaDeviceForAll),
                            strict=False,
                        )
        else:
            self.logger.info("    No finetuning specs found")

    def _optimizerSetups(self, resumeIter):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _optimizerSetups")

        # Hey, here we really have two different optimizers. So implmementation manually
        self.optimizerModels = {}
        hashEmbed_params = list(self.models["hashEmbed"].parameters())
        if config.hash_sparse:
            self.optimizerModels["optHashEmbed"] = optim.SparseAdam(
                hashEmbed_params, lr=config.sparse_adam_lr, betas=(0.9, 0.99), eps=1.0e-15
            )
        else:
            assert config.sparse_adam_lr == config.adam_lr
            self.optimizerModels["optHashEmbed"] = optim.Adam(
                hashEmbed_params, lr=config.sparse_adam_lr, betas=(0.9, 0.99), eps=1.0e-15
            )
        renderer_params = list(self.models["renderer"].parameters())
        self.optimizerModels["optRenderer"] = optim.Adam(
            renderer_params, lr=config.adam_lr, betas=(0.9, 0.99), eps=1.0e-15
        )

        # resume
        # if resumeIter == "latest" or resumeIter >= 0:
        if resumeIter >= 0:
            for k in self.optimizerModels.keys():
                _, self.optimizerModels[k] = load_network(
                    self.logDir,
                    self.optimizerModels[k],
                    k,
                    resumeIter,
                    map_location=self.cudaDeviceForAll,
                )

    def _netHookSetups(self, **kwargs):
        self.logger.info("[Trainer] MetaModelLoading - _netHookSetups")
        hook_type = kwargs["hook_type"]
        resumeIter = kwargs["resumeIter"]
        config = self.config
        if hook_type is not None:
            assert self.numMpProcess <= 0
            assert self.rank == self.trackedRank
            self.hookModels = {}
            for k in self.models.keys():
                if (
                    k != "meta"
                    and k not in self.models["meta"]["nonLearningModelNameList"]
                ):
                    self.hookModels[k] = PyTorchForwardHook(
                        hook_type,
                        self.models[k],
                        "%s%s%s%sI%d"
                        % (config.P, config.D, config.S, config.R, resumeIter),
                    )

    def metaModelLoading(self, **kwargs):
        # kwargs
        iter_label = kwargs["iter_label"]  # required for any case
        hook_type = kwargs["hook_type"]  # required for any case

        self._netConstruct()

        # net train mode (always set to train - if you are doing val, then you should eval it in
        # your function)
        for k in self.models.keys():
            if k != "meta":
                self.models[k].train()

        self._netRandomInitialization(iter_label=iter_label)
        self._netFinetuningInitialization()
        resumeIter = self._netResumingInitialization(iter_label=iter_label)

        # DDP
        if self.numMpProcess > 0:
            dist.barrier(device_ids=[self.rank])
            for k in self.models.keys():
                if (
                    k != "meta"
                    and k not in self.models["meta"]["nonLearningModelNameList"]
                ):
                    self.models[k] = torch.nn.parallel.DistributedDataParallel(
                        self.models[k],
                        device_ids=[self.cudaDeviceForAll],
                    )
            dist.barrier(device_ids=[self.rank])

        self._optimizerSetups(resumeIter)
        self._netHookSetups(hook_type=hook_type, resumeIter=resumeIter)

        return resumeIter

    def metaLossPrepare(self):
        pass

    def setupMetaVariables(self):
        meta = {}

        self.meta = meta

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
            self.testingOnTrainingDatasetObj = PRenderingNerfBlenderDataset(
                datasetConf_testingOnTraining
            )
        elif datasetConf_testingOnTraining["class"] == "RenderingLightStageOriginal2":
            self.testingOnTrainingDatasetObj = PRenderingLightStageOrigianl2(
                datasetConf_testingOnTraining
            )
        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf_testingOnTraining["class"])

        # Val Vis
        self.htmlStepperVal = HTMLStepper(
            self.monitorValLogDir, 40, "monitorVal"
        )
        self.testDataEntry = getTestDataEntryDict(
            wishedTestDataNickName=[
                "%sCase%dSplit%s"
                % (
                    list(self.config.datasetConfDict.values())[0]["dataset"],
                    list(self.config.datasetConfDict.values())[0]["caseID"],
                    "Val",
                )
            ],
            hdrScaling=list(self.config.datasetConfDict.values())[0]["hdrScaling"],
            white_back=list(self.config.datasetConfDict.values())[0]["white_back"],
            ifLoadDepth=list(self.config.datasetConfDict.values())[0]["ifLoadDepth"],
            depthFileListA0Tag=list(self.config.datasetConfDict.values())[0].get("depthFileListA0Tag", None),
            depthFileListDirTag=list(self.config.datasetConfDict.values())[0].get("depthFileListDifTag", None),
            bucket=list(self.config.datasetConfDict.values())[0].get("bucket", None),
            ray_near=list(self.config.datasetConfDict.values())[0]["ray_near"],
            ray_far=list(self.config.datasetConfDict.values())[0]["ray_far"],
        )[
            "%sCase%dSplit%s"
            % (
                list(self.config.datasetConfDict.values())[0]["dataset"],
                list(self.config.datasetConfDict.values())[0]["caseID"],
                "Val",
            )
        ]

    def initializeAll(self, **kwargs):
        # kwargs
        iter_label = kwargs["iter_label"]  # required for all cases
        hook_type = kwargs["hook_type"]  # required for all cases
        if_need_metaDataLoading = kwargs[
            "if_need_metaDataLoading"
        ]  # Set to true when training. Set to False when evaluating.

        self.setupMetaVariables()
        if if_need_metaDataLoading:
            self.metaDataLoading()
        self.resumeIter = self.metaModelLoading(
            iter_label=iter_label, hook_type=hook_type
        )
        self.metaLossPrepare()
        self.setupMonitor()

        self.logger.info("Initialization finished! Rank = %d" % self.rank)

    def saveAnomaly(self, **kwargs):
        message = kwargs['message']
        iterCount = kwargs['iterCount']
        ifStopHere = kwargs["ifStopHere"]
        # batch_thgpu = kwargs['batch_thgpu']
        # with open(self.logDir + 'anomaly/%s_thgpu_host_%s_iter_%d_rank_%d.pkl' %
        #         (message, gethostname(), iterCount, self.rank), 'wb') as f:
        #     pickle.dump(batch_thgpu, f)
        models = self.models
        for k in models.keys():
            if (k != 'meta') and (k not in models['meta']['nonLearningModelNameList']):
                if self.numMpProcess <= 0:
                    torch.save(
                        self.models[k].state_dict(),
                        self.logDir + 'anomaly/%s_models%s_host_%s_iter_%d_rank_%d.pth' %
                            (message, bt(k), gethostname(), iterCount, self.rank)
                    )
                else:
                    torch.save(
                        self.models[k].module.state_dict(),
                        self.logDir + 'anomaly/%s_models%s_host_%s_iter_%d_rank_%d.pth' %
                            (message, bt(k), gethostname(), iterCount, self.rank)
                    )
        for k in kwargs.keys():
            if k not in ["message", "iterCount", "ifStopHere"]:  # , "batch_thgpu"]:
                x = kwargs[k]
                with open(self.logDir + "anomaly/%s_%s_host_%s_iter_%d_rank_%d.pkl" % 
                        (message, k, gethostname(), iterCount, self.rank), "wb") as f:
                    pickle.dump(x, f)

        if ifStopHere:
            if self.numMpProcess > 0:
                raise ValueError
            else:
                import ipdb
                ipdb.set_trace()
                print(1 + 1)

    def saveModelSnapshot(self, **kwargs):
        iterCount = kwargs["iterCount"]
        if self.rank == self.trackedRank:
            if self.numMpProcess > 0:  # self.net.module
                for k in self.models.keys():
                    if (
                        k != "meta"
                        and k not in self.models["meta"]["nonLearningModelNameList"]
                    ):
                        save_network(
                            self.logDir,
                            self.models[k].module,
                            k,
                            str(iterCount),
                        )
                for k in self.optimizerModels.keys():
                    save_network(
                        self.logDir,
                        self.optimizerModels[k],
                        k,
                        str(iterCount),
                    )
            else:
                for k in self.models.keys():
                    if (
                        k != "meta"
                        and k not in self.models["meta"]["nonLearningModelNameList"]
                    ):
                        save_network(
                            self.logDir, self.models[k], k, str(iterCount)
                        )
                for k in self.optimizerModels.keys():
                    save_network(
                        self.logDir,
                        self.optimizerModels[k],
                        k,
                        str(iterCount),
                    )

    def saveBatchVis(self, batch_vis, **kwargs):
        if self.rank == self.trackedRank:
            iterCount = kwargs["iterCount"]
            with open(
                self.logDir + "dump/train_iter_%d.pkl" % iterCount, "wb"
            ) as f:
                pickle.dump(
                    {
                        k: batch_vis[k]
                        for k in batch_vis.keys()
                        if not k.startswith("feat") and not k.startswith("fullsurface")
                    },
                    f,
                )

    def stepMonitorImmediateFlow(self, batch_vis, **kwargs):
        iterCount = kwargs["iterCount"]
        ifMonitorDump = kwargs["ifMonitorDump"]
        ifIsTrackedRank = self.rank == self.trackedRank
        htmlStepper = self.htmlStepperImmediateFlow

        config = self.config
        P = config["P"]
        D = config["D"]
        S = config["S"]
        R = config["R"]

        datasetObj = self.testingOnTrainingDatasetObj
        dataset = datasetObj.datasetConf["dataset"]
        _dataset = "_" + dataset

        rank = self.rank
        numMpProcess = self.numMpProcess
        trackedRank = self.trackedRank
        realNumMpProcess = int(max([numMpProcess, 1]))

        indChosen = np.unique(batch_vis["indexID" + _dataset])
        if len(indChosen) > 2:
            indChosen = indChosen[:2]
        indVisChosen = copy.deepcopy(indChosen)

        benRecord = {}
        for j in indChosen:
            print(
                "    [Trainer Visualizer ImmediateFlow] Stepping Visualizer ImmediateFlow: %d"
                % j
            )
            bsv0 = datasetObj.getOneNP(int(j))
            ifRequiresDrawing = (j in indVisChosen) and (rank == trackedRank)
            bsv0 = benchmarkingPsnrLdrRenderingNerfBlenderFunc(
                bsv0,
                # misc
                cudaDevice=self.cudaDeviceForAll,
                datasetObj=datasetObj,
                iterCount=iterCount,
                # benchmarking rules
                ifFit=False,
                ssimWindow=11,
                orthogonalDensityResolution=128,  # not affecting PSNR
                marching_cube_thre=0.5,  # not affecting PSNR
                # drawing
                ifRequiresDrawing=ifRequiresDrawing,
                ifRequiresBenchmarking=True,
                # predicting
                ifRequiresPredictingHere=True,
                models=self.models,
                doPred0Func=self.doPred0,
                config=self.config,
            )

            tmp = [k for k in bsv0.keys() if k.startswith("predBen")]
            for k in tmp:
                _ = k[len("predBen") :]
                bsv0["finalBen%s" % bt(_)] = bsv0["predBen%s" % bt(_)]

            bsv0_toStore = {}
            for k in bsv0.keys():
                if (
                    k.startswith("finalBen")
                    or k
                    in [
                        "iterCount",
                        "visIndex",
                        "P",
                        "D",
                        "S",
                        "R",
                        "methodologyName",
                        "index",
                        "did",
                        "datasetID",
                        "dataset",
                        "flagSplit",
                    ]
                    or k.startswith("finalBen")
                ):
                    bsv0_toStore[k] = bsv0[k]
            benRecord[j] = bsv0_toStore

            if ifRequiresDrawing:
                PDSRI = "%s%s%s%sI%d" % (P, D, S, R, iterCount)
                summary0, txt0, brInds = addToSummary0Txt0BrInds(
                    summary0=OrderedDict([]),
                    txt0=[],
                    brInds=[0],
                    approachEntryDict={PDSRI: {"approachShownName": PDSRI}},
                    bsv0_forVis_dict={PDSRI: bsv0},
                )
                headerMessage = (
                    "%s-%s-%s-%s-I%d Dataset: %s, Index: %d(%d), visIndex: %d"
                    % (
                        P,
                        D,
                        S,
                        R,
                        iterCount,
                        bsv0["dataset"],
                        bsv0["index"],
                        bsv0["flagSplit"],
                        -1,
                    )
                )
                subMessage = "PSNR Fine: %.3f" % (
                    bsv0["finalBenPSNRFine"],
                )
                htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
            if ifMonitorDump and ifIsTrackedRank and ifRequiresDrawing:
                for meshName in ["coarse", "fine"]:
                    sysLabel = "world"
                    with open(
                        self.monitorImmediateFlowLogDir
                        + "%s%s%s%sI%d_%s%s_%s_%d(%d).obj"
                        % (
                            P,
                            D,
                            S,
                            R,
                            iterCount,
                            meshName,
                            bt(sysLabel),
                            bsv0["dataset"],
                            bsv0["index"],
                            bsv0["flagSplit"],
                        ),
                        "w",
                    ) as f:
                        pass
                        # dump_obj_np_fileObj(
                        #     f,
                        #     bsv0["%sVert%s" % (meshName, bt(sysLabel))],
                        #     bsv0["%sFace" % meshName],
                        # )

        tmp = {"finalBenValidCount": np.array(len(benRecord.keys()), dtype=np.float32)}
        keys = [
            k[len("finalBen") :]
            for k in list(list(benRecord.values())[0].keys())
            if k.startswith("finalBen")
        ]
        ar = {
            k: np.array(
                [x["finalBen" + k] for x in benRecord.values() if k != "RuntimeMachine"], dtype=np.float32
            ).mean()
            for k in keys
        }
        tmp.update({k: np.array(ar[k], dtype=np.float32) for k in keys})
        ben = tmp

        return ben

    def stepMonitorVal(self, **kwargs):
        iterCount = kwargs["iterCount"]
        ifMonitorDump = kwargs["ifMonitorDump"]
        ifIsTrackedRank = self.rank == self.trackedRank
        htmlStepper = self.htmlStepperVal

        config = self.config
        P = config["P"]
        D = config["D"]
        S = config["S"]
        R = config["R"]

        testDataEntry = self.testDataEntry
        datasetObj = testDataEntry["datasetObj"]
        rank = self.rank
        numMpProcess = self.numMpProcess
        trackedRank = self.trackedRank
        realNumMpProcess = int(max([numMpProcess, 1]))

        indChosen = testDataEntry["indChosen"]
        indVisChosen = testDataEntry["indVisChosen"]
        j_tot = len(indChosen)

        # re-order
        indNonVisChosen = list(set(indChosen) - set(indVisChosen))
        insertStartingIndex = int(
            math.floor(float(j_tot) / realNumMpProcess * trackedRank)
        )
        indChosen = (
            (indNonVisChosen[:insertStartingIndex] if indNonVisChosen else [])
            + indVisChosen.tolist()
            + (indNonVisChosen[insertStartingIndex:] if indNonVisChosen else [])
        )

        if numMpProcess <= 0:
            j1 = 0
            j2 = j_tot
        else:
            j1 = int(math.floor(float(j_tot) / realNumMpProcess * rank))
            j2 = int(math.floor(float(j_tot) / realNumMpProcess * (rank + 1)))
        testDataNickName = testDataEntry["testDataNickName"]
        benRecord = {}
        # j2 = j1 + 1  # debug
        for j in range(j1, j2):
            index = indChosen[j]
            print(
                "[Trainer Visualizer Val] Progress Iter%d: "
                "testDataNickName %s, index = %d, j = %d, j1 = %d, j2 = %d, "
                "rank = %d, numMpProcess = %d, j_tot = %d"
                % (
                    iterCount,
                    testDataNickName,
                    index,
                    j,
                    j1,
                    j2,
                    rank,
                    numMpProcess,
                    j_tot,
                )
            )
            batch0_np = testDataEntry["datasetObj"].getOneNP(index)
            batch_np = bsv02bsv(batch0_np)
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis,
                iterCount=iterCount,
                visIndex=0,
                dataset=None,
                P=P,
                D=D,
                S=S,
                R=R,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromAnotherBsv0(
                bsv0_initial,
                bsv2bsv0(batch_vis, visIndex=0),
                copiedKeys=list(set(batch_vis.keys()) - set(bsv0_initial)),
            )
            ifRequiresDrawing = (index in indVisChosen) and (rank == trackedRank)

            if testDataNickName.startswith(
                "renderingNerfBlender"
            ) or testDataNickName.startswith(
                "renderingCapture"
            ) or testDataNickName.startswith(
                "capture7jf"
            ):
                bsv0 = benchmarkingPsnrLdrRenderingNerfBlenderFunc(
                    bsv0_initial,
                    # misc
                    cudaDevice=self.cudaDeviceForAll,
                    datasetObj=testDataEntry["datasetObj"],
                    iterCount=iterCount,
                    # benchmarking rules
                    ifFit=False,
                    ssimWindow=11,
                    orthogonalDensityResolution=128,  # not affecting PSNR
                    marching_cube_thre=0.5,  # not affecting PSNR
                    # drawing
                    ifRequiresDrawing=ifRequiresDrawing,
                    ifRequiresBenchmarking=True,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=self.models,
                    doPred0Func=self.doPred0,
                    config=self.config,
                )
            else:
                raise NotImplementedError(
                    "Unknown testDataNickName: %s" % testDataNickName
                )

            if bsv0 is None:
                continue
            tmp = [k for k in bsv0.keys() if k.startswith("predBen")]
            for k in tmp:
                _ = k[len("predBen") :]
                bsv0["finalBen%s" % bt(_)] = bsv0["predBen%s" % bt(_)]

            bsv0_toStore = {}
            for k in bsv0.keys():
                if (
                    k.startswith("finalBen")
                    or k
                    in [
                        "iterCount",
                        "visIndex",
                        "P",
                        "D",
                        "S",
                        "R",
                        "methodologyName",
                        "index",
                        "did",
                        "datasetID",
                        "dataset",
                        "flagSplit",
                    ]
                    or k.startswith("finalBen")
                ):
                    bsv0_toStore[k] = bsv0[k]
            benRecord[j] = bsv0_toStore

            if ifRequiresDrawing:
                PDSRI = "%s%s%s%sI%d" % (P, D, S, R, iterCount)
                summary0, txt0, brInds = addToSummary0Txt0BrInds(
                    summary0=OrderedDict([]),
                    txt0=[],
                    brInds=[0],
                    approachEntryDict={PDSRI: {"approachShownName": PDSRI}},
                    bsv0_forVis_dict={PDSRI: bsv0},
                )
                headerMessage = (
                    "%s-%s-%s-%s-I%d Dataset: %s, Index: %d(%d), visIndex: %d"
                    % (
                        bsv0["P"],
                        bsv0["D"],
                        bsv0["S"],
                        bsv0["R"],
                        bsv0["iterCount"],
                        bsv0["dataset"],
                        bsv0["index"],
                        bsv0["flagSplit"],
                        bsv0["visIndex"],
                    )
                )
                subMessage = "PSNR Fine: %.3f" % (
                    bsv0["finalBenPSNRFine"],
                )
                htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
            if ifMonitorDump and ifIsTrackedRank and ifRequiresDrawing:
                for meshName in ["coarse", "fine"]:
                    sysLabel = "world"
                    with open(
                        self.monitorValLogDir
                        + "%s%s%s%sI%d_%s%s_%s_%d(%d).obj"
                        % (
                            P,
                            D,
                            S,
                            R,
                            iterCount,
                            meshName,
                            bt(sysLabel),
                            bsv0["dataset"],
                            bsv0["index"],
                            bsv0["flagSplit"],
                        ),
                        "w",
                    ) as f:
                        pass

        ar = []
        keys = [
            k[len("finalBen") :]
            for k in list(benRecord.values())[0].keys()
            if k.startswith("finalBen")
        ]
        for br in benRecord.values():
            ar.append(
                np.array([br["finalBen%s" % bt(k)] for k in keys if k != "RuntimeMachine"], dtype=np.float32)
            )
        ar = np.stack(ar, 1)
        arSum = ar.sum(1)
        arSum = np.concatenate([arSum, np.array([len(benRecord)], dtype=np.float32)], 0)
        arSum_thgpu = torch.from_numpy(arSum).to(self.cudaDeviceForAll)
        if numMpProcess > 0:
            dist.barrier()
            dist.all_reduce(arSum_thgpu, op=ReduceOp.SUM)
            dist.barrier()
        validCount = int(arSum_thgpu[-1].detach().cpu().numpy())
        assert validCount > 0, "validCount is 0"
        arMean = arSum_thgpu.detach().cpu().numpy() / float(validCount)
        ben = {
            "finalBen%s" % bt(keys[j]): np.array(arMean[j], dtype=np.float32)
            for j in range(len(keys))
        }
        ben["finalBenValidCount"] = np.array(validCount, dtype=np.float32)
        return ben

    def stepBatch(self, **kwargs):
        iterCount = kwargs["iterCount"]

        config = self.config
        tmp = OrderedDict([])
        tmpg = OrderedDict([])
        for dataset in config.datasetConfDict.keys():
            datasetConf = config.datasetConfDict[dataset]
            datasetObj = self.datasetObjDict[dataset]
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                ind = datasetObj.preInd[np.random.randint(
                    0,
                    datasetObj.preInd.shape[0],  # datasetObj.m_memory_bank,
                    datasetConf["batchSizePerProcess"],
                    dtype=np.int64,
                )]
                batchDid_thcpu = {
                    "valid_mask": torch.from_numpy(datasetObj.fg_masks[ind]),
                    "indexID": torch.from_numpy(datasetObj.fg_indexID[ind]),
                    "pixelID": torch.from_numpy(datasetObj.fg_pixelID[ind]),
                }
                assert torch.all(batchDid_thcpu["valid_mask"])
                batchDid_thgpu = castAnything(
                    batchDid_thcpu, "thcpu2thgpu", device=self.cudaDeviceForAll
                )
                ind_thgpu = torch.from_numpy(ind).to(self.cudaDeviceForAll)
                batchDid_thgpu["hdrs"] = datasetObj.fg_hdrs_thgpu[ind_thgpu, :]
                if datasetObj.ifLoadDepth:
                    # batchDid_thcpu["depths"] = torch.from_numpy(datasetObj.fg_depths[ind])
                    batchDid_thgpu["depths"] = datasetObj.fg_depths_thgpu[ind]
                if datasetObj.ifLoadNormal:
                    raise ValueError("This script does not read normals")
                    batchDid_thcpu["normals"] = torch.from_numpy(datasetObj.fg_normals[ind])
                if ("memory_bank_update_freq" in datasetConf.keys()) and (datasetConf["memory_bank_update_freq"] > 0):
                    if (iterCount > 0) and (iterCount % datasetConf["memory_bank_update_freq"] == 0):
                    # if (iterCount % datasetConf["memory_bank_update_freq"] == 0):
                        datasetObj.update_the_memory_bank_once()
            elif datasetConf["class"] == "RenderingLightStageOriginal2":
                rays_fg_count = datasetObj.all_fg_indexID.shape[0]
                ind_fg = np.random.randint(
                    0,
                    rays_fg_count,
                    datasetConf["batchSize"],  # the whole batch is only about foreground
                    dtype=np.int64,
                )
                batchDid_np = {
                    "hdrs": datasetObj.all_fg_hdrs[ind_fg],
                    "valid_mask": np.ones((ind_fg.shape[0],), dtype=bool),
                    "indexID": datasetObj.all_fg_indexID[ind_fg],
                    "pixelID": datasetObj.all_fg_pixelID[ind_fg],
                }
                if datasetConf.get("ifLoadDepth", False):
                    batchDid_np["depths"] = datasetObj.all_fg_zbuffer[ind_fg]
                batchDid_thcpu = castAnything(batchDid_np, "np2thcpu")
                batchDid_thgpu = castAnything(batchDid_thcpu, "thcpu2thgpu", device=self.cudaDeviceForAll)
            else:
                raise NotImplementedError(
                    "Unknown dataset class: %s" % datasetConf["class"]
                )
            tmp[dataset] = batchDid_thcpu
            tmpg[dataset] = batchDid_thgpu

        batch_thcpu = {}
        for dataset in tmp.keys():
            for k in tmp[dataset].keys():
                batch_thcpu[k + "_" + dataset] = tmp[dataset][k]

        batch_thgpu = {}
        for dataset in tmpg.keys():
            for k in tmpg[dataset].keys():
                batch_thgpu[k + "_" + dataset] = tmpg[dataset][k]

        return batch_thcpu, batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        assert len(self.datasetObjDict.keys()) == 1
        datasetObj = list(self.datasetObjDict.values())[0]
        datasetConf = datasetObj.datasetConf
        dataset = datasetObj.datasetConf["dataset"]

        if datasetConf["class"] == "RenderingNerfBlenderDataset":
            # rays
            E = datasetObj.A0["E"][batch_thcpu["indexID_%s" % dataset].numpy(), :]
            L = datasetObj.A0["L"][batch_thcpu["indexID_%s" % dataset].numpy(), :]
            U = datasetObj.A0["U"][batch_thcpu["indexID_%s" % dataset].numpy(), :]
            w2c = torch.FloatTensor(ELU2cam(np.concatenate([E, L, U], 1))).to(
                self.cudaDeviceForAll
            )
            c2w = torch.linalg.inv(w2c)
            rays_o = c2w[:, :3, 3]
            directions = torch.FloatTensor(
                datasetObj.directions_flattened[
                    batch_thcpu["pixelID_%s" % dataset].numpy(), :
                ]
            ).to(self.cudaDeviceForAll)
            rays_d = (c2w[:, :3, :3] * directions[:, None, :]).sum(2)
            rays_d = (
                rays_d / torch.norm(rays_d, p=2, dim=1)[:, None]
            )  # This was missing in the past
            near = datasetObj.near * torch.ones_like(rays_o[:, :1])
            far = datasetObj.far * torch.ones_like(rays_o[:, :1])
            batch_thgpu["rays_%s" % dataset] = torch.cat(
                [rays_o, rays_d, near, far], 1
            ).contiguous()
            batch_thgpu["zbuffer_to_euclidean_ratio_%s" % dataset] = torch.from_numpy(
                datasetObj.zbuffer_to_euclidean_ratio[batch_thcpu["pixelID_%s" % dataset].numpy()]
            ).float().contiguous().to(self.cudaDeviceForAll)

            # lgt
            if datasetObj.datasetConf["lgtLoadingMode"] == "QSPL":
                indexID = batch_thcpu["indexID_%s" % dataset].numpy()
                for k in ["lgtE", "objCentroid"]:
                    batch_thgpu["%s_%s" % (k, dataset)] = torch.from_numpy(
                        datasetObj.A0[k][indexID]
                    ).to(self.cudaDeviceForAll)
                omegaInput = (
                    batch_thgpu["objCentroid_%s" % dataset] - batch_thgpu["lgtE_%s" % dataset]
                )
                omegaInput_norm = torch.norm(omegaInput, p=2, dim=1)
                assert omegaInput_norm.min() > 1.0e-4
                omegaInput = omegaInput / omegaInput_norm[:, None]
                batch_thgpu["omegaInput_%s" % dataset] = omegaInput
            elif datasetObj.datasetConf["lgtLoadingMode"] == "OMEGA":
                lgtDatasetCache = datasetObj.meta["lgtDatasetCache"].cache
                assert len(lgtDatasetCache.keys()) == 1  # only one lighting dataset is assumed (the whole training set shares the same lighting dataset)
                A0_lgt = list(lgtDatasetCache.values())[0]["A0"]
                lgtID = datasetObj.A0["lgtIDList"][batch_thcpu["indexID_%s" % dataset].numpy()]
                omega = A0_lgt["omega"][lgtID]
                batch_thgpu["omegaInput_%s" % dataset] = torch.FloatTensor(omega).to(self.cudaDeviceForAll)
            else:
                raise NotImplementedError("Unknown lgtLoadingMode: %s" % datasetObj.datasetConf["lgtLoadingMode"])
        elif datasetConf["class"] == "RenderingLightStageOriginal2":
            # rays
            indexID = batch_thcpu["indexID_%s" % dataset].numpy()  # 0~(180*nOlat-1)
            groupViewID = indexID // datasetObj.nOlat
            focalLengthWidth = datasetObj.focalLengthWidth[groupViewID]
            focalLengthHeight = datasetObj.focalLengthHeight[groupViewID]
            c2w = datasetObj.c2w[groupViewID]

            pixelID = batch_thcpu["pixelID_%s" % dataset].numpy()
            rowID = pixelID // (datasetObj.winWidth + 2 * datasetConf["rays_background_pad_width"])
            colID = pixelID % (datasetObj.winWidth + 2 * datasetConf["rays_background_pad_width"])

            directions_unnormalized_cam = np.stack([
                (colID.astype(np.float32) + 0.5 - datasetObj.winWidth / 2.0 - datasetConf["rays_background_pad_width"]) / focalLengthWidth,
                (rowID.astype(np.float32) + 0.5 - datasetObj.winHeight / 2.0 - datasetConf["rays_background_pad_height"]) / focalLengthHeight,
                np.ones((indexID.shape[0],), dtype=np.float32),
            ], 1)

            directions_unnormalized = (c2w[:, :3, :3] * directions_unnormalized_cam[:, None, :]).sum(2)
            directions_norm = np.linalg.norm(directions_unnormalized, ord=2, axis=1)
            directions_normalized = directions_unnormalized / directions_norm[:, None]

            rays = np.concatenate([
                c2w[:, :3, 3],  # rays_o
                directions_normalized,  # rays_d
                datasetConf["ray_near"] * np.ones((indexID.shape[0], 1), dtype=np.float32),
                datasetConf["ray_far"] * np.ones((indexID.shape[0], 1), dtype=np.float32),
            ], 1)
            batch_thgpu["rays_%s" % dataset] = torch.from_numpy(rays).float().to(self.cudaDeviceForAll)
            zbuffer_to_euclidean_ratio = np.linalg.norm(directions_unnormalized_cam, ord=2, axis=1)
            batch_thgpu["zbuffer_to_euclidean_ratio_%s" % dataset] = torch.from_numpy(zbuffer_to_euclidean_ratio).to(self.cudaDeviceForAll)

            # QSPL lgt (You can also add omega)
            lgtE = datasetObj.olat_lgtEs.reshape((180 * datasetObj.nOlat, 3))[indexID, :]
            objCentroid = np.tile(datasetObj.objCentroid0[None, :], (lgtE.shape[0], 1))
            batch_thgpu["lgtE_%s" % dataset] = torch.from_numpy(lgtE).float().to(self.cudaDeviceForAll)
            batch_thgpu["objCentroid_%s" % dataset] = torch.from_numpy(objCentroid).float().to(self.cudaDeviceForAll)
            omegaInput = (
                batch_thgpu["objCentroid_%s" % dataset] - batch_thgpu["lgtE_%s" % dataset]
            )
            omegaInput_norm = torch.norm(omegaInput, p=2, dim=1)
            assert omegaInput_norm.min() > 1.0e-4
            omegaInput = omegaInput / omegaInput_norm[:, None]
            batch_thgpu["omegaInput_%s" % dataset] = omegaInput
        else:
            raise NotImplementedError("Unknown class: %s" % datasetConf["class"])

        return batch_thgpu

    @staticmethod
    def assertModelsTrainingMode(models):
        for k, v in models.items():
            if (k not in models["meta"]["nonLearningModelNameList"]) and (k != "meta"):
                assert v.training

    @staticmethod
    def assertModelEvalMode(models):
        for k, v in models.items():
            if (k not in models["meta"]["nonLearningModelNameList"]) and (k != "meta"):
                assert not v.training

    @staticmethod
    def setModelsTrainingMode(models):
        for k, v in models.items():
            if (k not in models["meta"]["nonLearningModelNameList"]) and (k != "meta"):
                v.train()

    @staticmethod
    def setModelsEvalMode(models):
        for k, v in models.items():
            if (k not in models["meta"]["nonLearningModelNameList"]) and (k != "meta"):
                v.eval()

    @classmethod
    def doPred0(cls, bsv0, **kwargs):
        datasetObj = kwargs["datasetObj"]
        models = kwargs["models"]
        config = kwargs["config"]
        cudaDevice = kwargs["cudaDevice"]
        orthogonalDensityResolution = kwargs["orthogonalDensityResolution"]
        marching_cube_thre = kwargs["marching_cube_thre"]
        cls.assertModelEvalMode(models)
        rays_thgpu = torch.from_numpy(bsv0["rays"]).to(cudaDevice)
        if ("omegaInput" not in bsv0.keys()):
            tmp = bsv0["objCentroid"] - bsv0["lgtE"]
            bsv0["omegaInput"] = tmp / np.linalg.norm(tmp, ord=2, axis=1)[:, None]
        omegaInput_thgpu = torch.from_numpy(bsv0["omegaInput"]).to(cudaDevice)
        depth_thgpu = torch.from_numpy(bsv0["scaledDepth"]).to(cudaDevice)
        mask_thgpu = torch.from_numpy(bsv0["valid_mask"]).to(cudaDevice) & (depth_thgpu < 100)
        zbuffer_to_euclidean_ratio = torch.from_numpy(bsv0["zbuffer_to_euclidean_ratio"]).to(cudaDevice)
        with torch.no_grad():
            tmp = cls.forwardRendering(
                rays_thgpu,
                omegaInput_thgpu,
                depth_thgpu,
                mask_thgpu,
                zbuffer_to_euclidean_ratio,
                models=models,
                config=config,
            )
            results = tmp["pred"]
            timeElapsed = tmp["timeElapsed"]
        bsv0["timeElapsed"] = timeElapsed
        bsv0["timeEvalMachine"] = gethostname()

        hdr0 = results.view(bsv0["winHeight"], bsv0["winWidth"], 3).detach().cpu().numpy()
        mask0 = (bsv0["valid_mask"] & (bsv0["scaledDepth"] < 100)).reshape((bsv0["winHeight"], bsv0["winWidth"]))
        mask3 = np.tile(mask0[:, :, None], (1, 1, 3))
        hdr0[mask3 == 0] = 0
        ldr0 = tonemap_srgb_to_rgb(torch.from_numpy(hdr0)).numpy()

        bsv0["ldrFinePred"] = ldr0.reshape((-1, 3)).copy()
        # bsv0["ldrCoarsePred"] = ldr0.reshape((-1, 3)).copy()
        bsv0["imgldrFinePred"] = ldr0.copy()
        bsv0["imghdrFinePred"] = hdr0.copy()
        # bsv0["imgldrCoarsePred"] = ldr0.copy()
        bsv0["depthFinePred"] = bsv0["scaledDepth"].reshape((bsv0["winHeight"], bsv0["winWidth"])).copy()
        # bsv0["depthCoarsePred"] = bsv0["depth"].reshape((bsv0["winHeight"], bsv0["winWidth"])).copy()
        bsv0["opacityFinePred"] = np.zeros_like(ldr0[:, :, 0])
        for meshName in ["coarse", "fine"]:
            bsv0["%sVertWorld" % meshName] = bsv0["rays"][:, 0:3] + bsv0["rays"][:, 3:6] * bsv0["scaledDepth"][:, None] * bsv0["zbuffer_to_euclidean_ratio"][:, None]
            bsv0["%sFace" % meshName] = np.array([[0, 1, 2]], dtype=np.int32)
        return bsv0

    @staticmethod
    def forwardRendering(rays, omegaInput, depths, mask_input, zbuffer_to_euclidean_ratio, **kwargs):
        models = kwargs["models"]
        config = kwargs["config"]
        B = rays.shape[0]

        mask = mask_input.detach().clone()
        mask[depths <= 1e-4] = False

        if torch.all(mask):
            # embedded_xyz = models["hashEmbed"](
            #     rays[:, 0:3].contiguous() + rays[:, 3:6].contiguous() * depths[:, None].contiguous() * zbuffer_to_euclidean_ratio[:, None].contiguous())
            t0 = time.time()
            embedded_xyz = models["hashEmbed"](
                rays[:, 0:3].contiguous() 
                +
                rays[:, 3:6].contiguous() *
                depths[:, None].contiguous() *
                zbuffer_to_euclidean_ratio[:, None].contiguous()
            )
            embedded_view_dir = models["viewEmbed"](rays[:, 3:6].contiguous())
            embedded_omega_input = models["viewEmbed"](omegaInput.contiguous())
            pred = models["renderer"](
                embedded_xyz,
                embedded_view_dir,
                embedded_omega_input,
                mode="simpler",
            )
            t1 = time.time()
        else:
            flag_select = mask
            x = (
                rays[flag_select, 0:3].contiguous()
                +
                rays[flag_select, 3:6].contiguous() *
                depths[flag_select, None].contiguous() *
                zbuffer_to_euclidean_ratio[flag_select, None].contiguous()
            )
            for c in range(3):
                x[:, c] = x[:, c].clamp(min=models["hashEmbed"].bounding_box_np[0][c] + 0.0001, max=models["hashEmbed"].bounding_box_np[1][c] - 0.0001)
            t0 = time.time()
            embedded_xyz = models["hashEmbed"](x)
            embedded_view_dir = models["viewEmbed"](rays[flag_select, 3:6].contiguous())
            embedded_omega_input = models["viewEmbed"](omegaInput[flag_select, :].contiguous())
            pred_select = models["renderer"](
                embedded_xyz,
                embedded_view_dir,
                embedded_omega_input,
                mode="simpler",
            )
            t1 = time.time()
            with torch.no_grad():
                ind = torch.zeros(flag_select.shape[0], dtype=torch.int64, device=flag_select.device)
                ind[flag_select] = torch.arange(flag_select.sum()).to(ind.device) + 1
                ind3 = ind[:, None].repeat(1, 3)
            pred_augmented = torch.cat([torch.zeros_like(pred_select[:1, :]), pred_select], 0)
            pred = torch.gather(pred_augmented, dim=0, index=ind3)

        return {
            "pred": pred,
            "timeElapsed": float(t1) - float(t0),
        }

    def forwardNetNeRF(self, batch_thgpu, **kwargs):
        models = kwargs["models"]
        config = kwargs["config"]
        dataset = kwargs["dataset"]
        iterCount = kwargs["iterCount"]

        self.assertModelsTrainingMode(models)

        wl = config.wl

        _dataset = "_" + dataset
        mask = batch_thgpu["valid_mask" + _dataset] & (batch_thgpu["depths" + _dataset] < 100)
        if ("omageInput" + _dataset) not in batch_thgpu.keys():
            batch_thgpu["omegaInput" + _dataset] = F.normalize(
                batch_thgpu["objCentroid" + _dataset] - batch_thgpu["lgtE" + _dataset]
            )
        if ("zbuffer_to_euclidean_ratio" + _dataset) not in batch_thgpu.keys():
            batch_thgpu["zbuffer_to_euclidean_ratio" + _dataset] = batch_thgpu["rays" + _dataset][:, 9]
        pred = self.forwardRendering(
            batch_thgpu["rays" + _dataset],
            batch_thgpu["omegaInput" + _dataset],
            batch_thgpu["depths" + _dataset],
            mask,
            batch_thgpu["zbuffer_to_euclidean_ratio" + _dataset],
            # depth=batch_thgpu.get("depths" + _dataset, None),
            models=models,
            config=config,
        )["pred"]
        batch_thgpu["pred" + _dataset] = pred

        # loss
        label = batch_thgpu["hdrs" + _dataset]
        t = (pred - label) / pred.detach().clamp(1.0e-3, 1)
        lossHdr = torch.norm(t, p=1.8) / pred.shape[0] * 20.0
        batch_thgpu["lossHdr"] = wl["lossHdr"] * lossHdr

        loss = 0.0
        for k in wl.keys():
            loss += batch_thgpu[k]
        batch_thgpu["loss"] = loss

        # stat
        batch_thgpu["ldrs" + _dataset] = tonemap_srgb_to_rgb(
            batch_thgpu["hdrs" + _dataset]
        )
        batch_thgpu["predLdr" + _dataset] = tonemap_srgb_to_rgb(
            pred
        )
        for coarseOrFine in ["coarse", "fine"]:
            batch_thgpu["statPsnr%s" % bt(coarseOrFine)] = psnr(
                batch_thgpu["predLdr" + _dataset],
                batch_thgpu["ldrs" + _dataset],
            )
        # for k in ["hashEmbed", "renderer"]:
        #     batch_thgpu["statLr%s" % bt(k)] = self.schedulerModels[k].get_lr()[0]

        return batch_thgpu

    def forwardBackwardUpdate(self, batch_thgpu, **kwargs):
        ifAllowBackward = kwargs["ifAllowBackward"]
        iterCount = kwargs["iterCount"]

        self.assertModelsTrainingMode(self.models)

        config = self.config

        batch_thgpu = self.forwardNetNeRF(
            batch_thgpu,
            models=self.models,
            dataset=list(self.config.datasetConfDict.values())[0]["dataset"],
            config=config,
            iterCount=iterCount,
        )

        if ifAllowBackward:
            self.backwardLoss(
                batch_thgpu,
                iterCount=iterCount,
                optimizerModels=self.optimizerModels,
                # schedulerModels=self.schedulerModels,
            )

        return batch_thgpu

    @classmethod
    def backwardLoss(cls, batch_thgpu, **kwargs):
        iterCount = kwargs["iterCount"]
        optimizerModels = kwargs["optimizerModels"]

        if iterCount > 0:
            for k in ["optHashEmbed", "optRenderer"]:
                optimizerModels[k].zero_grad()
            batch_thgpu["loss"].backward()
            for k in ["optHashEmbed", "optRenderer"]:
                optimizerModels[k].step()
        return batch_thgpu

    def trainNoException(self, **kwargs):
        config = self.config
        S = config.S
        R = config.R
        if self.resumeIter >= 0:
            iterCount = self.resumeIter
        else:
            iterCount = 0
        while True:
            
            if (
                iterCount >= 500000 + self.monitorMode + 2
            ) and (gethostname() not in ["cthulhu2", "fish3"]):  # 50W exit - for MAST clusters, and also make sure to save the last
                break

            timeIterStart = time.time()

            # Rtmp check
            if iterCount == 1000 and config.R.startswith("Rtmp"):
                self.logger.info(
                    "This is %s session, which is temporary!!! No Further Forwarding"
                    % config.R
                )
                return

            # whether or not to dump
            power10 = int(math.log10(float(iterCount + 1)))
            power10 = min(3, power10)
            divider = 10**power10
            divider = max(divider, 40000)

            # these ifs does not take the rank / numMpProcess into account
            monitorMode = self.monitorMode
            ifStore = (iterCount > self.resumeIter) and ((
                iterCount % max([divider, self.minSnapshotFreq]) == monitorMode
            ) or (iterCount == 10000 + monitorMode))
            # or iterCount == self.resumeIter + 1)
            ifBackupTheLatestModels = (iterCount % 10000 == 0) and (iterCount > 100000)
            ifMonitorDump = iterCount % self.monitorDumpFreq == monitorMode
            ifMonitorTrain = (iterCount > self.resumeIter) and (
                (iterCount % self.monitorTrainFreq == monitorMode)
                or (
                    iterCount < self.monitorTrainFreq
                    and iterCount % (self.monitorTrainFreq / 2) == monitorMode
                ) 
                or (iterCount == 10000 + monitorMode)
            )
            ifMonitorTrain = (ifMonitorTrain or ifMonitorDump)  # and (
            #     (iterCount >= self.monitorValFreq) or (R.startswith("Rtmpval"))
            # )
            ifMonitorVal = (iterCount > self.resumeIter) and (
                (iterCount % self.monitorValFreq == monitorMode)
                or (
                    iterCount < self.monitorValFreq
                    and iterCount % (self.monitorValFreq / 2) == monitorMode
                ) 
                or (iterCount == 10000 + monitorMode)
            )
            ifMonitorVal = (ifMonitorVal or ifMonitorDump) and (
                (iterCount >= self.monitorValFreq) or (R.startswith("Rtmpval"))
            )

            if iterCount - self.monitorMode in [10000, 40000]:
                ifStore, ifMonitorTrain, ifMonitorVal = True, True, True

            ifPrint = (
                ifStore
                or (iterCount - self.resumeIter) % self.printFreq == 0
                or iterCount < 20
            )
            if ifMonitorTrain or ifMonitorVal:
                ifPrint = True
            # Note ifStore do not do this - all the threads needs sync. ifPrint does not relate to sync.
            ifSaveToFinishedIter = iterCount % 50 == 0

            # ifMonitorImmediateFlow = iterCount == self.monitorMode
            # ifMonitorTrain = ifMonitorVal
            ifMonitorImmediateFlow = ifMonitorTrain

            if S.startswith("Sdummy"):
                ifStore = False
                ifBackupTheLatestModels = False
                ifMonitorDump = False
                ifMonitorImmediateFlow = False
                ifMonitorTrain = False
                ifMonitorVal = False
                ifSaveToFinishedIter = False

            # ifTracked
            ifTracked = self.rank == self.trackedRank or self.numMpProcess == 0

            # printing
            if ifPrint and ifTracked:
                self.logger.info(
                    "---------- Iter %d Training: Host %s GPU %s CPU %d "
                    "Rank %d NumMpProcess %d %s %s %s %s ---------"
                    % (
                        iterCount,
                        gethostname(),
                        os.environ["CUDA_VISIBLE_DEVICES"],
                        os.getpid(),
                        self.rank,
                        self.numMpProcess,
                        config.P,
                        config.D,
                        config.S,
                        config.R,
                    )
                )
                self.logger.info(
                    "    [TimeStamp] timeStamp Iter%d: " % iterCount
                    + time.strftime("%m/%d/%y %H:%M:%S", time.localtime())
                )
                # log virtual env name
                self.logger.info("    [VirtualEnv] %s" % sys.executable)

            # -------------------------------- batch data loading -------------------------------- #
            t = time.time()

            batch_thcpu, batch_thgpu = self.stepBatch(iterCount=iterCount)

            if ifPrint and ifTracked:
                self.logger.info(
                    "    [Timer] dataLoading Iter%d: %.3f seconds"
                    % (iterCount, time.time() - t)
                )

            # ----------------------------------- Preprocessing ----------------------------------- #
            t = time.time()
            batch_thgpu = self.batchPreprocessingTHGPU(
                batch_thcpu,
                batch_thgpu,
                datasetObjDict=self.datasetObjDict,
                iterCount=iterCount,
            )
            if ifPrint and ifTracked:
                self.logger.info(
                    "    [Timer] batchPreprocessingTHGPU Iter%d: %.3f seconds"
                    % (iterCount, time.time() - t)
                )

            # ------------------------------------ main course ------------------------------------ #
            t = time.time()

            batch_thgpu = self.forwardBackwardUpdate(
                batch_thgpu,
                ifTrain=True,
                iterCount=iterCount,
                ifAllowBackward=True,
                ifRequiresGrad=True,
            )
            if ifPrint and ifTracked:
                self.logger.info(
                    "    [Timer] forwardBackwardUpdate Iter%d: %.3f seconds"
                    % (iterCount, time.time() - t)
                )

            # ---------------------------------------- Meta --------------------------------------- #
            # Storing
            if ifSaveToFinishedIter and ifTracked:
                self.finishedIterCount = iterCount
                self.finishedBatchVis = castAnything(batch_thgpu, "thgpu2np")

            if ifStore and ifTracked and (iterCount > self.resumeIter):
                # save model snapshot
                self.saveModelSnapshot(iterCount=iterCount)

                # visualTrain direct saving (not including visualization)
                batch_vis = castAnything(batch_thgpu, "thgpu2np")
                self.saveBatchVis(batch_vis, iterCount=iterCount)
                # del batch_thgpu

            else:
                # del batch_thgpu
                pass

            if ifBackupTheLatestModels and ifTracked:
                self.saveModelSnapshot(iterCount="latest")

            # monitorImmediateFlow
            if ifMonitorImmediateFlow:
                if self.numMpProcess > 0:
                    dist.barrier(device_ids=[self.rank])

                # with torch.no_grad():
                if True:
                    self.setModelsEvalMode(self.models)
                    if self.numMpProcess > 0:
                        dist.barrier(device_ids=[self.rank])
                    batch_vis = castAnything(batch_thgpu, "thgpu2np")
                    ben = self.stepMonitorImmediateFlow(
                        batch_vis,
                        iterCount=iterCount,
                        ifMonitorDump=True,  # ifMonitorDump,
                    )
                    if self.numMpProcess > 0:
                        dist.barrier(device_ids=[self.rank])
                    self.setModelsTrainingMode(self.models)

                if self.numMpProcess > 0:
                    dist.barrier(device_ids=[self.rank])

            # MonitorVal
            if ifMonitorVal:
                if self.numMpProcess > 0:
                    dist.barrier(device_ids=[self.rank])

                # with torch.no_grad():
                if True:
                    self.setModelsEvalMode(self.models)
                    if self.numMpProcess > 0:
                        dist.barrier(device_ids=[self.rank])
                    ben = self.stepMonitorVal(
                        iterCount=iterCount,
                        ifMonitorDump=ifMonitorDump,
                    )
                    for k in ben.keys():
                        batch_thgpu["statVal" + bt(k)] = nanmean(
                            torch.from_numpy(ben[k]).to(self.cudaDeviceForAll)
                        )
                    if self.numMpProcess > 0:
                        dist.barrier(device_ids=[self.rank])
                    self.setModelsTrainingMode(self.models)

                if self.numMpProcess > 0:
                    dist.barrier(device_ids=[self.rank])

            # Print and Log the Loss and the Benchmarking
            if ifPrint and ifTracked:
                for k in batch_thgpu.keys():
                    if k.startswith("loss"):
                        self.logger.info(
                            "    [Loss] %s Iter%d: %.5f"
                            % (k, iterCount, float(batch_thgpu[k]))
                        )
                    if k.startswith("stat"):
                        self.logger.info(
                            "    [Stat] %s Iter%d: %.5f"
                            % (k, iterCount, float(batch_thgpu[k]))
                        )

            # Print and Log the Time
            if ifPrint and ifTracked:
                self.logger.info(
                    "[Timer] Total is %.3f seconds." % (time.time() - timeIterStart)
                )

            iterCount += 1
            del batch_thgpu
            pass

    def train(self, **kwargs):
        self.trainNoException()
