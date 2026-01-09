import copy
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict, OrderedDict
from socket import gethostname
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch_ema import ExponentialMovingAverage

from functools import partial

from torch.distributed import ReduceOp

from torch.cuda.amp.grad_scaler import GradScaler  # You need to import nerfacc first before this one

from Bprelight4.csvPsnrEntry.benchmarkingPsnrLdrRenderingNerfBlender import (
    benchmarkingPsnrLdrRenderingNerfBlenderFunc,
)
from Bprelight4.csvPsnrEntry.dumpHtmlForPrepickRelightFine import addToSummary0Txt0BrInds

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
    load_general,
    mergeFromAnotherBsv0,
    probe_load_network,
    save_network,
    save_general,
)
from codes_py.toolbox_framework.logger_v1 import LoggerDummy, Logger
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb
from codes_py.toolbox_3D.mesh_io_v1 import dumpPlyPointCloud
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh

from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_torch.hook_v1 import PyTorchForwardHook

from .dataset import PRenderingNerfBlenderDataset, PRenderingLightStageOrigianl3

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def nanmean(x):
    return x[torch.isfinite(x)].mean()


def bt(s):
    return s[0].upper() + s[1:]


class HdrClipRawNerfQuickLoss(nn.Module):  
    # quick loss means it does not consider valid_mask
    # but treats each pixel equally
    def __init__(self, eps=None, hdr_clip=None):
        super(HdrClipRawNerfQuickLoss, self).__init__()
        self.eps = eps if eps is not None else 1.0e-3
        self.hdr_clip = hdr_clip if hdr_clip is not None else 2.5
        
    def forward(self, inputs, targets):
        targets = torch.clamp(targets, max=self.hdr_clip)
        return (((inputs - targets) ** 2) / (torch.abs(inputs).detach() + self.eps)).mean()


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
        self.trackedRank = 0 if self.numMpProcess <= 0 else self.numMpProcess - 1
        self.projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"

        # set device.
        self.cudaDeviceForAll = "cuda:%d" % self.rank
        torch.cuda.set_device(self.cudaDeviceForAll)

        if self.numMpProcess <= 0:  # single threaded
            # assert self.rank == 0
            projRoot_alternative = self.projRoot
            self.logDir = checkPDSRLogDirNew(
                config,
                projRoot=projRoot_alternative,
                ifInitial=config.R.startswith("Rtmp"),
            )
        else:  # DDP
            dist.init_process_group(
                backend="nccl", rank=self.rank, world_size=self.numMpProcess
            )
            dist.barrier(device_ids=[self.rank])
            self.setup_for_distributed(self.rank == self.trackedRank)
            dist.barrier(device_ids=[self.rank])
            projRoot_alternative = self.projRoot
            if self.rank == self.trackedRank:
                self.logDir = checkPDSRLogDirNew(
                    config,
                    # projRoot=self.projRoot,
                    projRoot=projRoot_alternative,
                    ifInitial=config.R.startswith("Rtmp"),
                )
            else:
                # self.logDir = self.projRoot + "v/P/%s/%s/%s/%s/" % (
                self.logDir = projRoot_alternative + "v/P/%s/%s/%s/%s/" % (
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

        # random seeding  # the ending integer

        # """ Unless you are debugging, fixing random seed for learning from such a big dataset seems not very helpful (ctrl-C would restart the dataloader iteration)
        self.seed = self.get_trailing_number(config.R) + self.rank
        self.logger.info('[Random Seed] seed = %d, R = %s, trailing number = %d, rank = %d' %
                      (self.seed, config.R,
                       self.get_trailing_number(config.R), self.rank))
        # enable them only for debugging purpose (fix random seed to chase a particular data point)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # """

        # cuda backend
        torch.backends.cudnn.benchmark = True

        # meta params (not influencing model training)
        self.printFreq = 10
        self.minSnapshotFreq = 40000
        self.samplingVerbose = False
        self.monitorTrainFreq = 2000 if False else 80000
        self.monitorValFreq = 10000 if False else 80000
        self.monitorDumpFreq = 40000
        self.monitorMode = 960 # which iteration is the first to step the monitors

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
            else:
                keys = list(datasetConf.keys())
                for k in keys:
                    if k.startswith("batchSize"):
                        datasetConf[k + "PerProcess"] = datasetConf[k]

        return config

    def metaDataLoading(self, **kwargs):
        config = self.config
        self.logger.info("[Trainer] MetaDataLoading")

        datasetConfDict = config.datasetConfDict

        datasetObjDict = OrderedDict([])

        for datasetConf in datasetConfDict.values():
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                datasetObj = PRenderingNerfBlenderDataset(datasetConf, if_need_metaDataLoading=True)
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
        raise NotImplementedError

    def _netRandomInitialization(self, iter_label):
        # random initialization
        # (Different threads might result in different model weights. Only Rank 0 is used)
        self.logger.info("[Trainer] MetaModelLoading - _netRandomInitialization")

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
                    and ((k not in self.models["meta"]["nonLearningModelNameList"]) or (k in self.models["meta"]["nonLearningButToSave"]))
                ):
                    _, self.models[k] = load_network(
                        self.logDir,
                        self.models[k],
                        k,
                        resumeIter,
                        map_location=self.cudaDeviceForAll,
                    )
            for k in self.optimizerModels.keys():
                _, self.optimizerModels[k] = load_general(
                    self.logDir,
                    self.optimizerModels[k],
                    k,
                    "optimizer",
                    resumeIter,
                    map_location=self.cudaDeviceForAll,
                )
            for k in self.gradScalerModels.keys():
                _, self.gradScalerModels[k] = load_general(
                    self.logDir,
                    self.gradScalerModels[k],
                    k,
                    "gradScaler",
                    resumeIter,
                    map_location=self.cudaDeviceForAll,
                )
            for k in self.schedulerModels.keys():
                _, self.schedulerModels[k] = load_general(
                    self.logDir,
                    self.schedulerModels[k],
                    k,
                    "scheduler",
                    resumeIter,
                    map_location=self.cudaDeviceForAll,
                )
            for k in self.emaModels.keys():
                _, self.emaModels[k] = load_general(
                    self.logDir,
                    self.emaModels[k],
                    k,
                    "ema",
                    resumeIter,
                    map_location=self.cudaDeviceForAll,
                )
        if resumeIter == "latest":
            resumeIter = -1
        return resumeIter

    def _netFinetuningInitialization(self):
        config = self.config
        self.logger.info("[Trainer] MetaModelLoading - _netFinetuningInitialization")
        if hasattr(self.config, "finetuningPDSRI"):
            for k in self.models.keys():
                if (k != "model") and (k != "occGrid"):
                    continue
                if (k == "occGrid") and ((config.P != finetuningPDSRI["P"]) or (config.D != finetuningPDSRI["D"])):
                    continue
                finetuningPDSRI = self.config.finetuningPDSRI
                if (k != "meta") and (
                    k not in self.models["meta"]["nonLearningModelNameList"]
                ):
                    if finetuningPDSRI["P"] == "PrefnerfmlpBerkeley":
                        fn = (
                                self.projRoot
                                + "v/P/%s/%s/%s/%s/models/%s_net_%d.pth"
                                % (
                                    finetuningPDSRI["P"],
                                    finetuningPDSRI["D"],
                                    finetuningPDSRI["S"],
                                    finetuningPDSRI["R"],
                                    "nerfFine",
                                    finetuningPDSRI["I"],
                                )
                            )
                    else:
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
                    assert os.path.isfile(fn), fn
                    with open(fn, "rb") as f:
                        loaded_state_dict = torch.load(f, map_location=self.cudaDeviceForAll)
                    if (config.P == finetuningPDSRI["P"]) and (config.D == finetuningPDSRI["D"]):
                        self.models[k].load_state_dict(  # k == "model"
                            loaded_state_dict,
                            strict=True,
                        )
                    else:
                        keys = list(loaded_state_dict.keys())
                        keys_to_del = [k for k in keys if ((not k.startswith("xyz_encoding_")) and (not k.startswith("sigma.")))]
                        for _ in keys_to_del:
                            del loaded_state_dict[_]
                        self.models[k].load_state_dict(  # k == "model"
                            loaded_state_dict,
                            strict=False,
                        )
        else:
            self.logger.info("    No finetuning specs found")

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
        )  # when you use tcnn with half precision, it is important to use gard_scaler

        # lr_scheduler
        self.schedulerModels = {}
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iterCount: config.scheduler_base ** min(iterCount / 30000, 1)
        )
        self.schedulerModels["all"] = scheduler(self.optimizerModels["all"])

        # ema
        self.emaModels = {}
        self.emaModels["all"] = ExponentialMovingAverage(self.models["model"].parameters(), decay=0.95)

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
        self._optimizerSetups()
        resumeIter = self._netResumingInitialization(iter_label=iter_label)
        # Now resuming also involves resuming the gradScaler and optimizers

        # DDP
        if self.numMpProcess > 0:
            dist.barrier(device_ids=[self.rank])
            for k in self.models.keys():
                if (
                    (k != "meta")
                    and (k not in self.models["meta"]["nonLearningModelNameList"])  # self.models["occGrid"] is ruled out by this line
                ):
                    self.models[k] = torch.nn.parallel.DistributedDataParallel(
                        self.models[k],
                        device_ids=[self.cudaDeviceForAll],
                        # find_unused_parameters=True,  # currently avoid using multi-gpu on this.
                    )
            dist.barrier(device_ids=[self.rank])

        self._netHookSetups(hook_type=hook_type, resumeIter=resumeIter)

        return resumeIter

    def metaLossPrepare(self):
        self.criterion = HdrClipRawNerfQuickLoss()
        self.criterion_mask = torch.nn.MSELoss(reduction="none")

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
                datasetConf_testingOnTraining, if_need_metaDataLoading=False
            )
        elif datasetConf_testingOnTraining["class"] == "RenderingLightStageOriginal3":
            self.testingOnTrainingDatasetObj = PRenderingLightStageOrigianl3(
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
        )[testDataNickName]

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

    def saveModelSnapshot(self, **kwargs):
        iterCount = kwargs["iterCount"]
        if self.rank == self.trackedRank:
 
            for k in self.models.keys():
                if (
                    k != "meta"
                    and ((k not in self.models["meta"]["nonLearningModelNameList"]) or (k in self.models["meta"]["nonLearningButToSave"]))
                ):
                    save_network(
                        self.logDir,
                        self.models[k].module if hasattr(self.models[k], "module") else self.models[k],
                        k,
                        str(iterCount),
                    )
            # optimizerModels
            for k in self.optimizerModels.keys():
                save_general(self.logDir, self.optimizerModels[k], k, "optimizer", str(iterCount))
            # gradScalrModels
            for k in self.gradScalerModels.keys():
                save_general(self.logDir, self.gradScalerModels[k], k, "gradScaler", str(iterCount))
            # schedulerModels
            for k in self.schedulerModels.keys():
                save_general(self.logDir, self.schedulerModels[k], k, "scheduler", str(iterCount))
            # emaModels
            for k in self.emaModels.keys():
                save_general(self.logDir, self.emaModels[k], k, "ema", str(iterCount))

    # You need to determine the "rank" on whether to check anomaly outside of this method
    # sometimes you wish to check on all the ranks, sometimes only one.
    # For bsv0, probably all the ranks. 
    def nanAnomalyCheckOnBatch(self, batch_thgpu, **kwargs):
        message = kwargs["message"]
        iterCount = kwargs['iterCount']
        ifStopHere = kwargs["ifStopHere"]

        flagAnomaly = False
        for k in batch_thgpu.keys():
            if type(batch_thgpu[k]) == torch.Tensor:
                if not torch.all(torch.isfinite(batch_thgpu[k])):
                    print(k)
                    flagAnomaly = True
                    break

        if not flagAnomaly:
            return

        # save the bsv0
        with open(self.logDir + "anomaly/%s_batch_host_%s_iter_%d_rank_%d.pkl" % (
            message, gethostname(), iterCount, self.rank
        ), "wb") as f:
            pickle.dump(castAnything(batch_thgpu, "thgpu2np"), f)

        # also save self.models
        self.saveModelSnapshot(iterCount=iterCount)  # Note that this will save to "models/"

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

    # You need to determine the "rank" on whether to check anomaly outside of this method
    # sometimes you wish to check on all the ranks, sometimes only one.
    # For model, probably only one rank.
    def nanAnomalyCheckOnModel(self, batch_thgpu, **kwargs):
        message = kwargs['message']
        iterCount = kwargs['iterCount']
        ifStopHere = kwargs["ifStopHere"]
        # batch_thgpu = kwargs['batch_thgpu']
        # with open(self.logDir + 'anomaly/%s_thgpu_host_%s_iter_%d_rank_%d.pkl' %
        #         (message, gethostname(), iterCount, self.rank), 'wb') as f:
        #     pickle.dump(batch_thgpu, f)
        models = self.models

        # Check if we need to dump or stop, if not, just exit method
        # Note that in model_checking, the flagAnamoaly is conducted in the per_model fashion
        flagAnamoaly_all = False
        for k in models.keys():
            if (k != 'meta') and (k not in models['meta']['nonLearningModelNameList']):
                if self.numMpProcess <= 0:
                    state_dict = self.models[k].state_dict()
                else:
                    state_dict = self.models[k].module.state_dict()
                flagAnamoaly = False
                for kk, v in state_dict.items():
                    if not torch.all(torch.isfinite(v)):
                        print("Detected that %s %s has nans!" % (k, kk))
                        flagAnamoaly = True
                        break
                if flagAnamoaly:
                    flagAnamoaly_all = True

        if not flagAnamoaly_all:
            return

        # dump the whole self.models
        self.saveModelSnapshot(iterCount=iterCount)  # Note this will save to "models/"

        # also save batch_thgpu / bsv0
        with open(self.logDir + "anomaly/%s_batch_host_%s_iter_%d_rank_%d.pkl" % (
            message, gethostname(), iterCount, self.rank
        ), "wb") as f:
            pickle.dump(castAnything(batch_thgpu, "thgpu2np"), f)

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
                callFlag="stepMonitorImmediateFlow",
                logDir=self.logDir,
                # benchmarking rules
                ifRequiresBenchmarking=True,
                orthogonalDensityResolution=256,  # not affecting PSNR
                marching_cube_thre=0.001,  # not affecting PSNR
                ssimWindow=11,
                ifFit=False,
                # drawing
                ifRequiresDrawing=ifRequiresDrawing,
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
                for meshName in ["fine"]:  # , "depth"]:
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
                        dump_obj_np_fileObj(
                            f,
                            bsv0["%sVert%s" % (meshName, bt(sysLabel))],
                            bsv0["%sFace" % meshName],
                        )

        tmp = {"finalBenValidCount": np.array(len(benRecord.keys()), dtype=np.float32)}
        keys = [
            k[len("finalBen") :]
            for k in list(list(benRecord.values())[0].keys())
            if k.startswith("finalBen")
        ]
        ar = {
            k: np.array(
                [x["finalBen" + k] for x in benRecord.values()], dtype=np.float32
            ).mean()
            for k in keys if k not in ["RuntimeMachine"]
        }
        tmp.update({k: np.array(ar[k], dtype=np.float32) for k in keys if k not in ["RuntimeMachine"]})
        ben = tmp

        return ben

    def stepMonitorTrain(self, batch_vis, **kwargs):
        iterCount = kwargs["iterCount"]
        ifMonitorDump = kwargs["ifMonitorDump"]
        ifIsTrackedRank = self.rank == self.trackedRank
        htmlStepper = self.htmlStepperTrain

        config = self.config
        P = config["P"]
        D = config["D"]
        S = config["S"]
        R = config["R"]

        trainDataEntry = self.trainDataEntry
        datasetObj = trainDataEntry["datasetObj"]
        rank = self.rank
        numMpProcess = self.numMpProcess
        trackedRank = self.trackedRank
        realNumMpProcess = int(max([numMpProcess, 1]))

        indChosen = trainDataEntry["indChosen"]
        if len(indChosen) > 2:
            indChosen = indChosen[:2]
        indVisChosen = trainDataEntry["indVisChosen"]
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
        trainDataNickName = trainDataEntry["testDataNickName"]
        benRecord = {}
        # j2 = j1 + 1  # debug
        for j in range(j1, j2):
            index = indChosen[j]
            print(
                "[Trainer Visualizer Train] Progress Iter%d: "
                "trainDataNickName %s, index = %d, j = %d, j1 = %d, j2 = %d, "
                "rank = %d, numMpProcess = %d, j_tot = %d"
                % (
                    iterCount,
                    trainDataNickName,
                    index,
                    j,
                    j1,
                    j2,
                    rank,
                    numMpProcess,
                    j_tot,
                )
            )
            batch0_np = trainDataEntry["datasetObj"].getOneNP(index)
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

            if trainDataNickName.startswith(
                "renderingNerfBlender"
            ) or trainDataNickName.startswith("renderingCapture"):
                bsv0 = benchmarkingPsnrLdrRenderingNerfBlenderFunc(
                    bsv0_initial,
                    # misc
                    cudaDevice=self.cudaDeviceForAll,
                    datasetObj=trainDataEntry["datasetObj"],
                    iterCount=iterCount,
                    # benchmarking rules
                    orthogonalDensityResolution=256,  # not affecting PSNR
                    marching_cube_thre=0.001,  # not affecting PSNR
                    ssimWindow=11,
                    ifFit=False,
                    # drawing
                    ifRequiresDrawing=ifRequiresDrawing,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=self.models,
                    doPred0Func=self.doPred0,
                    config=self.config,
                )
            else:
                raise NotImplementedError(
                    "Unknown trainDataNickName: %s" % trainDataNickName
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
                subMessage = "PSNR Coarse: %.3f, PSNR Fine: %.3f" % (
                    bsv0["finalBenPSNRCoarse"],
                    bsv0["finalBenPSNRFine"],
                )
                htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
            if ifMonitorDump and ifIsTrackedRank and ifRequiresDrawing:
                for meshName in ["coarse", "fine"]:
                    sysLabel = "world"
                    with open(
                        self.monitorTrainLogDir
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
                        dump_obj_np_fileObj(
                            f,
                            bsv0["%sVert%s" % (meshName, bt(sysLabel))],
                            bsv0["%sFace" % meshName],
                        )

        ar = []
        keys = [
            k[len("finalBen") :]
            for k in list(benRecord.values())[0].keys()
            if k.startswith("finalBen")
        ]
        for br in benRecord.values():
            ar.append(
                np.array([br["finalBen%s" % bt(k)] for k in keys], dtype=np.float32)
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
                    callFlag="stepMonitorVal",
                    logDir=self.logDir,
                    # benchmarking rules
                    ifRequiresBenchmarking=True,
                    orthogonalDensityResolution=256,  # not affecting PSNR
                    marching_cube_thre=0.001,  # not affecting PSNR
                    ssimWindow=11,
                    ifFit=False,
                    # drawing
                    ifRequiresDrawing=ifRequiresDrawing,
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
                for meshName in ["fine"]:  # , "depth"]:
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
                        dump_obj_np_fileObj(
                            f,
                            bsv0["%sVert%s" % (meshName, bt(sysLabel))],
                            bsv0["%sFace" % meshName],
                        )

        if len(list(benRecord.values())) > 0:
            ar = []
            keys = [
                k[len("finalBen") :]
                for k in list(benRecord.values())[0].keys()
                if k.startswith("finalBen")
            ]
            for br in benRecord.values():
                ar.append(
                    np.array([br["finalBen%s" % bt(k)] for k in keys if k not in ["RuntimeMachine"]], dtype=np.float32)
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
        else:
            ben = {}
            ben["finalBenValidCount"] = np.array(0, dtype=np.float32)
        return ben

    def stepBatch(self, **kwargs):
        config = self.config
        tmp = OrderedDict([])
        for dataset in config.datasetConfDict.keys():
            # try:
            #     batchDid_thcpu = next(self.dataLoaderIterDict[dataset])
            # except:
            #     self.dataLoaderIterDict[dataset] = iter(self.dataLoaderDict[dataset])
            #     batchDid_thcpu = next(self.dataLoaderIterDict[dataset])
            datasetConf = config.datasetConfDict[dataset]
            datasetObj = self.datasetObjDict[dataset]
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                ind = np.random.randint(
                    0,
                    datasetObj.m_rays,
                    datasetConf["batchSizePerProcess"],
                    dtype=np.int64,
                )
                # we do the random colors for all the pixels - just that for 
                #   foreground pixels (valid_mask is True)
                #   These random colors will not be useful at all.
                # Since the real light stage does not have the ground truth alpha channel
                #   our applying of the random_background_color can only be hard 
                #   (so not really soft compositing between random_background_color and the rgbs)
                #   This means, for pixels whose alpha channel is between 0 and 1, 
                #   the "random background color" would be 0 (for white_background == False)
                #   or be 1 (for white_background == True)
                # The purpose of introducing the background color is mainly for
                #   letting the random colors of the background to shine through
                # This background color things will be useful only under D0bgrandom
                #   D0 / D00 / D01 won't be using random_background_color at all.
                random_background_color = np.random.rand(
                    datasetConf["batchSizePerProcess"], 3
                ).astype(np.float32)
                # Make only the background part to be random,
                #   foreground part to be 0 or 1 (according to whether 
                #   config.white_background is True or False)
                valid_mask = datasetObj.all_masks[ind]
                random_background_color[np.where(valid_mask), :] = (
                    1 if config.white_back else 0
                )  # OK this has already done so!
                batchDid_thcpu = {
                    "rgbs": torch.from_numpy(
                        datasetObj.all_rgbs[ind].astype(np.float32) / 255.0
                    ),
                    # "valid_mask": torch.from_numpy(datasetObj.all_masks[ind]),
                    "valid_mask": torch.from_numpy(valid_mask),
                    "random_background_color": torch.from_numpy(random_background_color),
                    # "dilated_mask": torch.from_numpy(datasetObj.all_dilates[ind]),
                    "indexID": torch.from_numpy(datasetObj.all_indexID[ind]),
                    "pixelID": torch.from_numpy(datasetObj.all_pixelID[ind]),
                    "hdrs": torch.from_numpy(datasetObj.all_hdrs[ind]),
                }
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
            # batch_thgpu["caseSplitInsideIndexList_%s" % dataset] = torch.from_numpy(
            #     caseSplitInsideIndexList
            # ).long().to(self.cudaDeviceForAll)  # requested by nerfstudio
        elif datasetConf["class"] == "RenderingLightStageOriginal3":
            # indexID = batch_thcpu["indexID_%s" % dataset].numpy()  # 0~179
            # focalLengthWidth = datasetObj.focalLengthWidth[indexID]
            # focalLengthHeight = datasetObj.focalLengthHeight[indexID]
            # c2w = datasetObj.c2w[indexID]
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

            random_background_color = batch_thcpu["random_background_color_%s" % dataset].numpy()

            rays = np.concatenate([
                (c2w[:, :3, 3] - sceneShiftFirst[None, :]) * sceneScaleSecond,  # rays_o
                directions_normalized,  # rays_d
                # datasetConf["ray_near"] * np.ones((indexID.shape[0], 1), dtype=np.float32),
                # datasetConf["ray_far"] * np.ones((indexID.shape[0], 1), dtype=np.float32),
                np.nan * np.zeros((indexID.shape[0], 4), dtype=np.float32),
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
        datasetConf = datasetObj.datasetConf
        models = kwargs["models"]
        config = kwargs["config"]
        cudaDevice = kwargs["cudaDevice"]
        orthogonalDensityResolution = kwargs["orthogonalDensityResolution"]
        marching_cube_thre = kwargs["marching_cube_thre"]
        callFlag=kwargs["callFlag"]
        logDir = kwargs["logDir"]
        lgtMode = kwargs.get("lgtMode", "pointLightMode")
        cls.assertModelEvalMode(models)
        rays_thgpu = torch.from_numpy(bsv0["rays"]).to(cudaDevice)

        with torch.no_grad():
        # if True:  # debug the normal also for the test cases
            # with torch.cuda.amp.autocast(enabled=True):
            if rays_thgpu.shape[0] == 2048 * 1366:
                # chunk = 1 * 1366  # 683 * 64
                chunk = 683
            elif rays_thgpu.shape[0] == 800 * 800:
                chunk = 4 * 800
            else:
                # raise NotImplementedError("Unknown rays_thgpu.shape[0] = %d" % (rays_thgpu.shape[0]))
                # chunk = rays_thgpu.shape[0]
                chunk = 1024
            # assert rays_thgpu.shape[0] % chunk == 0
            tot = int(math.ceil(float(rays_thgpu.shape[0]) / chunk))
            results_list = []
            
            # if datasetConf["lgtLoadingMode"] == "ENVMAP":
            if ("envmaps" in datasetConf.keys()):
                # Assumes all the envmaps samples are the same
                # forward the nerf appearance parameters just for once (to accelerate)

                hyperOutputs = None
                envmaps_reference = None
               
                tmp = bsv0["envmaps"].reshape((bsv0["envmaps"].shape[0], datasetConf["envmapHeight"] * datasetConf["envmapWidth"] * 3))
                assert np.all(tmp.min(0) == tmp.max(0))
                
                # bsv0["envmap"] = bsv0["envmaps"][0, :, :, :]

            t0 = time.time()
            for i in range(0, rays_thgpu.shape[0], chunk):
                # print("doPred0: %d / %d" % (i, rays_thgpu.shape[0]))
                # elif datasetConf["lgtLoadingMode"] == "ENVMAP":
                if ("envmaps" in bsv0.keys()):
                    lgtInput = {
                        k: torch.from_numpy(
                            bsv0[k][i : i + chunk, :]
                        ).to(rays_thgpu.device)
                        for k in ["envmaps"]
                    }
                    if hyperOutputs is None:
                        hyperOutputs = models["model"].hyper(
                            lgtInput["envmaps"][:1, :, :, :].view(
                                1, datasetConf["envmapHeight"] * datasetConf["envmapWidth"] * 3
                            )
                        )
                        hyperOutputs = {
                            k: v.repeat(chunk, 1, 1) for k, v in hyperOutputs.items()
                        }
                        envmaps_reference = lgtInput["envmaps"][0, :, :, :]
                    assert torch.all(lgtInput["envmaps"].min(0).values == envmaps_reference)
                    assert torch.all(lgtInput["envmaps"].max(0).values == envmaps_reference)
                    
                # if datasetConf["lgtLoadingMode"] == "QSPL":
                elif ("lgtE" in bsv0.keys()):
                    lgtInput = {
                        k: torch.from_numpy(
                            bsv0[k][i : i + chunk, :]
                        ).to(rays_thgpu.device)
                        for k in ["lgtE", "objCentroid"]
                    }
                    hyperOutputs = None
                elif ("lgtE0" in bsv0.keys()):
                    # Coding principle: lgtE0 --> lgtE and objCentroid0 --> objCentroid np.tile or torch.Tensor.repeat does not cost huge memory
                    # However, do not do the same to envmap16x32. 512 dim is not like 3 dim, when you are tiling (1366x2048 times)
                    lgtInput = {
                        "lgtE": torch.from_numpy(bsv0["lgtE0"]).float().to(rays_thgpu.device)[None, :].repeat(
                            rays_thgpu.shape[0], 1
                        ),
                        "objCentroid": torch.from_numpy(bsv0["objCentroid0"]).float().to(rays_thgpu.device).repeat(
                            rays_thgpu.shape[0], 1
                        ),
                    }
                    hyperOutputs = None
                else:
                    # raise ValueError("Unknown datasetConf['lgtLoadingMode']: %s" % datasetConf["lgtLoadingMode"])
                    raise ValueError("It is not clear which of the branch it should go to.")
                # print("doPred0: %d / %d" % (i, rays_thgpu.shape[0]))
                results_list.append(cls.forwardRendering(
                    rays_thgpu[i:i + chunk],
                    models=models,
                    config=config,
                    iterCount=int(bsv0["iterCount"]),
                    cpu_ok=True,
                    force_all_rays=True,
                    requires_grad=False,  # debug the normal also for the test case
                    if_query_the_expected_depth=True,
                    lgtInput=lgtInput,
                    callFlag=callFlag,
                    logDir=logDir,
                    hyperOutputs=hyperOutputs,
                    lgtMode=lgtMode,
                    envmap0_thgpu=None if (lgtMode != "envmapMode") else (
                        torch.from_numpy(bsv0["envmap0"]).to(rays_thgpu.device)
                    ),
                    specRoughnessID=bsv0.get("specRoughnessID", None),
                ))
                # print((i, rays_thgpu.shape[0]))
                # torch.cuda.empty_cache()
            t1 = time.time()
            bsv0["timeElapsed"] = t1 - t0
            bsv0["timeEvalMachine"] = gethostname()
            results = {k: torch.cat([x[k] for x in results_list], 0) for k in results_list[0].keys()}
        for k in results.keys():
            assert "float32" in str(results[k].dtype), k
        # bsv0["rgbFinePred"] = results["rgb_fine"].detach().cpu().numpy()
        winWidth = int(bsv0["winWidth"])
        winHeight = int(bsv0["winHeight"])
        if winWidth * winHeight > rays_thgpu.shape[0]:  # the distill case
            bsv0["imghdrFinePred"] = results["hdr_fine"].detach().cpu().numpy()
            bsv0["imghdr2FinePred"] = results["hdr2_fine"].detach().cpu().numpy()
            bsv0["imghdr3FinePred"] = results["hdr3_fine"].detach().cpu().numpy()
            return bsv0
        else:
            assert winWidth * winHeight == rays_thgpu.shape[0]
        hdrFinePred = results["hdr_fine"].detach().cpu().numpy()
        bsv0["imghdrFinePred"] = hdrFinePred.reshape((winHeight, winWidth, 3))
        if "hdr2_fine" in results.keys():
            bsv0["imghdr2FinePred"] = results["hdr2_fine"].detach().cpu().numpy().reshape((winHeight, winWidth, 3))
        if "hdr3_fine" in results.keys():
            bsv0["imghdr3FinePred"] = results["hdr3_fine"].detach().cpu().numpy().reshape((winHeight, winWidth, 3))
        ldrFinePred = torch.clamp(tonemap_srgb_to_rgb(results["hdr_fine"].detach()), min=0, max=1).cpu().numpy()
        bsv0["imgFinePred"] = ldrFinePred.reshape((winHeight, winWidth, 3))
        bsv0["ldrFinePred"] = ldrFinePred  # benchmarking needs this one
        # bsv0["imgFinePred"] = bsv0["rgbFinePred"].reshape((winHeight, winWidth, 3))

        bsv0["depthFinePred"] = (
            results["depth_fine"].detach().cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        )
        bsv0["opacityFinePred"] = (
            results["opacity_fine"].detach().cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        )
        if "normal2_fine" in results.keys():
            bsv0["normal2"] = results["normal2_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "normal3_fine" in results.keys():
            bsv0["normal3"] = results["normal3_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "normal2DotH_fine" in results.keys():
            bsv0["normal2DotH"] = results["normal2DotH_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        if "normal3DotH_fine" in results.keys():
            bsv0["normal3DotH"] = results["normal3DotH_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsPointlightOpacities_fine" in results.keys():
            bsv0["hintsPointlightOpacities"] = results["hintsPointlightOpacities_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        for i in range(4):
            if ("hintsPointlightGGX%d_fine" % i) in results.keys():
                bsv0["hintsPointlightGGX%d" % i] = results["hintsPointlightGGX%d_fine" % i].detach(
                    ).cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefOpacities_fine" in results.keys():
            bsv0["hintsRefOpacities"] = results["hintsRefOpacities_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth)).astype(np.float32)
        if "hintsRefSelf_fine" in results.keys():
            bsv0["hintsRefSelf"] = results["hintsRefSelf_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth, 3)).astype(np.float32)
        if "hintsRefLevels_fine" in results.keys():
            bsv0["hintsRefLevels"] = results["hintsRefLevels_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth, -1)).astype(np.float32)
        if "hintsRefLevelsColor_fine" in results.keys():
            bsv0["hintsRefLevelsColor"] = results["hintsRefLevelsColor_fine"].detach(
                ).cpu().numpy().reshape((winHeight, winWidth, -1, 3)).astype(np.float32)

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
                t = models["meta"]["density_fn"](
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
        B = rays.shape[0]
        results = defaultdict(list)
        # batch_thgpu = kwargs["batch_thgpu"]  # debug purpose - for inspecting - do not use for computation

        model = models["model"]
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        bg_color = rays[:, 10:13]

        # outputs = model.render(
        outputs = models["meta"]["render_fn"](
            rays_o, rays_d, occGrid=models["occGrid"], staged=False, bg_color=bg_color, perturb=True, force_all_rays=force_all_rays,
            min_near=0.2, density_thresh=10, bg_radius=-1,
            # batch_thgpu=batch_thgpu,  # debug purpose
            iterCount=iterCount,
            requires_grad=requires_grad,
            if_query_the_expected_depth=if_query_the_expected_depth,
            lgtInput=lgtInput,
            callFlag=callFlag,
            logDir=logDir,
            model14=models.get("model14", None),
            hyperOutputs=hyperOutputs,
            lgtMode=lgtMode,
            envmap0=envmap0,
            specRoughnessID=kwargs.get("specRoughnessID", None),
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

    def forwardNetNeRF(self, batch_thgpu, **kwargs):
        models = kwargs["models"]
        config = kwargs["config"]
        dataset = kwargs["dataset"]
        iterCount = kwargs["iterCount"]

        wl = config.wl

        self.assertModelsTrainingMode(models)

        models["occGrid"].update_every_n_steps(
            step=iterCount - max(self.resumeIter, 0),
            occ_eval_fn=partial(models["model"].density, if_do_bound_clamp=True, if_output_only_sigma=True),
            occ_thre=10,
            n=16,
        )
        if iterCount < config.empty_cache_stop_iter:
            torch.cuda.empty_cache()

        _dataset = "_" + dataset
        # For this mode, you need to modify the rays[:, 10:13]
        # (where valid_mask == False pixels need to be set to 1 or 0)
        lgtInput = {
            "lgtE": batch_thgpu["lgtE" + _dataset],
            "objCentroid": batch_thgpu["objCentroid" + _dataset],
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
        )

        assert torch.all(
            batch_thgpu["random_background_color_%s" % dataset] == 
            batch_thgpu["rays_%s" % dataset][:, 10:13]
        )
        gt_hdr = torch.where(
            batch_thgpu["valid_mask_%s" % dataset][:, None].repeat(1, 3),
            batch_thgpu["hdrs_%s" % dataset],
            batch_thgpu["random_background_color_%s" % dataset],
        )
        
        lossMain = wl["lossMain"] * self.criterion(results["hdr_fine"], gt_hdr).nanmean()
        if ("lossMask" in wl.keys()) and (wl["lossMask"] > 0):
            lossMask = wl["lossMask"] * (
                (1.0 - batch_thgpu["valid_mask" + _dataset].float()) 
                * self.criterion_mask(
                    results["opacity_fine"], torch.zeros_like(results["opacity_fine"])
                )
            ).mean()
        else:
            lossMask = 0

        if ("lossTie" in wl.keys()) and (wl["lossTie"] > 0):
            lossTie = wl["lossTie"] * results["lossTieRays_fine"].mean()
        else:
            lossTie = 0

        if ("lossBackface" in wl.keys()) and (wl["lossBackface"] > 0):
            lossBackface = wl["lossBackface"] * results["lossBackfaceRays_fine"].mean()
        else:
            lossBackface = 0

        if ("lossBfdirect" in wl.keys()) and (wl["lossBfdirect"] > 0) and (iterCount > config.lossBfdirectStartingIter + self.monitorMode):
            lossBfdirect = wl["lossBfdirect"] * results["lossBfdirectRays_fine"].mean()
        else:
            lossBfdirect = 0
        
        batch_thgpu["lossMain"] = lossMain
        batch_thgpu["lossMask"] = lossMask
        batch_thgpu["lossTie"] = lossTie
        batch_thgpu["lossBackface"] = lossBackface
        batch_thgpu["lossBfdirect"] = lossBfdirect
        batch_thgpu["loss"] = lossMain + lossMask + lossTie + lossBackface + lossBfdirect

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
                gradScalerModels=self.gradScalerModels,
            )

        return batch_thgpu

    # @classmethod
    def backwardLoss(self, batch_thgpu, **kwargs):
        iterCount = kwargs["iterCount"]
        optimizerModels = kwargs["optimizerModels"]
        gradScalerModels = kwargs["gradScalerModels"]

        if iterCount > 0:
            optimizerModels["all"].zero_grad()
            # batch_thgpu["loss"].backward()
            # optimizerModels["all"].step()
            gradScalerModels["all"].scale(batch_thgpu["loss"]).backward()
            gradScalerModels["all"].step(optimizerModels["all"])
            # scale = gradScalerModels["all"].get_scale()
            gradScalerModels["all"].update()

            self.schedulerModels["all"].step()
            self.emaModels["all"].update()

        return batch_thgpu

    def trainNoException(self, **kwargs):
        # pyrenderManager = kwargs["pyrenderManager"]

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
            power10 = min(4, power10)
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

                # visualVal forwarding and saving (not including visualization)
                # self.saveValBatchVis(iterCount=iterCount)
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
