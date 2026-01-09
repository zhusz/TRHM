import copy

import numpy as np

from .renderingNerfBlender.renderingNerfBlenderDatasetNew2 import (
    RenderingNerfBlenderDataset,
)
from .renderingNerfBlender.renderingLightStageOriginal3 import (
    RenderingLightStageOriginal3
)
from .renderingNerfBlender.renderingLightStageLfhyper import RenderingLightStageLfhyper
from .renderingNerfBlender.renderingNerfBlenderDatasetLfhyper import (
    RenderingNerfBlenderDatasetLfhyper,
    RenderingNerfBlenderDatasetDemoEnvmap,
)


def bt(s):
    return s[0].upper() + s[1:]


def xt(s):
    return s[0].lower() + s[1:]


nerfSynthetic_blenderFrameTagList = [
    "0000", "0001", "0136", "0029",
    "0001", "0000", "0186", "0002",
]
nerfSynthetic_blenderFrameTagList *= 4

nerfSynthetic_hdrScalingList = [
    1.0, 4.0, 4.0, 1.0,
    2.0, 1.0, 4.0, 1.0,  # 4.0 / 16.0 was because it was too dark

    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 4.0, 1.0,

    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,  # We do not do hdrScaling 4 times under SSS

    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
]
# nerfSynthetic_hdrScalingList *= 4

nerfSyntheticHyper_hyperHdrScalingList = [
    4.0, 4.0, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0,
]

# capture7jfHyper_hyperHdrScalingList = [
capture7jf_hdrScalingList = [
    None,
    1.0, 1.0, 1.0,  # red_candle, jade_dragon, jade_fish
    None,
    1.0, 1.0, 1.0, 1.0,  # candle_cake, soap_lavender, silicon_blue_form, candle_face
    None,
    1.0,  # white_candle_s
]
capture7jfHyper_hyperHdrScalingList = [
    None,
    4.0, 4.0, 4.0,
    None,
    4.0, 4.0, 4.0, 4.0,
    None,
    4.0,
]


def getTestDataEntryDict(**kwargs):
    wishedTestDataNickName = kwargs["wishedTestDataNickName"]
    fyiDatasetConf = kwargs["fyiDatasetConf"]
    cudaDevice = kwargs.get("cudaDevice", None)
    assert type(wishedTestDataNickName) is list or wishedTestDataNickName is None

    testDataEntryDict = {}

    # --------------------- renderingNerfBlender1CaseXSplitVal --------------------- #
    for testDataNickName in wishedTestDataNickName:
        if testDataNickName.startswith("rendering") or testDataNickName.startswith("capture7jf") or (
            testDataNickName.startswith("lfhyperDemo")
        ):
            assert testDataNickName.find("Case") == testDataNickName.rfind("Case")
            loc_Case = testDataNickName.find("Case")
            assert testDataNickName.find("Split") == testDataNickName.rfind("Split")
            loc_Split = testDataNickName.find("Split")
            dataset = testDataNickName[:loc_Case]
            caseID = int(testDataNickName[loc_Case + 4 : loc_Split])
            split = testDataNickName[loc_Split + 5 :].lower()
            if (
                testDataNickName.startswith("renderingNerfBlenderExpp500")
                or testDataNickName.startswith("renderingNerfBlenderExpp600")
                or testDataNickName.startswith("renderingNerfBlenderExisting")
            ):  # nerf square
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                # datasetConf["hdrScaling"] = fyiDatasetConf.hdrScaling
                datasetConf["winWidth"] = 800
                datasetConf["winHeight"] = 800
                datasetConf["getRaysMeans"] = fyiDatasetConf.getRaysMeans
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = True
                datasetConf["ifLoadHdr"] = False
                datasetConf["ifLoadDepth"] = fyiDatasetConf.get("ifLoadDepth", False)
                datasetConf["depthFileListA0Tag"] = fyiDatasetConf.get("depthFileListA0Tag", None)
                datasetConf["blenderFrameTag"] = fyiDatasetConf.get("blenderFrameTag", None)
                datasetConf["ifLoadNormal"] = fyiDatasetConf.get("ifLoadNormal", False)
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtLoadingMode"] = "NONE"
            elif (
                testDataNickName.startswith("renderingNerfBlender58Pointk8rot4")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["hdrScaling"] = nerfSynthetic_hdrScalingList[caseID]  # 4.0 if caseID in [6, 7] else 1.0
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                datasetConf["winWidth"] = 800
                datasetConf["winHeight"] = 800
                datasetConf["getRaysMeans"] = fyiDatasetConf.getRaysMeans
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = False
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtLoadingMode"] = "QSPL"
                datasetConf["ifLoadDepth"] = fyiDatasetConf.ifLoadDepth
                datasetConf["loadDepthTag"] = fyiDatasetConf.loadDepthTag
            elif (
                testDataNickName.startswith("renderingNerfBlender58Point")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["hdrScaling"] = nerfSynthetic_hdrScalingList[caseID]  # 4.0 if caseID in [6, 7] else 1.0
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                datasetConf["winWidth"] = fyiDatasetConf.winWidth  # mostly 800, but it can also be 400
                datasetConf["winHeight"] = fyiDatasetConf.winHeight  # mostly 800, but it can also be 400
                datasetConf["getRaysMeans"] = fyiDatasetConf.getRaysMeans
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = True
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtLoadingMode"] = "QSPL"
                datasetConf["ifLoadDepth"] = fyiDatasetConf.ifLoadDepth
                datasetConf["loadDepthTag"] = fyiDatasetConf.loadDepthTag

                datasetConf["ifPreloadHighlight"] = True  # evaluate the highlight and sublight pixels
                datasetConf["highlightSplit"] = 3  # testing mode only keeps testing cases, training mdoe (not here) only keeps training cases
            elif (
                testDataNickName.startswith("renderingNerfBlender58EnvolatGray16x32k8distillV100")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingNerfBlenderDatasetLfhyper"
                datasetConf["winWidth"] = fyiDatasetConf.winWidth
                datasetConf["winHeight"] = fyiDatasetConf.winHeight
                datasetConf["ray_near"] = fyiDatasetConf.ray_near
                datasetConf["ray_far"] = fyiDatasetConf.ray_far
                datasetConf["rays_background_pad_width"] = 0
                datasetConf["rays_background_pad_height"] = 0
                datasetConf["getRaysMeans"] = fyiDatasetConf.getRaysMeans
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["hyperHdrScaling"] = nerfSyntheticHyper_hyperHdrScalingList[caseID]
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = True
                datasetConf["ifLoadDepth"] = False
                datasetConf["loadDepthTag"] = None
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtDataset"] = fyiDatasetConf.lgtDataset
                datasetConf["lgt_quantile_cut_min"] = fyiDatasetConf.lgt_quantile_cut_min
                datasetConf["lgt_quantile_cut_max"] = fyiDatasetConf.lgt_quantile_cut_max
                datasetConf["lgt_quantile_cut_fixed"] = fyiDatasetConf.lgt_quantile_cut_fixed
                datasetConf["if_lgt_load_highres"] = False
                datasetConf["debugReadTwo"] = False
            elif (
                testDataNickName.startswith("lfhyperDemoRenderingNerfBlenderExistingLgt")
            ):
                datasetConf = {}
                assert testDataNickName.startswith("lfhyperDemo")
                assert testDataNickName.find("Lgt") > 0
                datasetConf["dataset"] = xt(dataset[len("lfhyperDemo"):dataset.find("Lgt")])
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingNerfBlenderDatasetDemoEnvmap"
                datasetConf["getRaysMeans"] = fyiDatasetConf.getRaysMeans
                datasetConf["winWidth"] = fyiDatasetConf.winWidth
                datasetConf["winHeight"] = fyiDatasetConf.winHeight
                datasetConf["rays_background_pad_width"] = 0
                datasetConf["rays_background_pad_height"] = 0
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["hyperHdrScaling"] = nerfSyntheticHyper_hyperHdrScalingList[caseID]
                datasetConf["lgtDataset"] = xt(dataset[dataset.find("Lgt"):])
                datasetConf["lgt_quantile_cut_min"] = None
                datasetConf["lgt_quantile_cut_max"] = None
                datasetConf["lgt_quantile_cut_fixed"] = 1.0
                datasetConf["if_lgt_load_highres"] = True

                datasetConf["lgtLoadingMode"] = "ENVMAP"
            elif (
                testDataNickName.startswith("renderingCapture7jfHdrPointrottest3")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["hdrScaling"] = 1.0
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                datasetConf["winWidth"] = 1366
                datasetConf["winHeight"] = 2048
                datasetConf["getRaysMeans"] = "ELU"  # fyiDatasetConf.getRaysMeans
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = False
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtLoadingMode"] = "QSPL"
            elif (
                testDataNickName.startswith("renderingCapture7jfEnvmapStandardBlenderRot1")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["hdrScaling"] = 1.0
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                datasetConf["winWidth"] = 1366
                datasetConf["winHeight"] = 2048
                datasetConf["getRaysMeans"] = "ELU"  # fyiDatasetConf.getRaysMeans
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = False
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtLoadingMode"] = "ENVMAP"
            elif (
                testDataNickName.startswith("renderingCapture7jfHdrPointFree")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["hdrScaling"] = 1.0
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                datasetConf["winWidth"] = 800
                datasetConf["winHeight"] = 800
                datasetConf["getRaysMeans"] = "ELU"
                datasetConf["ray_near"] = np.nan
                datasetConf["ray_far"] = np.nan
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = False
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["ifLoadDepth"] = fyiDatasetConf.ifLoadDepth
                datasetConf["loadDepthTag"] = fyiDatasetConf.loadDepthTag
                datasetConf["ifLoadNormal"] = fyiDatasetConf.get("ifLoadNormal", True)
                datasetConf["lgtLoadingMode"] = "QSPL"
            elif (
                testDataNickName.startswith("renderingCapture7jfHdrEnvmapFree") or
                testDataNickName.startswith("renderingCapture7jfHdrEnvmapLgtStandardBlender1k") or
                testDataNickName.startswith("renderingCapture7jfHdrEnvmapTraj")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["hdrScaling"] = 1.0
                datasetConf["class"] = "RenderingNerfBlenderDataset"
                datasetConf["winWidth"] = 800
                datasetConf["winHeight"] = 800
                datasetConf["getRaysMeans"] = "ELU"
                datasetConf["ray_near"] = np.nan
                datasetConf["ray_far"] = np.nan
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["white_back"] = fyiDatasetConf.white_back
                datasetConf["ifLoadImg"] = False
                datasetConf["ifLoadHdr"] = False
                datasetConf["ifLoadMaskYesLoss"] = False
                datasetConf["ifMaskDilate"] = 0
                datasetConf["lgtLoadingMode"] = "ENVMAPLFHYPER"
                datasetConf["lgt_quantile_cut_min"] = None  # fyiDatasetConf.lgt_quantile_cut_min
                datasetConf["lgt_quantile_cut_max"] = None  # fyiDatasetConf.lgt_quantile_cut_max
                datasetConf["lgt_quantile_cut_fixed"] = 1.0  # important to standardize this
                datasetConf["if_lgt_load_highres"] = True
            elif (
                testDataNickName.startswith("capture7jfLfhyperDemo")
            ):
                # testing mode only
                # lgtDataset can be arbitrary
                # capture7jfHyperDemoLgtXXXCase"$r"SplitAlltest
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageLfhyper"
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                assert testDataNickName.startswith("capture7jfLfhyperDemoLgt"), testDataNickName  # because the lgtDataset (including envmap lgt datasets) are all required to starts with "lgt"
                tmp = testDataNickName[len("capture7jfLfhyperDemo"):loc_Case]
                lgtDataset = tmp[0].lower() + tmp[1:]
                datasetConf["lgtDataset"] = lgtDataset
                # the following two are not used under the testing mode
                datasetConf["lgt_quantile_cut_min"] = None  # fyiDatasetConf.lgt_quantile_cut_min
                datasetConf["lgt_quantile_cut_max"] = None  # fyiDatasetConf.lgt_quantile_cut_max
                datasetConf["lgt_quantile_cut_fixed"] = 1.0  # important to standardize this
                datasetConf["hyperHdrScaling"] = capture7jfHyper_hyperHdrScalingList[caseID]
                datasetConf["if_lgt_load_highres"] = True
            elif (
                testDataNickName.startswith("capture7jfLfhyper")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageLfhyper"
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["lgtDataset"] = fyiDatasetConf.lgtDataset
                datasetConf["lgt_quantile_cut_min"] = fyiDatasetConf.lgt_quantile_cut_min
                datasetConf["lgt_quantile_cut_max"] = fyiDatasetConf.lgt_quantile_cut_max
                datasetConf["lgt_quantile_cut_fixed"] = 1.0  # important to standardize this
                datasetConf["hyperHdrScaling"] = capture7jfHyper_hyperHdrScalingList[caseID]
                datasetConf["if_lgt_load_highres"] = fyiDatasetConf.if_lgt_load_highres
            elif (
                testDataNickName.startswith("capture7jfHyper") and not testDataNickName.startswith("capture7jfHyperDemo")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageHyper"
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["lgtDataset"] = fyiDatasetConf.lgtDataset
                datasetConf["lgt_quantile_cut_min"] = fyiDatasetConf.lgt_quantile_cut_min
                datasetConf["lgt_quantile_cut_max"] = fyiDatasetConf.lgt_quantile_cut_max
                datasetConf["lgt_quantile_cut_fixed"] = 1.0  # important to standardize this
                datasetConf["hyperHdrScaling"] = capture7jfHyper_hyperHdrScalingList[caseID]
            elif (
                testDataNickName.startswith("capture7jfHyperDemo")
            ):
                # testing mode only
                # lgtDataset can be arbitrary
                # capture7jfHyperDemoLgtXXXCase"$r"SplitAlltest
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageHyper"
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                assert testDataNickName.startswith("capture7jfHyperDemoLgt")  # because the lgtDataset (including envmap lgt datasets) are all required to starts with "lgt"
                tmp = testDataNickName[len("capture7jfHyperDemo"):loc_Case]
                lgtDataset = tmp[0].lower() + tmp[1:]
                datasetConf["lgtDataset"] = lgtDataset
                # the following two are not used under the testing mode
                datasetConf["lgt_quantile_cut_min"] = None  # fyiDatasetConf.lgt_quantile_cut_min
                datasetConf["lgt_quantile_cut_max"] = None  # fyiDatasetConf.lgt_quantile_cut_max
                datasetConf["lgt_quantile_cut_fixed"] = 1.0  # important to standardize this
                datasetConf["hyperHdrScaling"] = capture7jfHyper_hyperHdrScalingList[caseID]
            elif (
                testDataNickName.startswith("capture7jfLfdistill")
            ):  # Only deal with the lens-flare-corrupted frames
                datasetConf = {}
                datasetConf["dataset"] = dataset
                assert testDataNickName.startswith("capture7jfLfdistillCase")  # no need to specify something like 512
                datasetConf["bucket"] = "lfdistill"
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageOriginal3"
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["singleImageForegroundZoneNumberUpperBound"] = -1
                datasetConf["singleImageBackgroundZoneNumberLowerBound"] = -1
                datasetConf["lgtLoadingMode"] = "QSPL"  # It is pointLight. There is no ambiguity here.
            elif (
                testDataNickName.startswith("capture7jf") and (not testDataNickName.startswith("capture7jfFree")) and (not "Distill" in testDataNickName) and (not "Envmap" in testDataNickName)
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                datasetConf["bucket"] = fyiDatasetConf.bucket
                # standard case
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageOriginal3"
                datasetConf["white_back"] = False
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["hdrScaling"] = capture7jf_hdrScalingList[caseID]
                datasetConf["singleImageForegroundZoneNumberUpperBound"] = -1
                datasetConf["singleImageBackgroundZoneNumberLowerBound"] = -1
                datasetConf["lgtLoadingMode"] = "QSPL"
                datasetConf["ifLoadDepth"] = fyiDatasetConf.get("ifLoadDepth", False)
                datasetConf["loadDepthTag"] = fyiDatasetConf.get("loadDepthTag", None)
            elif (
                testDataNickName.startswith("capture7jfFree") and (not "Envmap" in testDataNickName) and (not "Distill" in testDataNickName)
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                cut_off_right = testDataNickName.rfind("Case")
                datasetConf["bucket"] = "ol3t" + testDataNickName[int(len("capture7jf")):cut_off_right]
                # e.g. testDataNickName == "capture7jfFree3CaseX", bucket == "ol3tFree3"
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageOriginal3"
                datasetConf["white_back"] = False
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["hdrScaling"] = capture7jf_hdrScalingList[caseID]
                datasetConf["singleImageForegroundZoneNumberUpperBound"] = -1
                datasetConf["singleImageBackgroundZoneNumberLowerBound"] = -1
                datasetConf["lgtLoadingMode"] = "QSPL"  # It is olat, but there is no ambiguity here
                datasetConf["ifLoadDepth"] = fyiDatasetConf.get("ifLoadDepth", False)
                datasetConf["loadDepthTag"] = fyiDatasetConf.get("loadDepthTag", None)
            elif (
                testDataNickName.startswith("capture7jfFree") and ("Envmap" in testDataNickName) and (not testDataNickName.startswith("capture7jfFreeEnvmap"))
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                cut_off_mid = testDataNickName.find("Envmap")
                cut_off_right = testDataNickName.rfind("Case")
                datasetConf["bucket"] = (
                    "ol3t" + testDataNickName[int(len("capture7jf")):cut_off_mid] +
                    "Envmap" + testDataNickName[int(cut_off_mid + len("Envmap")):cut_off_right]
                )
                # e.g. testDataNickName == "capture7jfFree3EnvmapLgtStandardBlenderCaseX"
                # bucket == "ol3tFree3EnvmapLgtStandardBlender"
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageOriginal3"
                datasetConf["white_back"] = False
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["hdrScaling"] = capture7jf_hdrScalingList[caseID]
                datasetConf["singleImageForegroundZoneNumberUpperBound"] = -1
                datasetConf["singleImageBackgroundZoneNumberLowerBound"] = -1
                datasetConf["lgtLoadingMode"] = "QSPL"  # It is a double lgt (irrelevant olat and envmap). There is no ambiguity here.
            elif (
                testDataNickName.startswith("capture7jfDistill")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                cut_off_right = testDataNickName.rfind("Case")
                datasetConf["bucket"] = xt(testDataNickName[int(len("capture7jf")):cut_off_right])
                # e.g. testDataNickName == "capture7jfDistill512CaseX"
                # bucket == "distill512"
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageOriginal3"
                datasetConf["white_back"] = False
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["singleImageForegroundZoneNumberUpperBound"] = -1
                datasetConf["singleImageBackgroundZoneNumberLowerBound"] = -1
                datasetConf["lgtLoadingMode"] = "QSPL"  # It is pointLight. There is no ambiguity here.
                datasetConf["ifLoadDepth"] = fyiDatasetConf.get("ifLoadDepth", False)
                datasetConf["loadDepthTag"] = fyiDatasetConf.get("loadDepthTag", None)
            elif (
                testDataNickName.startswith("capture7jfEnvmap")
            ):
                datasetConf = {}
                datasetConf["dataset"] = dataset
                cut_off_right = testDataNickName.rfind("Case")
                datasetConf["bucket"] = xt(testDataNickName[int(len("capture7jf")):cut_off_right])
                # e.g. testDataNickName == "capture7jfEnvmapLavalAug2CaseX"
                # bucket == "envmapLavalAug2"
                datasetConf["caseID"] = caseID
                datasetConf["split"] = split
                datasetConf["singleImageMode"] = True
                datasetConf["class"] = "RenderingLightStageOriginal3"
                datasetConf["white_back"] = False
                datasetConf["ray_near"] = fyiDatasetConf.get("ray_near", np.nan)
                datasetConf["ray_far"] = fyiDatasetConf.get("ray_far", np.nan)
                datasetConf["minBound"] = fyiDatasetConf.minBound
                datasetConf["maxBound"] = fyiDatasetConf.maxBound
                datasetConf["sceneShiftFirst"] = fyiDatasetConf.sceneShiftFirst
                datasetConf["sceneScaleSecond"] = fyiDatasetConf.sceneScaleSecond
                datasetConf["singleImageForegroundZoneNumberUpperBound"] = -1
                datasetConf["singleImageBackgroundZoneNumberLowerBound"] = -1
                datasetConf["lgtLoadingMode"] = "ENVMAP"  # It is pointLight. There is no ambiguity here.
                datasetConf["ifPreLoadHighLight"] = False
                datasetConf["envmapHdrScaling"] = fyiDatasetConf["envmapHdrScaling"]
                datasetConf["envmapHeight"] = fyiDatasetConf["envmapHeight"]
                datasetConf["envmapWidth"] = fyiDatasetConf["envmapWidth"]
                datasetConf["envmapDataset"] = fyiDatasetConf["envmapDataset"]
            else:
                raise ValueError("Unknown testDataNickName: %s" % testDataNickName)

            meta = {}
            # You shall flag out (0) all unselected cases
            if datasetConf["class"] == "RenderingNerfBlenderDataset":
                datasetObj = RenderingNerfBlenderDataset(datasetConf, if_need_metaDataLoading=False)
            elif datasetConf["class"] == "RenderingLightStageOriginal3":
                datasetObj = RenderingLightStageOriginal3(datasetConf)
            elif datasetConf["class"] == "RenderingCropDatasetSimple":
                datasetObj = RenderingCropDatasetSimple(datasetConf)
            elif datasetConf["class"] == "RenderingLightStageLfhyper":
                datasetObj = RenderingLightStageLfhyper(datasetConf, cudaDevice=cudaDevice)
            # elif datasetConf["class"] == "RenderingLightStageHyperDemo":
            #     datasetObj = RenderingLightStageHyperDemo(datasetConf, cudaDevice=cudaDevice)
            elif datasetConf["class"] == "RenderingNerfBlenderDatasetLfhyper":
                datasetObj = RenderingNerfBlenderDatasetLfhyper(
                    datasetConf, if_need_metaDataLoading=False, cudaDevice=cudaDevice)
            elif datasetConf["class"] == "RenderingNerfBlenderDatasetDemoEnvmap":
                datasetObj = RenderingNerfBlenderDatasetDemoEnvmap(
                    datasetConf, cudaDevice=cudaDevice,
                )
            else:
                raise NotImplementedError("Unknown class: %s" % datasetConf["class"])
            if split == "test":
                tmp = np.where(datasetObj.flagSplit == 3)[0]
                interval = max(int(len(tmp)) // 16, 1)
                indChosen = tmp[::interval][:16]
                indVisChosen = indChosen
            elif split == "alltest":
                indChosen = np.where(datasetObj.flagSplit == 3)[0]
                # indVisChosen = list(np.where(datasetObj.flagSplit == 3)[0])
                indVisChosen = indChosen  # [:2]
            elif split == "val":
                tmp = np.where(datasetObj.flagSplit == 2)[0]
                interval = max(int(len(tmp)) // 16, 1)
                indChosen = tmp[::(interval + 1)][:16]
                indVisChosen = indChosen  # [:2]
            elif split == "train":
                tmp = np.where(datasetObj.flagSplit == 1)[0]
                interval = max(int(len(tmp)) // 16, 1)
                indChosen = tmp[::interval][:16]
                # indVisChosen = copy.deepcopy(indChosen)
                indVisChosen = indChosen  # [:2]
            elif split == "alltrain":
                indChosen = np.where(datasetObj.flagSplit == 1)[0]
                indVisChosen = indChosen  # [:2]
            elif split == "allval":
                indChosen = np.where(datasetObj.flagSplit == 2)[0]
                indVisChosen = indChosen  # [:2]
            elif split == "allvalid":
                indChosen = list(np.where(datasetObj.flagSplit > 0)[0])
                # indVisChosen = list(np.where(datasetObj.flagSplit > 0)[0])
                indVisChosen = indChosen  # [:2]
            elif split == "allinvalid":
                indChosen = list(np.where(datasetObj.flagSplit == 0)[0])
                indVisChosen = indChosen
            elif split == "all":
                indChosen = list(range(datasetObj.flagSplit.shape[0]))
                indVisChosen = indChosen
            else:
                raise NotImplementedError("Unknown split: %s" % split)
            entry = {
                "testDataNickName": testDataNickName,
                "meta": meta,
                "datasetObj": datasetObj,
                "indChosen": indChosen,
                "indVisChosen": indVisChosen,
            }
            testDataEntryDict[entry["testDataNickName"]] = entry
            del entry

        else:
            raise ValueError("Unknown testDataNickName: %s" % testDataNickName)

    # --------------------- More to Come --------------------- #

    return testDataEntryDict
