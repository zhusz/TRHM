import argparse
import os
import pickle
import sys
import typing
import copy
from collections import OrderedDict
import numpy as np
import cv2

import torch

projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
sys.path.append(projRoot + "src/versions/")

from configs_registration import getConfigGlobal

from codes_py.toolbox_3D.mesh_io_v1 import dump_obj_np_fileObj
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, to_heatmap
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb_np
from codes_py.toolbox_framework.framework_util_v4 import (  # noqa
    bsv02bsv,
    constructInitialBatchStepVis0,
    mergeFromBatchVis,
    splitPDDrandom,
    splitPDSRI,
)

from ..approachEntryPool import getApproachEntry

from ..testDataEntry.testDataEntryPool import getTestDataEntryDict

sys.path.append(projRoot + "src/P/")


def bt(s):
    return s[0].upper() + s[1:]


def addToSummary0Txt0BrInds(
    summary0, txt0, brInds, approachEntryDict, bsv0_forVis_dict, **kwargs
):
    input_brInds = copy.deepcopy(brInds)

    # Not Functional
    for approachNickName in approachEntryDict.keys():
        approachEntry = approachEntryDict[approachNickName]
        approachShownName = approachEntry["approachShownName"]
        bsv0_forVis = bsv0_forVis_dict[approachNickName]

        winWidth = int(bsv0_forVis.get("winWidth", 1366))
        winHeight = int(bsv0_forVis.get("winHeight", 2048))

        if "envmapOriginalNormalized" in bsv0_forVis.keys():
            summary0["%s envmapOriginalNormalized" % approachNickName] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["envmapOriginalNormalized"]), a_min=0, a_max=1)
            txt0.append("")
        if "envmapNormalized" in bsv0_forVis.keys():
            summary0["%s envmapNormalized" % approachNickName] = cv2.resize(
                np.clip(tonemap_srgb_to_rgb_np(bsv0_forVis["envmapNormalized"]), a_min=0, a_max=1),
                (512, 256),
                interpolation=cv2.INTER_NEAREST,
            )
            txt0.append("")
        if len(summary0) > 0:
            brInds.append(len(summary0))

        # Label Part (HDR - Blender exr)
        if "imghdrs" in bsv0_forVis.keys():
            summary0[
                "%s Labelled Image (Tonemapped HDR)" % approachShownName
            ] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["imghdrs"]), a_min=0, a_max=1
            )
            txt0.append("")

        # Fine Part
        if "envmapVis" in bsv0_forVis.keys():
            e = bsv0_forVis["envmapVis"]
            winWidth = tmp.shape[1]
            winHeight = tmp.shape[0]
            tmp[:e.shape[0], winWidth - e.shape[1]:, :] = e
            tmp[:e.shape[0], winWidth - e.shape[1] - 1, :] = 1
            tmp[e.shape[0], winWidth - e.shape[1]:, :] = 1
            del e, winWidth, winHeight
        if "imghdrFinePred" in bsv0_forVis.keys():
            tmp = np.clip(tonemap_srgb_to_rgb_np(bsv0_forVis["imghdrFinePred"].reshape((winHeight, winWidth, 3))), a_min=0, a_max=1)
        else:
            tmp = np.clip(tonemap_srgb_to_rgb_np((bsv0_forVis["imghdr2FinePred"] + bsv0_forVis["imghdr3FinePred"]).reshape((winHeight, winWidth, 3))), a_min=0, a_max=1)

        if "imgrawhdr2FinePred" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "rawhdr2 tonemapped (Row 1)"] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["imgrawhdr2FinePred"]).reshape((winHeight, winWidth, 3)), a_min=0, a_max=1
            )
            txt0.append("min: %.3f, max: %.3f" % (
                bsv0_forVis["imgrawhdr2FinePred"].min(),
                bsv0_forVis["imgrawhdr2FinePred"].max(),
            ))

        if "imghdr2FinePred" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "hdr2 tonemapped (Row 1)"] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["imghdr2FinePred"].reshape((winHeight, winWidth, 3))), a_min=0, a_max=1
            )
            txt0.append("min: %.3f, max: %.3f" % (
                bsv0_forVis["imghdr2FinePred"].min(),
                bsv0_forVis["imghdr2FinePred"].max(),
            ))

        summary0["%s Fine Prediction Image" % approachShownName] = tmp
        del tmp
        txt0.append("Fine PSNR: %.3f, SSIM: %.3f, LPIPS: %.3f TE: %.3f TM: %s" % (
            bsv0_forVis["finalBenPSNRFine"], bsv0_forVis["finalBenSSIMFine"], bsv0_forVis["finalBenLPIPSFine"], bsv0_forVis["timeElapsed"], bsv0_forVis["timeEvalMachine"],
        ))
        if "pixelwisePSNRFine" in bsv0_forVis.keys():
            summary0["%s Fine PSNR pixel wise" % approachShownName] = to_heatmap(
                bsv0_forVis["pixelwisePSNRFine"], cmap="inferno", vmin=10.0, vmax=50.0
            ).astype(np.float32)
            txt0.append("")
        if "depthFinePred" in bsv0_forVis.keys():
            summary0["%s Fine Prediction Depth" % approachShownName] = to_heatmap(
                bsv0_forVis["depthFinePred"]
            ).reshape((winHeight, winWidth, 3)).astype(np.float32)
            txt0.append("")
        if "opacityFinePred" in bsv0_forVis.keys():
            summary0["%s Fine Prediction Opacity" % approachShownName] = to_heatmap(
                bsv0_forVis["opacityFinePred"], cmap="inferno", vmin=0, vmax=1,
            ).reshape((winHeight, winWidth, 3)).astype(np.float32)
            txt0.append("Opacity Max: %.3f" % bsv0_forVis["opacityFinePred"].max())

        def f(ax):
            sysLabel = "world"
            ax.scatter(
                bsv0_forVis["fineVert%s" % bt(sysLabel)][:, 0],
                bsv0_forVis["fineVert%s" % bt(sysLabel)][:, 1],
                c="r",
                s=0.02,
                marker=".",
            )

        summary0["%s Fine Prediction FloorPlan" % approachShownName] = getPltDraw(f) if ("fineVertWorld" in bsv0_forVis.keys()) else np.ones((winHeight, winWidth), dtype=np.float32)
        txt0.append("")

        def g(ax):
            sysLabel = "world"
            ax.scatter(
                bsv0_forVis["fineVert%s" % bt(sysLabel)][:, 0],
                bsv0_forVis["fineVert%s" % bt(sysLabel)][:, 2],
                c="b",
                s=0.02,
                marker=".",
            )

        summary0["%s Fine Prediction FrontView" % approachShownName] = getPltDraw(g) if ("fineVertWorld" in bsv0_forVis.keys()) else np.ones((winHeight, winWidth), dtype=np.float32)
        txt0.append("")

        def h(ax):
            sysLabel = "world"
            ax.scatter(
                bsv0_forVis["fineVert%s" % bt(sysLabel)][:, 1],
                bsv0_forVis["fineVert%s" % bt(sysLabel)][:, 2],
                c="g",
                s=0.02,
                marker=".",
            )

        summary0["%s Fine Prediction SideView" % approachShownName] = getPltDraw(h) if ("fineVertWorld" in bsv0_forVis.keys()) else np.ones((winHeight, winWidth), dtype=np.float32)
        txt0.append("")

        brInds.append(len(summary0))

        # envmap
        if "envmap" in bsv0_forVis.keys():
            summary0["%s envmap" % approachShownName] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["envmap"]), a_min=0, a_max=1)
            txt0.append("")
        if "hintsRefOpacities" in bsv0_forVis.keys():
            tmp = bsv0_forVis["hintsRefOpacities"]
            tmp = tmp.reshape((winHeight, winWidth))
            summary0["%s hintsRefOpacities" % approachShownName] = np.clip(tmp, a_min=0, a_max=1)
            txt0.append("min: %f, max: %f" % (tmp.min(), tmp.max()))
        if "hintsRefSelf" in bsv0_forVis.keys():
            tmp = bsv0_forVis["hintsRefSelf"]
            tmp = tmp.reshape((winHeight, winWidth, 3))
            summary0["%s hintsRefSelf" % approachShownName] = np.clip(tonemap_srgb_to_rgb_np(tmp), a_min=0, a_max=1)
            txt0.append("min: %f, max: %f" % (tmp.min(), tmp.max()))
        if ("hintsRefLevelsColor" in bsv0_forVis.keys()):
            tmp_env = bsv0_forVis["hintsRefLevelsColor"]
            tmp_env = tmp_env.reshape((winHeight, winWidth, tmp_env.shape[-2], tmp_env.shape[-1]))
            tmp_env_l = np.clip(tonemap_srgb_to_rgb_np(tmp_env), a_min=0, a_max=1)
            tmp_opacities = bsv0_forVis["hintsRefOpacities"]
            tmp_opacities = tmp_opacities.reshape((winHeight, winWidth))
            tmp_self = bsv0_forVis["hintsRefSelf"]
            tmp_self = tmp_self.reshape((winHeight, winWidth, 3))
            tmp = tmp_env * (1 - tmp_opacities[:, :, None, None]) + tmp_self[:, :, None, :] * tmp_opacities[:, :, None, None]
            tmp_l = np.clip(tonemap_srgb_to_rgb_np(tmp), a_min=0, a_max=1)
            for l in range(tmp.shape[2]):
                summary0["%s hintsRefLevelsColor%d (envmap)" % (approachShownName, l)] = tmp_env_l[:, :, l, :]
                txt0.append("")
                summary0["%s hintsRefLevelsColor%d (integrated)" % (approachShownName, l)] = tmp_l[:, :, l, :]
                txt0.append("")
        elif ("hintsRefLevels" in bsv0_forVis.keys()):
            tmp = bsv0_forVis["hintsRefLevels"].reshape((winHeight, winWidth, bsv0_forVis["hintsRefLevels"].shape[-1]))
            for l in range(tmp.shape[2] // 1):
                rgb_l = tonemap_srgb_to_rgb_np(tmp[:, :, 1 * l:1 * (l + 1)])[:, :, 0]
                quant = 0.999
                summary0["%s hintsRefLevels%d" % (approachShownName, l)] = np.clip(rgb_l / np.quantile(rgb_l, quant), a_min=0, a_max=1)
                txt0.append("min: %f, max: %f, quantile %.3f: %f" % (rgb_l.min(), rgb_l.max(), quant, np.quantile(rgb_l, quant)))

        brInds.append(len(summary0))

        # hintsRefEnv and hintsRefDistribute
        if ("hintsRefEnv" in bsv0_forVis.keys()):
            summary0["%s hintsRefEnv" % approachShownName] = np.clip(bsv0_forVis["hintsRefEnv"].reshape((winHeight, winWidth)), a_min=0, a_max=1)
            txt0.append("min: %.3f, max: %.3f" % (bsv0_forVis["hintsRefEnv"].min(), bsv0_forVis["hintsRefEnv"].max()))
            tmp = bsv0_forVis["hintsRefDistribute"].reshape((winHeight, winWidth, bsv0_forVis["hintsRefDistribute"].shape[1]))
            for l in range(tmp.shape[2]):
                summary0["%s distribute level %d" % (approachShownName, l)] = np.clip(
                    tmp[:, :, l], a_min=0, a_max=1)
                txt0.append("min: %.3f, max: %.3f" % (
                    tmp[:, :, l].min(),
                    tmp[:, :, l].max(),
                ))

        brInds.append(len(summary0))

        # hints
        def insertHint(k):
            if k in bsv0_forVis.keys():
                v = bsv0_forVis[k].copy()

                # assert v.shape[0] == winWidth * winHeight
                if v.shape[0] != winWidth * winHeight:
                    assert v.shape[0] * v.shape[1] == winWidth * winHeight
                    assert len(v.shape) in [2, 3]
                    # No need to do anything to update v's shape (v's current shape is already good)
                else:
                    assert v.shape[0] == winWidth * winHeight
                    if len(v.shape) == 1:
                        v = v.reshape((winHeight, winWidth))
                    elif len(v.shape) == 2:
                        v = v.reshape((winHeight, winWidth, v.shape[-1]))
                    else:
                        raise ValueError("len(v.shape) is unexpected: %d" % len(v.shape))
                summary0[approachShownName + " " + k] = v
                txt0.append("min %f max %f" % (v.min(), v.max()))
            return summary0, txt0

        def insertHintHeatmap(k, cmap="inferno", vmin=0, vmax=None):
            if k in bsv0_forVis.keys():
                v = bsv0_forVis[k].copy()

                overlay = np.clip(
                    tonemap_srgb_to_rgb_np(bsv0_forVis["imghdrs"]), a_min=0, a_max=1
                ).copy()
                mask0 = v > 0.1 * v.max()
                r = overlay[:, :, 0]
                g = overlay[:, :, 1]
                b = overlay[:, :, 2]
                r[mask0] = 1
                g[mask0] = 0
                b[mask0] = 0
                overlay[:, :, 0] = r
                overlay[:, :, 1] = g
                overlay[:, :, 2] = b
                summary0[approachShownName + " " + k + "Overlay"] = overlay
                txt0.append("")

        if "hintsRefColor" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "hints_ref_color"] = bsv0_forVis["hintsRefColor"].reshape(
                (winHeight, winWidth, 3)
            )
            txt0.append("")

        if "imghdr2FinePred" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "hdr2 tonemapped"] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["imghdr2FinePred"].reshape((winHeight, winWidth, 3))), a_min=0, a_max=1
            )
            txt0.append("min: %.3f, max: %.3f" % (
                bsv0_forVis["imghdr2FinePred"].min(),
                bsv0_forVis["imghdr2FinePred"].max(),
            ))

        if "imghdr3FinePred" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "hdr3 maintaining hdr"] = np.clip(
                bsv0_forVis["imghdr3FinePred"].reshape((winHeight, winWidth, 3)), a_min=0, a_max=1
            )
            txt0.append("min: %.3f, max: %.3f" % (
                bsv0_forVis["imghdr3FinePred"].min(),
                bsv0_forVis["imghdr3FinePred"].max(),
            ))
            summary0[approachShownName + " " + "hdr3 tonemapped"] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["imghdr3FinePred"].reshape((winHeight, winWidth, 3))),
                a_min=0, a_max=1,
            )
            txt0.append("")

        # insertHint("normal2")
        if "normal2" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "normal2"] = np.abs(bsv0_forVis["normal2"].copy()).reshape((winHeight, winWidth, 3))
            txt0.append("")

        if "normal2DotH" in bsv0_forVis.keys():
            summary0["%s relu(normal2_dot_H)" % approachShownName] = np.clip(
                bsv0_forVis["normal2DotH"].reshape((winHeight, winWidth)), a_min=0, a_max=1,
            )
            txt0.append("min: %f, max: %f" % (bsv0_forVis["normal2DotH"].min(), bsv0_forVis["normal2DotH"].max()))

        if "normal3" in bsv0_forVis.keys():
            summary0[approachShownName + " " + "normal3"] = np.abs(bsv0_forVis["normal3"].reshape((winHeight, winWidth, 3)).copy())
            txt0.append("")

        if "normal3DotH" in bsv0_forVis.keys():
            summary0["%s relu(normal3_dot_H)" % approachShownName] = np.clip(
                bsv0_forVis["normal3DotH"].reshape((winHeight, winWidth)), a_min=0, a_max=1,
            )
            txt0.append("")

        insertHint("hintsPointlightOpacities")

        if "imghdrs" in bsv0_forVis.keys():
            summary0["%s Before GGX (label)" % approachShownName] = np.clip(
                tonemap_srgb_to_rgb_np(bsv0_forVis["imghdrs"]), a_min=0, a_max=1
            )
            txt0.append("")

        for i in range(4):
            insertHintHeatmap("hintsPointlightGGX%d" % i)

        brInds.append(len(summary0))

        # normalsPredVis
        for k in [
            "NdotHVisFinePred", "hdrBasicVisFinePred", "hdrHighlightVisFinePred",
            "tintVisFinePred", "intensityVisFinePred", "blinnPhongAlphaVisFinePred",
            "normalPredVisFinePred", "normalFromTauVisFinePred",
        ]:
            if k in bsv0_forVis.keys():
                summary0[k] = bsv0_forVis[k]
                txt0.append("Min Pixel %.3f, Max Pixel %.3f, Mean Pixel %.3f" % (
                    bsv0_forVis[k].min(),
                    bsv0_forVis[k].max(),
                    bsv0_forVis[k].mean(),
                ))

        if len(summary0) > brInds[-1]:
            brInds.append(len(summary0))

    return summary0, txt0, brInds


def dump(bsv0_forVis_dict, **kwargs):
    dumpDir = kwargs["dumpDir"]

    # total (labelled mesh)
    pass  # None for NeRF

    # per appraoch
    for approachNickName in bsv0_forVis_dict.keys():
        sysLabel = "world"
        bsv0_forVis = bsv0_forVis_dict[approachNickName]
        for meshName in ["coarse", "fine"]:
            if ("%sFace" in meshName):
                with open(
                    dumpDir
                    + "%s_%s_%d(%d)_%s.obj"
                    % (
                        approachNickName,
                        bsv0_forVis["dataset"],
                        bsv0_forVis["index"],
                        bsv0_forVis["flagSplit"],
                        meshName,
                    ),
                    "w",
                ) as f:
                    dump_obj_np_fileObj(
                        f,
                        bsv0_forVis["%sVert%s" % (meshName, bt(sysLabel))],
                        bsv0_forVis["%sFace" % meshName],
                    )


def main():
    testDataNickName = os.environ["DS"]
    approachNickNames = os.environ["MCs"].rstrip(",").split(",")
    cudaDevice = "cuda:0"

    # general
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
    # testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])
    Btag = os.path.realpath(__file__).split("/")[-3]
    assert Btag.startswith("B")
    Btag_root = projRoot + "v/B/%s/" % Btag
    assert os.path.isdir(Btag_root)
    # testDataEntry = testDataEntryDict[testDataNickName]

    # from testDataEntry
    testDataEntryCache = {}
    approachEntryDict = {}
    for approachNickName in approachNickNames:  # actually you only need one item in testDataEntryCache
        if approachNickName not in approachEntryDict.keys():
            approachEntryDict[approachNickName] = getApproachEntry(approachNickName)
        approachEntry = approachEntryDict[approachNickName]
        methodologyName = approachEntry["methodologyName"]
        if methodologyName.startswith("P"):
            methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = splitPDSRI(
                methodologyName
            )
            getConfigFunc = getConfigGlobal(
                methodologyP, methodologyD, methodologyS, methodologyR, wishedClassNameList="",
            )["getConfigFunc"]
            config = getConfigFunc(methodologyP, methodologyD, methodologyS, methodologyR)
            assert len(config.datasetConfDict.keys()) == 1
            datasetConf = list(config.datasetConfDict.values())[0]
            testDataEntryCache[approachNickName] = getTestDataEntryDict(
                wishedTestDataNickName=[testDataNickName],
                fyiDatasetConf=datasetConf,
                cudaDevice=cudaDevice,
            )
        else:
            # Actually you only need one item in testDataEntryCache

            # Generally this is for baselines with external_codes (e.g. iron / InverseTranslucent)
            # You need to set a default fyiDatasetConf
            raise ValueError("Unknown methodologyName: %s" % methodologyName)

    # html setups
    visualDir = projRoot + "cache/B_%s/%s_dumpHtmlForPrepick/" % (
        Btag,
        testDataNickName,
    )
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 20, testDataNickName)
    dumpDir = visualDir
    os.makedirs(dumpDir, exist_ok=True)

    testDataEntry_global = list(testDataEntryCache.values())[0][testDataNickName]
    indVisChosen_global = testDataEntry_global["indVisChosen"]
    datasetObj_global = testDataEntry_global["datasetObj"]
    datasetConf_global = datasetObj_global.datasetConf
    dataset_global = datasetConf_global["dataset"]
    for j in indVisChosen_global:
        print("Processing index %d" % j)
        bsv0_forVis_dict = {}
        for approachNickName in approachNickNames:

            if approachNickName not in approachEntryDict.keys():
                approachEntryDict[approachNickName] = getApproachEntry(approachNickName)
            approachEntry = approachEntryDict[approachNickName]
            approachShownName = approachEntry["approachShownName"]
            scriptTag = approachEntry["scriptTag"]

            outputVis_root = Btag_root + "vis/%s/%s_%s/" % (
                scriptTag,
                testDataNickName,
                approachNickName,
            )
            outputVisFileName = outputVis_root + "%08d.pkl" % j

            if not os.path.isfile(outputVisFileName):
                print("File does not exist: %s." % outputVisFileName)
                import ipdb

                ipdb.set_trace()
                raise ValueError("You cannot proceed.")

            with open(outputVisFileName, "rb") as f:
                bsv0_forVis = pickle.load(f)
            bsv0_forVis_dict[approachNickName] = bsv0_forVis

        summary0 = OrderedDict([])
        txt0 = []
        brInds = [0]
        summary0, txt0, brInds = addToSummary0Txt0BrInds(
            summary0,
            txt0,
            brInds,
            approachEntryDict,
            bsv0_forVis_dict,
            visualDir=visualDir,
            testDataNickName=testDataNickName,
            dataset=dataset_global,
        )

        headerMessage = (
            "testDataNickName: %s, Dataset: %s, Index: %s, flagSplit: %d"
            % (
                testDataNickName,
                dataset_global,
                bsv0_forVis["index"],
                bsv0_forVis["flagSplit"],
            )
        )
        if "groupViewID" in bsv0_forVis.keys():
            headerMessage += " groupViewID: %d, groupID: %d, viewID: %d, ccID: %d, olatID: %d" % (
                bsv0_forVis["groupViewID"],
                bsv0_forVis["groupID"],
                bsv0_forVis["viewID"],
                bsv0_forVis.get("ccID", -1),
                bsv0_forVis.get("olatID", -1),
            )
        subMessage = ""
        htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
        if j == indVisChosen_global[0]:
            dump(bsv0_forVis_dict, dumpDir=dumpDir)
            # only dump the first one (the others are the same for the same scene)


if __name__ == "__main__":
    main()
