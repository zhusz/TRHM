import os
import pickle
import numpy as np
import cv2
from collections import OrderedDict

from codes_py.py_ext.misc_v1 import tabularPrintingConstructing

from codes_py.toolbox_framework.framework_util_v4 import (  # noqa
    bsv02bsv,
    constructInitialBatchStepVis0,
    mergeFromBatchVis,
    splitPDDrandom,
    splitPDSRI,
)

from configs_registration import getConfigGlobal

from .benchmarkingPsnrLdrRenderingNerfBlender import (  # noqa
    benchmarkingPsnrLdrRenderingNerfBlenderFunc,
)
from .dumpCsvRelight import assertBenchmarkings, get_s
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict


def testApproachOnTestData(testDataNickName, approachEntry, **kwargs):
    cudaDevice = "cuda:0"  # CUDA_VISIBLE_DEVICES env var already imposed

    # from testDataEntry
    # datasetObj = testDataEntry["datasetObj"]
    # indChosen = testDataEntry["indChosen"]
    # indVisChosen = testDataEntry["indVisChosen"]
    # testDataNickName = testDataEntry["testDataNickName"]
    # flagSplit = datasetObj.flagSplit
    # datasetMeta = testDataEntry["meta"]

    # from approachEntry
    approachNickName, methodologyName, scriptTag = (
        approachEntry["approachNickName"],
        approachEntry["methodologyName"],
        approachEntry["scriptTag"],
    )
    if methodologyName.startswith("P"):
        methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = splitPDSRI(
            methodologyName
        )
    else:
        methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = None, None, None, None, None

    # assertions and mkdir
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
    Btag = os.path.realpath(__file__).split("/")[-3]
    assert Btag.startswith("B")
    Btag_root = projRoot + "v/B/%s/" % Btag
    assert os.path.isdir(Btag_root), Btag_root
    outputBen_root = Btag_root + "ben/%s/%s_%s/" % (
        scriptTag,
        testDataNickName,
        approachNickName,
    )
    os.makedirs(outputBen_root, exist_ok=True)
    outputVis_root = Btag_root + "vis/%s/%s_%s/" % (
        scriptTag,
        testDataNickName,
        approachNickName,
    )
    os.makedirs(outputVis_root, exist_ok=True)

    # trainer
    if methodologyP is not None:
        DTrainer = getConfigGlobal(
            methodologyP,
            methodologyD,
            methodologyS,
            methodologyR,
            wishedClassNameList=["DTrainer"],
        )["exportedClasses"]["DTrainer"]
        getConfigFunc = getConfigGlobal(
            methodologyP,
            methodologyD,
            methodologyS,
            methodologyR,
            wishedClassNameList=[],
        )["getConfigFunc"]
        config = getConfigFunc(methodologyP, methodologyD, methodologyS, methodologyR)
        trainer = DTrainer(
            config,
            rank=0,
            numMpProcess=0,
            ifDumpLogging=False,
        )
        trainer.initializeAll(
            iter_label=int(methodologyI[1:]), hook_type=None, if_need_metaDataLoading=False
        )
        trainer.setModelsEvalMode(trainer.models)
    else:
        trainer = None

    # testDataEntry
    testDataEntryDict = getTestDataEntryDict(
        wishedTestDataNickName=[testDataNickName],
        fyiDatasetConf=list(trainer.config.datasetConfDict.values())[0],
        cudaDevice=cudaDevice,
    )
    testDataEntry = testDataEntryDict[testDataNickName]
    datasetObj = testDataEntry["datasetObj"]
    flagSplit = datasetObj.flagSplit
    indChosen = testDataEntry["indChosen"]
    indVisChosen = testDataEntry["indVisChosen"]

    # record
    bsv0_retrieveList_dict = OrderedDict([])
    bsv0_retrieveList_dict[approachNickName] = OrderedDict([])

    j1 = os.environ.get("J1", "-1")
    j2 = os.environ.get("J2", "-1")
    if j1 == "":
        j1 = -1
    else:
        j1 = int(j1)
    if j2 == "":
        j2 = -1
    else:
        j2 = int(j2)

    # counter
    if (j1 >= 0) and (j2 >= 0):
        count = j1
    else:
        count = 0

    p1 = j1
    p2 = j2

    # interesting variable naming convention...
    # incChosen refers to the whole test set's ids
    # i refers to j1 / j2 / p1 / p2, which i-th item in the whole indChosen
    # j refers to the true index in the indChosen (the whole training set)...
    for i, j in enumerate(indChosen):
        if (j1 >= 0) and (j2 >= 0) and ((i < p1) or (i >= p2)):
            continue

        if ("Distill512" in datasetObj.datasetConf["dataset"]):
            """
            # outputBenFileName = "/shared/zhusz/data2/distill/%s_%s/hdr2_%08d.exr" % (
            # outputBenFileName = "/scratch/one_month/2024_02/zhusz/data3/remote_syncdata_2/sa/v/misc/distill/%s_%s/hdr2_%08d.exr" % (
            outputBenFileName = projRoot + "v/misc/distill/%s_%s/hdr2_%08d.exr" % (
                testDataNickName, approachNickName, j
            )
            # outputVisFileName = outputBenFileName
            outputVisFileName = projRoot + "v/misc/distill/%s_%s/bsv0_%08d.pkl" % (
                testDataNickName, approachNickName, j
            )
            """
            # No special treatment now
            outputBenFileName = outputBen_root + "%08d.pkl" % j
            outputVisFileName = outputVis_root + "%08d.pkl" % j
        elif ("Lfdistill" in datasetObj.datasetConf["dataset"]):
            outputBenFileName = projRoot + "v/misc/lfdistill/%s_%s/hdr23_%08d.pkl" % (
                testDataNickName, approachNickName, j
            )
            outputVisFileName = outputBenFileName
            # outputVisFileName = projRoot + "v/misc/lfdistill/%s_%s/bsv0_%08d.pkl" % (
            #     testDataNickName, approachNickName, j
            # )
        else:
            outputBenFileName = outputBen_root + "%08d.pkl" % j
            outputVisFileName = outputVis_root + "%08d.pkl" % j

        if os.path.isfile(outputBenFileName) and (
            (j not in indVisChosen) or (os.path.isfile(outputVisFileName))
        ):
            print(
                "Reading %s (flagSplit %d) - count (%d / %d)"
                % (outputBenFileName, flagSplit[j], count, len(indChosen))
            )

            if ("Distill512" in datasetObj.datasetConf["dataset"]) or ("Lfdistill" in datasetObj.datasetConf["dataset"]):
                count += 1
                continue
            else:
                with open(outputBenFileName, "rb") as f:
                    bsv0_toStore = pickle.load(f)
        else:
            print(
                "Processing %s (flagSplit %d) - count (%d / %d)"
                % (outputBenFileName, flagSplit[j], count, len(indChosen))
            )

            batch0_np = datasetObj.getOneNP(j)
            batch_np = bsv02bsv(batch0_np)
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis,
                iterCount=trainer.resumeIter,
                visIndex=0,
                dataset=None,  # only during training you need to input the dataset here.
                P=methodologyP,
                D=methodologyD,
                S=methodologyS,
                R=methodologyR,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromBatchVis(
                bsv0_initial, batch_vis, dataset=None, visIndex=0
            )

            # step
            if testDataNickName.startswith(
                "renderingNerfBlender"
            ) or testDataNickName.startswith(
                "renderingCapture"
            ) or testDataNickName.startswith(
                "capture7jf"
            ) or testDataNickName.startswith(
                "renderingFlagship"
            ):
                bsv0 = benchmarkingPsnrLdrRenderingNerfBlenderFunc(
                    bsv0_initial,
                    # misc
                    cudaDevice=cudaDevice,
                    datasetObj=datasetObj,
                    callFlag="benchmarking",
                    logDir=trainer.logDir,
                    # benchmarking rules
                    # ifRequiresBenchmarking=not("Distill512" in datasetObj.datasetConf["dataset"]) and not("Lfdistill" in datasetObj.datasetConf["dataset"]),
                    ifRequiresBenchmarking=not("Lfdistill" in datasetObj.datasetConf["dataset"]),
                    ifFit=False,
                    ssimWindow=11,
                    orthogonalDensityResolution=128,  # not affecting PSNR
                    marching_cube_thre=0.5,  # not affecting PSNR
                    # drawing
                    ifRequiresDrawing=(j in indVisChosen),
                    ifRequiresMesh=False,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=trainer.models,
                    doPred0Func=trainer.doPred0,
                    config=trainer.config,
                    iterCount=trainer.resumeIter,
                    lgtMode="pointLightMode",
                    if_only_predict_hdr2=False,  # ("Distill512" in datasetObj.datasetConf["dataset"]),
                )
            else:
                raise NotImplementedError(
                    "Unknown testDataNickName: %s" % testDataNickName
                )

            # special operation for distillation (training set make up) - do not dump anything, but just store the predicted hdr2 values
            # (the test dataset loader should only directs to the foreground masked rays)
            # Prediction should not go through the geometry prediction part
            if ("Distill512" in datasetObj.datasetConf["dataset"]):
                pass  # no special treatment now

            elif ("Lfdistill" in datasetObj.datasetConf["dataset"]):
                valid_mask_original = bsv0["valid_mask_original"]
                hdr2_masked = bsv0["imghdr2FinePred"]
                hdr3_masked = bsv0["imghdr3FinePred"]
                lgtE = bsv0["lgtE"]
                assert np.all(np.min(lgtE, 0) == np.max(lgtE, 0))
                lgtE0 = lgtE[0, :]
                os.makedirs(os.path.dirname(outputVisFileName), exist_ok=True)
                assert outputVisFileName.endswith(".pkl")
                with open(outputVisFileName, "wb") as f:
                    pickle.dump({
                        "valid_mask_original": valid_mask_original,
                        "hdr2_masked": hdr2_masked,
                        "hdr3_masked": hdr3_masked,
                        "lgtE0": lgtE0,
                    }, f)
                count += 1
                continue

            # outputBenFile
            bsv0_toStore = {
                k: bsv0[k]
                for k in bsv0.keys()
                if (
                    k
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
                        # specific
                        # "imghdrFinePred",
                        # "envmapVis",
                    ]
                    or ("finalBen" in k)
                )
            }

            if os.path.isfile(outputBenFileName):
                pass
                """
                with unio.open(outputBenFileName, uconf, "rb") as f:
                    bsv0_loaded = pickle.load(f)
                if testDataNickName.startswith("nerfBlender8"):
                    assertBenchmarkings(
                        bsv0_toStore,
                        bsv0_loaded,
                    )
                else:
                    raise NotImplementedError(
                        "Unknown testDataNickName: %s" % testDataNickName
                    )
                """
            else:
                with open(outputBenFileName, "wb") as f:
                    pickle.dump(bsv0_toStore, f)

            # outputVisFile
            if (j in indVisChosen) and (not os.path.isfile(outputVisFileName)):
                bsv0_forVis = {
                    k: bsv0[k]
                    for k in bsv0.keys()
                    if (
                        k
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
                            "flagSplit",  # all the above shall be kept
                            "winWidth",
                            "winHeight",
                            "imghdrs",  # 800x800x3, label, created (you need to know the width and height rather than width * height)
                            "imghdrFinePred",  # 800x800x3, pred, keep
                            "opacityFinePred",  # 800x800, pred, keep
                            "depthFinePred",  # 800x800, pred, keep
                            "fineVertWorld",
                            "fineFace",

                            "pixelID_highlight",
                            "pixelID_sublight",

                            # hints
                            "imghdr2FinePred",
                            "imghdr3FinePred",

                            # "normal3",
                            "normal3DotH",
                            "normal2",
                            "normal2DotH",
                            "hintsPointlightOpacities",
                            "hintsPointlightGGX0",
                            "hintsPointlightGGX1",
                            "hintsPointlightGGX2",
                            "hintsPointlightGGX3",
                            "hintsRefOpacities",
                            "hintsRefSelf",
                            "hintsRefLevels",

                            # All the IDs
                            "groupViewID",
                            "groupID",
                            "viewID",
                            "ccID",
                            "olatID",

                            # "envmap0",

                            # benchmarkings / figuring purpose
                            # "pixelwisePSNRFine",  # 800x800, pred, keep
                            # "envmapVis",  # 192x384x3, label (visualization purpose), keep

                            # distill debug
                            # "valid_mask_original",

                            # speed
                            "timeElapsed",
                            "timeEvalMachine",
                        ]
                    )
                    or k.startswith("finalBen")
                }
                with open(outputVisFileName, "wb") as f:
                    pickle.dump(bsv0_forVis, f)

        bsv0_retrieveList_dict[approachNickName][j] = bsv0_toStore
        """ no special treatment for distil512
        if not("Distill512" in datasetObj.datasetConf["dataset"]):
            print(
                "    Benchmarking Representative: finalBenPSNRFine %.2f"
                % bsv0_toStore.get(
                    "finalBenPSNRFine", "Key '%s' Not Available" % "finalBenPSNRFine"
                )
            )
        """

        count += 1

    if (j1 < 0) and (j2 < 0):
        s = get_s(
            bsv0_retrieveList_dict,
            {approachEntry["approachNickName"]: approachEntry},
            testDataEntry,
        )
        print(
            tabularPrintingConstructing(
                s,
                field_names=list(list(s.values())[0].keys()),
                ifNeedToAddTs1=False,
            )
        )