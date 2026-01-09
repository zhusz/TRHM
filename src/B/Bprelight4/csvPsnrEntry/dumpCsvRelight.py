import os
import sys
from collections import OrderedDict
import pickle
import numpy as np
import torch

projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
sys.path.append(projRoot + "src/versions/")
sys.path.append(projRoot + "src/P/")
from codes_py.py_ext.misc_v1 import tabularPrintingConstructing, tabularCvsDumping
from ..approachEntryPool import getApproachEntry
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from configs_registration import getConfigGlobal
from codes_py.toolbox_framework.framework_util_v4 import (  # noqa
    bsv02bsv,
    constructInitialBatchStepVis0,
    mergeFromBatchVis,
    splitPDDrandom,
    splitPDSRI,
)

governedScriptTagList = ["1E", "1Eldr", "1G3", "1G4plain", "1G5plain", "1Ef", "1Eldrf", "1Eldrenvmapf"]


def bt(s):
    return s[0].upper() + s[1:]


def assertBenchmarkings(bsv0_toStore, bsv0_loaded, **kwargs):
    pass


def get_s(bsv0_retrieveList_dict, approachEntryDict, testDataEntry, **kwargs):
    # from testDataEntry
    indChosen = testDataEntry["indChosen"]

    s = OrderedDict([])
    for approachNickName in bsv0_retrieveList_dict.keys():
        approachShownName = approachEntryDict[approachNickName]["approachShownName"]
        s[approachNickName] = OrderedDict([])
        s[approachNickName]["method"] = approachShownName

        record = {}

        benPool = ["PSNRCoarse", "SSIMCoarse", "LPIPSCoarse", "PSNRFine", "SSIMFine", "LPIPSFine", "PSNRHighSubLightHDRFine", "PSNRHighLightHDRFine", "PSNRHighSubLightFine", "PSNRHighLightFine"]

        for k in benPool:
            record[k] = []

        for j in indChosen:
            bsv0 = bsv0_retrieveList_dict[approachNickName][j]
            for k in benPool:
                if "finalBen%s" % bt(k) in bsv0.keys():
                    record[k].append(float(bsv0["finalBen%s" % bt(k)]))

        for k in benPool:
            if len(record[k]) == 0:
                s[approachNickName][k] = "-"
                continue
            t = np.array(record[k], dtype=np.float32)

            if "Light" in k:
                pass
            else:
                assert np.isnan(t).sum() == 0

            t = np.nanmean(t)
            if k.startswith("LPIPS"):
                s[approachNickName][k] = "%.3f" % t
            elif k.startswith("PSNR"):
                s[approachNickName][k] = "%.1f" % t
            elif k.startswith("SSIM"):
                s[approachNickName][k] = "%.1f" % (t * 100)
            else:
                raise NotImplementedError("Unknown benPool k: %s" % k)
        s[approachNickName]["ValidCount"] = np.isfinite(record["PSNRFine"]).sum()
    return s


def readBenchmarkings(testDataNickName, approachNickNames, Btag_root, approachNickNamesTemplate):
    # get fyiDatasetConf
    approachEntry = getApproachEntry(approachNickNames[0])
    approachNickName, methodologyName, scriptTag = (
        approachEntry["approachNickName"],
        approachEntry["methodologyName"],
        approachEntry["scriptTag"],
    )
    methodologyP, methodologyD, methodologyS, methodologyR, methodologyI = splitPDSRI(
        methodologyName
    )
    getConfigFunc = getConfigGlobal(
        methodologyP,
        methodologyD,
        methodologyS,
        methodologyR,
        wishedClassNameList=[],
    )["getConfigFunc"]
    config = getConfigFunc(methodologyP, methodologyD, methodologyS, methodologyR)
    assert len(config.datasetConfDict) == 1
    fyiDatasetConf = list(config.datasetConfDict.values())[0]
    del approachEntry, approachNickName, methodologyName, scriptTag
    del methodologyP, methodologyD, methodologyS, methodologyR, methodologyI, getConfigFunc, config

    # testDataEntry
    testDataEntry = getTestDataEntryDict(
        wishedTestDataNickName=[testDataNickName],
        fyiDatasetConf=fyiDatasetConf,
    )[testDataNickName]
    datasetObj = testDataEntry["datasetObj"]
    dataset = datasetObj.datasetConf["dataset"]
    indChosen = testDataEntry["indChosen"]
    flagSplit = datasetObj.flagSplit

    # run
    bsv0_retrieveList_dict = OrderedDict([])
    approachEntryDict = {}
    for (approachNickName, approachNickNameTemplate) in zip(approachNickNames, approachNickNamesTemplate):
        approachEntry = getApproachEntry(approachNickName)
        approachEntryDict[approachNickNameTemplate] = approachEntry

        approachNickName, methodologyName, scriptTag, approachShownName = \
            approachEntry['approachNickName'], approachEntry['methodologyName'], \
            approachEntry['scriptTag'], approachEntry['approachShownName']

        if scriptTag in governedScriptTagList:
            print("Currently Gathering Approach %s" % approachNickName)

            # assertions and mkdir
            outputBen_root = Btag_root + 'ben/%s/%s_%s/' % \
                             (scriptTag, testDataNickName, approachNickName)
            assert os.path.isdir(outputBen_root), outputBen_root

            # record
            bsv0_retrieveList_dict[approachNickNameTemplate] = OrderedDict([])
            count = 0
            for j in indChosen:
                outputBenFileName = outputBen_root + '%08d.pkl' % j
                if not os.path.isfile(outputBenFileName):
                    print('Problem occurred. File not exist for %s. Please check.' % outputBenFileName)
                    import ipdb
                    ipdb.set_trace()
                    raise ValueError('You cannot proceed.')
                else:
                    with open(outputBenFileName, 'rb') as f:
                        bsv0 = pickle.load(f)
                    bsv0_retrieveList_dict[approachNickNameTemplate][j] = bsv0
                count += 1

    all_s = get_s(
        bsv0_retrieveList_dict,
        approachEntryDict,
        testDataEntry,
    )
    return all_s


def main():  # load from "ben"
    # get from the env var
    testDataNickName = os.environ["DS"]
    approachNickNames = os.environ["MCs"].rstrip(",").split(",")

    # general
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    s = readBenchmarkings(testDataNickName, approachNickNames, Btag_root)
    
    print(tabularPrintingConstructing(
        s,
        field_names=list(list(s.values())[0].keys()),
        ifNeedToAddTs1=False,
    ))
    os.makedirs(projRoot + 'cache/B_%s/' % Btag, exist_ok=True)
    csv_fn = projRoot + 'cache/B_%s/benchmarking_%s_%s.csv' % \
             (Btag, Btag, testDataNickName)
    tabularCvsDumping(
        fn=csv_fn,
        s=s,
        fieldnames=list(list(s.values())[0].keys()),
        ifNeedToAddTs1=False,
    )


# BNs=PSNRCoarse,PSNRFine, SSIMFine,ValidCount Cs=0,1,2,3,4,5,6,7 DS=renderingNerfBlenderExistingCase*SplitAlltest MCs=PrefnerfmlpBerkeleyD0largeSgExistingwbs8scalebetterR*I480960On1Ef OPENCV_IO_ENABLE_OPENEXR=true CUDA_VISIBLE_DEVICES=0 python -m Bprelight4.csvPsnrEntry.dumpCsvRelight
def mains():  # put multiple cases together
    caseIDs = [int(x) for x in os.environ["Cs"].strip(",").rstrip(",").split(",")]
    testDataNickNameTemplate = os.environ["DS"]
    assert testDataNickNameTemplate.find("*") == testDataNickNameTemplate.rfind("*") >= 0
    approachNickNamesTemplate = os.environ["MCs"].strip(",").rstrip(",").split(",")
    for approachNickNameTemplate in approachNickNamesTemplate:
        assert approachNickNameTemplate.find("*") == approachNickNameTemplate.rfind("*") >= 0
    benchmarkingNames = os.environ["BNs"].strip(",").rstrip(",").split(",")
    
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)

    record_s = OrderedDict([])  # major/middle/minor: caseID/approaches/benchmarkingNames
    for caseID in caseIDs:
        print("mains reading caseID %d" % caseID)

        testDataNickName = testDataNickNameTemplate.replace("*", "%d" % caseID)
        approachNickNames = [x.replace("*", "%d" % caseID) for x in approachNickNamesTemplate]
        
        s = readBenchmarkings(testDataNickName, approachNickNames, Btag_root, approachNickNamesTemplate)
        record_s[caseID] = s
        del s

    # this is a 3-way table (caseIDs, aproaches, benchmarkingNames)
    # You got to visualize it in a 2D table
    # ==================== Organize into the form you like ==================== #
    out_s = OrderedDict([])
    for approachNickNameTemplate in approachNickNamesTemplate:
        for caseID in caseIDs:
            out_s["Approach_%s_caseID_%d" % (approachNickNameTemplate, caseID)] = record_s[caseID][approachNickNameTemplate]
    # ========================================================================= #

    print(tabularPrintingConstructing(
        out_s,
        field_names=list(list(out_s.values())[0].keys()),
        ifNeedToAddTs1=False,
    ))
    os.makedirs(projRoot + 'cache/B_%s/' % Btag, exist_ok=True)
    csv_fn = projRoot + 'cache/B_%s/benchmarking_%s_%s.csv' % \
             (Btag, Btag, testDataNickName)
    tabularCvsDumping(
        fn=csv_fn,
        s=out_s,
        fieldnames=list(list(out_s.values())[0].keys()),
        ifNeedToAddTs1=False,
    )


if __name__ == '__main__':
    mains()
