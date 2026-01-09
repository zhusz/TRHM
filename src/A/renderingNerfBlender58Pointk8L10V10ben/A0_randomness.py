
# A/renderingNerfBlender58Pointk8L10V10ben
# 58: the camera now fully mimics the original nerf_synthetic, rather than the light stage cameras
# k8: resulution to be 400 rather than the default 800.
# L10: 10 lightings in total - for benchmarking purpose
# V10: 10 test views - 0, 20, .., 180 of the test views

# 32 cases: sceneNerfBlender5Complete-(0~7) and sceneNerfBlender6Diffuse-(0~7) and sceneNerfBlender7SSS-(0~7) and sceneNerfBlender8sssWSpecularity-(0~7)

# m = 32 (32 scenes) * 100 (10 test lightings and 10 test views)
#   = 3200
# only render case 0-7: [0, 800) 

import json
import math
import os
import pickle
import sys
import typing

import numpy as np


def getNeureconDepthFileListExisting(winSize):  # So for your own datasetConf, you should index from this generated existing depthFileList
    # Make sure you are working with the nerf_synthetic existing styled scenario (400x8)
    depthFileList = []
    for caseID in range(8):
        for split, rTot in [("train", 100), ("val", 100), ("test", 200)]:
            for caseSplitInsideIndex in range(rTot):
                depthFileList.append(
                    "v/misc/nerf_synthetic_original_cameras_surfacenerf_results/nerf_synthetic_original_cameras_neureconwm_results/scene_%d/%s/r_%d_winSize_%d.pkl" % (
                        caseID, split, caseSplitInsideIndex, winSize
                    ))
    return depthFileList


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
    saveRoot = projRoot + "v/A/%s/" % dataset

    A0 = {}
    for k in [
        "sceneDatasetList",
        "sceneIDList",
        # "lgtDatasetList",  # Just write "None"
        # "lgtIDList",  # Just write -1
        # "lgtStrengthList", always 1, will not be recording
        # "mimickingSceneDatasetList", We do not do mimicking here, as mimicking only specify lights, not cameras
        # "mimickingSceneIDList",
        # "mimickingGroupIDList",
        # "mimickingOlatIDList",
        # "groupIDList",
        # "viewIDList",
        "lgtE",  # We are doing SH lighting, not point lights
        "lgtEnergy",
        "objCentroid",  # We keep it, but it is not very useful, as we do uncenetered SHL0 rendering
        "E",
        "L",
        "U",
        "winWidth",
        "winHeight",
        "focalLengthWidth",
        "focalLengthHeight",
        "fovRad",
        "flagSplit",
        "caseIDList",
    ]:
        A0[k] = []

    with open(projRoot + "v/A/renderingNerfBlenderExisting/A0_randomness.pkl", "rb") as f:
        A0_existing = pickle.load(f)

    # set your set
    tupScene = [  # (sceneDataset, sceneID)
        [(sceneDataset, x) for x in range(8)]
        for sceneDataset in ["sceneNerfBlender5Complete", "sceneNerfBlender6Diffuse", "sceneNerfBlender7SSS", "sceneNerfBlender8sssWSpecularity"]
    ]
    tupScene = tupScene[0] + tupScene[1] + tupScene[2] + tupScene[3]
    winSize = 800.0
    setLgtRadius = 100.0
    setLgtEnergy = 2.e5

    lgtDataset = "lgtEnvolatGray16x32"  # Do remember this is just for you to get the omega
    with open(projRoot + "v/A/%s/A0_main.pkl" % lgtDataset, "rb") as f:
        omega_lgt = pickle.load(f)["omega"]

    caseCount = 0
    testLgtIDs = np.array(list(range(33, 256, 20))[:10], dtype=np.int32)
    nTestLgt = 10
    assert testLgtIDs.shape == (nTestLgt,)
    testViewIDs = np.arange(0, 200, 20).astype(np.int32) + 200  # You still need to add the 400 * sceneID inside the loop
    nTestView = 10
    assert testViewIDs.shape == (nTestView,)
    nTest = nTestLgt * nTestView
    testLgtIDs_, testViewIDs_ = np.meshgrid(testLgtIDs, testViewIDs)
    testLgtIDs_ = testLgtIDs_.reshape(-1)
    testViewIDs_ = testViewIDs_.reshape(-1)
    
    for (sceneDataset, sceneID) in tupScene:
        print("Working on A0_randomness for %s: (%s-%d)" % (dataset, sceneDataset, sceneID))
        A0["sceneDatasetList"] += [sceneDataset] * nTest
        A0["sceneIDList"] += [sceneID * np.ones((nTest,), dtype=np.int32)]
        for k in ["E", "L", "U", "fovRad", "flagSplit"]:
            A0[k] += [A0_existing[k][sceneID * 400 + testViewIDs_]]
        for k in ["winWidth", "winHeight", "focalLengthWidth", "focalLengthHeight"]:
            A0[k] += [
                A0_existing[k][sceneID * 400 + testViewIDs_] / 800.0 * float(winSize)
            ]
        A0["objCentroid"] += [np.zeros((nTest, 3), dtype=np.float32)]
        # A0["lgtE"] += [np.tile(-omega_lgt[lgtID, :][None, :] * setLgtRadius, (nTest, 1))]
        A0["lgtE"] += [-omega_lgt[testLgtIDs_, :] * setLgtRadius]
        A0["lgtEnergy"] += [setLgtEnergy * np.ones((nTest,), dtype=np.float32)]
        A0["caseIDList"] += [caseCount * np.ones((nTest,), dtype=np.int32)]

        caseCount += 1

    m = 3200
    for k in [
        "sceneIDList",
        "lgtE",
        "lgtEnergy",
        "objCentroid",
        "E",
        "L",
        "U",
        "winWidth",
        "winHeight",
        "focalLengthWidth",
        "focalLengthHeight",
        "fovRad",
        "flagSplit",
        "caseIDList",
    ]:
        A0[k] = np.concatenate(A0[k], 0)
    for k in A0.keys():
        assert len(A0[k]) == m, (k, len(A0[k]))
    A0["m"] = m
    caseTot = caseCount
    A0["caseTot"] = caseTot

    # caseIDList + flagSplit --> caseSplitInsideIndexList (used for put down r_%d)
    A0["caseSplitInsideIndexList"] = -np.ones((m,), dtype=np.int32)
    for caseID in range(caseTot):
        for splitID in [1, 2, 3]:
            ind = np.where((A0["caseIDList"] == caseID) & (A0["flagSplit"] == splitID))[
                0
            ]
            A0["caseSplitInsideIndexList"][ind] = np.arange(len(ind))
    assert np.all(A0["caseSplitInsideIndexList"] >= 0)
    assert np.all(A0["flagSplit"] >= 0)
    assert np.all(A0["caseIDList"] >= 0)
    # check (caseID, splitID, caseSplitInsideIndexList) typle should be unique for all
    checkIndex = (
        A0["caseIDList"]
        + caseTot * A0["flagSplit"]
        + caseTot
        * 4
        * A0["caseSplitInsideIndexList"]  # 4: account in flagSplit == [0, 1, 2, 3]
    )
    assert len(np.unique(checkIndex)) == len(checkIndex)
    # Then nameList and rootList
    A0["nameList"] = [
        "v/R/%s/R2/%sCase%d/%s/r_%d.png"
        % (
            dataset,
            dataset,
            A0["caseIDList"][j],
            ["invalid", "train", "val", "test"][A0["flagSplit"][j]],
            A0["caseSplitInsideIndexList"][j],
        )
        for j in range(m)
    ]
    A0["nameListExr"] = [
        "v/R/%s/R2exr/%sCase%d/%s/r_%d.exr"
        % (
            dataset,
            dataset,
            A0["caseIDList"][j],
            ["train", "val", "test"][A0["flagSplit"][j] - 1],
            A0["caseSplitInsideIndexList"][j],
        )
        for j in range(m)
    ]
    A0["nameListExrDepth"] = [
        "v/R/%s/R2exrDepth/%sCase%d/%s/r_%d.exr"
        % (
            dataset,
            dataset,
            A0["caseIDList"][j],
            ["train", "val", "test"][A0["flagSplit"][j] - 1],
            A0["caseSplitInsideIndexList"][j],
        )
        for j in range(m)
    ]
    A0["nameListExrNormal"] = [
        "v/R/%s/R2exrNormal/%sCase%d/%s/r_%d.exr"
        % (
            dataset,
            dataset,
            A0["caseIDList"][j],
            ["train", "val", "test"][A0["flagSplit"][j] - 1],
            A0["caseSplitInsideIndexList"][j],
        )
        for j in range(m)
    ]
    mappingIndex = []
    for repitition in range(4):  # 32 scenes
        for caseID in range(8):
            mappingIndex.append(400 * caseID + testViewIDs_)
    mappingIndex = np.concatenate(mappingIndex, 0)
    nameListNeureconwmDepthExisting = getNeureconDepthFileListExisting(800)
    nameListNeureconwmDepth = [nameListNeureconwmDepthExisting[int(j)] for j in mappingIndex.tolist()]
    A0["nameListNeureconwmDepth"] = nameListNeureconwmDepth

    with open(saveRoot + "A0_randomness.pkl", "wb") as f:
        pickle.dump(A0, f)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == "__main__":
    main()
