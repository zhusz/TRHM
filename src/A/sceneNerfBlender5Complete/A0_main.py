# sceneNerfBlender5Compelte-A0
#   Complete version of the nerf_synthetic blender file datasets (made up the missing texture files)
#   should be able to reproduce all the rendering as released in the nerf dataset

import os
import pickle
import typing


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
    saveRoot = projRoot + "v/A/%s/" % dataset

    m = 8
    sceneNameList = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]
    sceneNameDict = {sceneNameList[j]: j for j in range(m)}
    scneeNameRetrieveList = {j: sceneNameList[j] for j in range(m)}
    nameList = [
        "v/fast/blend_files_complete/%s.blend" % sceneName
        for sceneName in sceneNameList
    ]
    for j in range(m):
        assert os.path.isfile(projRoot + nameList[j])

    with open(saveRoot + "A0_main.pkl", "wb") as f:
        pickle.dump(
            {
                "m": m,
                "dataset": dataset,
                "sceneNameList": sceneNameList,
                "sceneNameDict": sceneNameDict,
                "sceneNameRetrieveList": scneeNameRetrieveList,
                "nameList": nameList,
            },
            f,
        )


if __name__ == "__main__":
    main()
