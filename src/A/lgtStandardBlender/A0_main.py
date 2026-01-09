import os
import pickle
import typing
import numpy as np
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
    saveRoot = projRoot + "v/A/%s/" % dataset

    m = 8
    lgtNameList = [
        "city",
        "courtyard",
        "forest",
        "interior",
        "night",
        "studio",
        "sunrise",
        "sunset",
    ]
    lgtNameDict = {lgtNameList[j]: j for j in range(m)}
    lgtNameRetrieveList = {j: lgtNameList[j] for j in range(m)}
    nameList = [
        "v/fast/%s/%s.exr" % (dataset, lgtNameList[j]) for j in range(m)
    ]
    lgtType = "envmap"
    winWidth = np.zeros((m,), dtype=np.int32)
    winHeight = np.zeros((m,), dtype=np.int32)

    for j in range(m):
        assert os.path.isfile(projRoot + nameList[j]), projRoot + nameList[j]
        tmp = cv2.imread(projRoot + nameList[j], flags=cv2.IMREAD_UNCHANGED)
        assert len(tmp.shape) == 3
        assert tmp.shape[2] == 3
        winWidth[j] = tmp.shape[1]
        winHeight[j] = tmp.shape[0]

    with open(saveRoot + "A0_main.pkl", "wb") as f:
        pickle.dump(
            {
                "m": m,
                "dataset": dataset,
                "lgtNameList": lgtNameList,
                "lgtNameDict": lgtNameDict,
                "lgtNameRetrieveList": lgtNameRetrieveList,
                "nameList": nameList,
                "lgtType": lgtType,
                "winWidth": winWidth,
                "winHeight": winHeight,
            },
            f,
        )


if __name__ == "__main__":
    main()
