import os
import cv2
import pickle
import numpy as np
projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    dataset_original = "lgtStandardBlender"
    assert dataset == dataset_original + "1k"
    A0_fn_original = projRoot + "v/A/%s/A0_main.pkl" % dataset_original
    assert os.path.isfile(A0_fn_original), A0_fn_original
    with open(A0_fn_original, "rb") as f:
        A0_original = pickle.load(f)
    assert A0_original["m"] == 8
    
    A0 = {}
    A0["m"] = A0_original["m"]
    A0["dataset"] = dataset
    A0["lgtNameList"] = A0_original["lgtNameList"]
    A0["lgtNameDict"] = A0_original["lgtNameDict"]
    A0["lgtNameRetrieveList"] = A0_original["lgtNameRetrieveList"]
    A0["nameList"] = ["v/R/%s/R1/%08d.exr" % (dataset, j) for j in range(int(A0["m"]))]
    A0["lgtType"] = A0_original["lgtType"]
    A0["winWidth"] = 512 * np.ones((A0["m"],), dtype=np.int32)
    A0["winHeight"] = 256 * np.ones((A0["m"],), dtype=np.int32)
    A0["flagSplit"] = 3 * np.ones((A0["m"],), dtype=np.int32)
    # We do not do A0["hdrsum3"] here. Do it only if you need it.
    A0["nameListOriginalHighRes"] = A0_original["nameList"]

    for j in range(A0["m"]):
        assert os.path.isfile(projRoot + A0["nameListOriginalHighRes"][j]), projRoot + A0["nameListOriginalHighRes"][j]

    A0_fn = projRoot + "v/A/%s/A0_main.pkl" % dataset
    with open(A0_fn, "wb") as f:
        pickle.dump(A0, f)


if __name__ == "__main__":
    main()
