import os
import pickle
import numpy as np
projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    with open(projRoot + "v/A/lgtCrawl541/A0_main.pkl", "rb") as f:
        A0_highres = pickle.load(f)
    A0 = A0_highres  # we only change the winWidth and winHeight, and of course nameList staffs
    winWidth = 512
    winHeight = 256
    A0["winWidth"][:] = winWidth
    A0["winHeight"][:] = winHeight
    A0["nameListOriginalHighRes"] = A0["nameList"]
    A0["nameList"] = ["v/R/%s/R1/%08d.exr" % (dataset, j) for j in range(int(A0["m"]))]

    m = int(A0["m"])
    A0["flagSplit"] = 3 * np.ones((m,), dtype=np.int32)

    output_fn = projRoot + "v/A/%s/A0_main.pkl" % dataset
    with open(output_fn, "wb") as f:
        pickle.dump(A0, f)


if __name__ == "__main__":
    main()
