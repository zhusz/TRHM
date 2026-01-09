import os
import numpy as np
import pickle
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    with open(projRoot + "v/A/%s/A0_main.pkl" % dataset, "rb") as f:
        A0 = pickle.load(f)

    m = int(A0["m"])
    j1 = int(os.environ["J1"])
    j2 = int(os.environ["J2"])

    for j in range(j1, j2):
        output_fn = projRoot + A0["nameList"][j]
        if os.path.isfile(output_fn):
            print("Skipping R1 for %s: %d / %d (j1 = %d, j2 = %d, progress = %.f%%)." % (
                dataset, j, m, j1, j2, float(j - j1) / float(j2 - j1) * 100
            ))
            continue
        else:
            print("Processing R1 for %s: %d / %d (j1 = %d, j2 = %d, progress = %.f%%)." % (
                dataset, j, m, j1, j2, float(j - j1) / float(j2 - j1) * 100
            ))
        # input_fn = "/shared/zhusz/data2/remote_slowdata_remote/sa/" + (
        #     A0["nameListOriginalHighRes"][j][len("remote_fastdata/"):]
        # )
        input_fn = projRoot + A0["nameListOriginalHighRes"][j]
        assert os.path.isfile(input_fn), input_fn
        tmp = cv2.imread(input_fn, flags=cv2.IMREAD_UNCHANGED)
        tmp = cv2.resize(tmp, (int(A0["winWidth"][j]), int(A0["winHeight"][j])), interpolation=cv2.INTER_LINEAR)
        assert os.path.isdir(os.path.dirname(output_fn)), output_fn
        cv2.imwrite(output_fn, tmp)


if __name__ == "__main__":
    main()
