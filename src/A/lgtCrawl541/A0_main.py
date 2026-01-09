import os
import sys
import pickle
import typing
import numpy as np
import cv2
from collections import OrderedDict
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
sys.path.append(projRoot + "src/versions/")

from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb_np
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    saveRoot = projRoot + "v/A/%s/" % dataset

    m = 541
    # envmap_root = "/shared/yang/zhusz/remote_fastdata_yang/sa/crawl/polyhaven541/"
    envmap_root = projRoot + "v/fast/envmaps/crawl/polyhaven541/"
    fns = sorted(os.listdir(envmap_root))
    assert len(fns) == m
    
    lgtNameList = [fn[:-4] for fn in fns]
    lgtNameDict = {lgtNameList[j]: j for j in range(m)}
    lgtNameRetrieveList = {j: lgtNameList[j] for j in range(m)}
    nameList = [
        ("v/fast/envmaps/crawl/polyhaven541/" + x + ".exr") for x in lgtNameList
    ]
    lgtType = "envmap"
    winWidth = 4096 * np.ones((m,), dtype=np.int32)
    winHeight = 2048 * np.ones((m,), dtype=np.int32)
    
    with open(saveRoot + "A0_main.pkl", "wb") as f:
        pickle.dump({
            "m": m,
            "dataset": dataset,
            "lgtNameList": lgtNameList,
            "lgtNameDict": lgtNameDict,
            "lgtNameRetrieveList": lgtNameRetrieveList,
            "nameList": nameList,
            "lgtType": lgtType,
            "winWidth": winWidth,
            "winHeight": winHeight,
        }, f)

    """
    visualDir = projRoot + "cache/A/%s/dumpAll/" % dataset
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 50, "dumpAll")
    brFreq = 5

    summary0 = OrderedDict([])
    txt0 = []
    for j in range(m):
        fn = fns[j]
        print("Processing A0 for %s: (%s)-%d / %d" % (dataset, fn, j, m))

        envmap_bgr = cv2.imread(envmap_root + fn, flags=cv2.IMREAD_UNCHANGED)
        # assert envmap_bgr.shape == (2048, 4096, 3), (envmap_bgr.shape, fn)
        # assert envmap_bgr.dtype == np.float32, (envmap_bgr.dtype, fn)

        envmap_shape_message = ""
        if envmap_bgr.shape == (2048, 4096, 3):
            pass
        else:
            envmap_shape_message = ("shape is: %s" % str(envmap_bgr.shape))

        envmap_dtype_message = ""
        if envmap_bgr.dtype == np.float32:
            pass
        else:
            envmap_dtype_message = ("dtype is: %s"% str(envmap_bgr.dtype))

        envmaphdr = np.stack([envmap_bgr[:, :, 2], envmap_bgr[:, :, 1], envmap_bgr[:, :, 0]], 2)
        envmapldr = np.clip(tonemap_srgb_to_rgb_np(envmaphdr), a_min=0, a_max=1)
        envmapldr_resized = cv2.resize(envmapldr, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        
        summary0["envmapldr %d (%s) %s %s" % (j, fn, envmap_shape_message, envmap_dtype_message)] = \
            envmapldr_resized
        txt0.append("")
        if j % brFreq == brFreq - 1:
            htmlStepper.step2(
                summary0, txt0, brInds=(0,),
                headerMessage="%s %d-%d" % (dataset, j - brFreq + 1, j),
                subMessage="",
            )
            summary0 = OrderedDict([])
            txt0 = []
    """


if __name__ == "__main__":
    main()
