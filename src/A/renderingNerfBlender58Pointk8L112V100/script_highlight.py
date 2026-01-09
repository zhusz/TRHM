import os
import sys
import cv2
import pickle
import numpy as np
from collections import OrderedDict
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
sys.path.append(projRoot + "src/versions/")
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb_np


bgrmean_highlight_thre_table = {
    0: ("max", 2.0),
    1: ("max", 0.5),
    3: ("max", 1.5),
    5: ("max", 10.0),
    7: ("mean", 1.0),
}


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    A0_fn = projRoot + "v/A/%s/A0_randomness.pkl" % dataset
    assert os.path.isfile(A0_fn), A0_fn

    with open(A0_fn, "rb") as f:
        A0 = pickle.load(f)

    caseID = int(os.environ["caseID"])

    ind = np.where((A0["caseIDList"] == caseID) & (A0["flagSplit"] == 1))[0]

    winHeight = 800
    winWidth = 800
    xi = np.arange(winWidth).astype(np.int32)
    yi = np.arange(winHeight).astype(np.int32)
    x, y = np.meshgrid(xi, yi)
    x = x.reshape(-1)
    y = y.reshape(-1)
    sublight_dilate_kernel = np.ones((3, 3), dtype=np.float32)

    visualDir = projRoot + "cache/A/%s/highlight/" % dataset
    assert os.path.isdir(visualDir), visualDir
    htmlStepper = HTMLStepper(visualDir, 100, "highlight")

    # p1 = int(os.environ["J1"])  # j ranges in [0, 11216), no need to add the zero index (7 * 11216)
    # p2 = int(os.environ["J2"])

    indexID_highlight = []
    pixelID_highlight = []
    hdr_highlight = []
    indexID_sublight = []
    pixelID_sublight = []
    hdr_sublight = []

    output_fn = projRoot + "v/misc/%s/%s_caseID_%d_highlight.pkl" % (dataset, dataset, caseID)
    assert os.path.isdir(os.path.dirname(output_fn)), os.path.dirname(output_fn)

    for i, j in enumerate(ind.tolist()):
        fn_input = projRoot + A0["nameListExr"][j]
        print("Processing highlight for %s Case %d j = %d (%d / %d, progress = %.1f%%) fn_input = %s" % (
            dataset, caseID, j, i, ind.shape[0], float(i) / float(ind.shape[0]) * 100, fn_input
        ))
        assert os.path.isfile(fn_input), fn_input
        bgra = cv2.imread(fn_input, flags=cv2.IMREAD_UNCHANGED)
        assert bgra.shape == (800, 800, 4)
        assert bgra.dtype == np.float32
        a = bgra[:, :, 3]
        
        bgrfunc, bgrfunc_highlight_thre = bgrmean_highlight_thre_table[caseID]
        if bgrfunc == "mean":
            bgrmean = bgra[:, :, :3].mean(2)
            map_highlight = ((bgrmean > bgrfunc_highlight_thre) & (a > 0))
        elif bgrfunc == "max":
            bgrmax = bgra[:, :, :3].max(2)
            map_highlight = ((bgrmax > bgrfunc_highlight_thre) & (a > 0))
        else:
            raise NotImplementedError("Unknown bgrfunc: %s" % bgrfunc)
        y_highlight, x_highlight = np.where(map_highlight)
        map_sublight = cv2.dilate(
            map_highlight.astype(np.float32), sublight_dilate_kernel, iterations=1
        ).astype(bool)
        map_sublight[map_highlight] = False
        map_sublight[a == 0] = False
        y_sublight, x_sublight = np.where(map_sublight)

        bgr = bgra[:, :, :3]
        rgb = bgr[:, :, ::-1]

        # record
        indexID_highlight.append(
            j * np.ones((y_highlight.shape[0],), dtype=np.int32)
        )
        t = y_highlight * winWidth + x_highlight
        pixelID_highlight.append(t)
        hdr_highlight.append(rgb.reshape((-1, 3))[t, :])
        indexID_sublight.append(
            j * np.ones((y_sublight.shape[0],), dtype=np.int32)
        )
        t = y_sublight * winWidth + x_sublight
        pixelID_sublight.append(t)
        hdr_sublight.append(rgb.reshape((-1, 3))[t, :])

        # continue  # not to do visualization now
        # visualize
        ldr = np.clip(tonemap_srgb_to_rgb_np(rgb), a_min=0, a_max=1)
        r = ldr[:, :, 0]
        g = ldr[:, :, 1]
        b = ldr[:, :, 2]
        summary0 = OrderedDict([])
        txt0 = []
        summary0["ldr"] = ldr
        txt0.append("")
        r_highlight = r.copy()
        r_highlight[map_highlight] = 1
        g_highlight = g.copy()
        g_highlight[map_highlight] = 0
        b_highlight = b.copy()
        b_highlight[map_highlight] = 0
        summary0["ldr with highlight"] = np.stack([r_highlight, g_highlight, b_highlight], 2)
        txt0.append("# = %d" % y_highlight.shape[0])
        r_sublight = r_highlight.copy()
        r_sublight[map_sublight] = 0
        g_sublight = g_highlight.copy()
        g_sublight[map_sublight] = 0
        b_sublight = b_highlight.copy()
        b_sublight[map_sublight] = 1
        summary0["ldr with sublight"] = np.stack([r_sublight, g_sublight, b_sublight], 2)
        txt0.append("# = %d" % y_sublight.shape[0])
        htmlStepper.step2(
            summary0, txt0, brInds=(0,),
            headerMessage="Dataset %s caseID %d j = %d (%d / %d)" % (
                dataset, caseID, j, i, ind.shape[0]
            ),
            subMessage=fn_input,
        )

    indexID_highlight = np.concatenate(indexID_highlight, 0)
    pixelID_highlight = np.concatenate(pixelID_highlight, 0)
    hdr_highlight = np.concatenate(hdr_highlight, 0)
    indexID_sublight = np.concatenate(indexID_sublight, 0)
    pixelID_sublight = np.concatenate(pixelID_sublight, 0)
    hdr_sublight = np.concatenate(hdr_sublight, 0)

    # with open(output_fn, "wb") as f:
    #     pickle.dump({
    #         "indexID_highlight": indexID_highlight,
    #         "pixelID_highlight": pixelID_highlight,
    #         "hdr_highlight": hdr_highlight,
    #         "indexID_sublight": indexID_sublight,
    #         "pixelID_sublight": pixelID_sublight,
    #         "hdr_sublight": hdr_sublight,
    #     }, f)


if __name__ == "__main__":
    main()
