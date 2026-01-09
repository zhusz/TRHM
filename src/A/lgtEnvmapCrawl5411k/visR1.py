import os
import sys
import numpy as np
import pickle
projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
sys.path.append(projRoot + "src/B/")
sys.path.append(projRoot + "src/versions/")
from collections import OrderedDict
from Bprelight4.testDataEntry.renderingNerfBlender.renderingLightStageLfhyper import OnlineAugmentableEnvmapDatasetCache
from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb_np
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    cudaDevice = "cuda:0"
    envmapDatasetObj = OnlineAugmentableEnvmapDatasetCache(
        lgtDataset=dataset,
        projRoot=projRoot,
        cudaDevice=cudaDevice,
        if_lgt_load_highres=False,
        quantile_cut_min=0.9999,
        quantile_cut_max=1.0,
        quantile_cut_fixed=1.0,
    )

    A0_fn = projRoot + "v/A/%s/A0_main.pkl" % dataset
    assert os.path.isfile(A0_fn), A0_fn
    with open(A0_fn, "rb") as f:
        A0 = pickle.load(f)

    visualDir = projRoot + "cache/A/%s/visR1/" % dataset
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 20, "visR1")

    summary0 = OrderedDict([])
    txt0 = []
    count = 0
    for j in range(envmapDatasetObj.m):
        print("%s visR1 %d / %d" % (dataset, j, envmapDatasetObj.m))

        tmp = envmapDatasetObj.queryEnvmap(
            ind=np.array([j], dtype=np.int32),
            if_augment=False,
            if_return_all=True,
        )
        envmap_original_normalized = tmp["envmap_original_normalized"]
        hdr = envmap_original_normalized[0, :, :, :].detach().cpu().numpy()
        ldr = np.clip(tonemap_srgb_to_rgb_np(hdr), a_min=0, a_max=1)

        summary0["%s j = %d (flagSplit = %d)" % (dataset, j, A0["flagSplit"][j])] = ldr
        txt0.append("")
        count += 1

        if count == 4:
            htmlStepper.step2(
                summary0, txt0, brInds=(0, ),
                headerMessage="%s" % dataset,
                subMessage="",
            )
            summary0 = OrderedDict([])
            txt0 = []
            count = 0


if __name__ == "__main__":
    main()
