# (nerf)

import numpy as np
import torch
from mediancut_v1.csrc.mediancut_main import doMedianCut


def doMedianCut_np_wrapper(envmap, n, smallHdrSumThre=0.1):
    # Inputs
    # envmap: (winHeight, winWidth, 3(hdr-rgb)) must be float32
    # n: to cut the envmap into 2^n regions. We also define m to be 2^n
    
    # Outputs
    # m: a number that can be slightly smaller than 2 ** n, due to the fact that very small region (2x2) won't be further splitted.
    # outputRegion: (m, 4)  (XDXD)
    # outputCentroid: (m, 2) still float32
    # outputAccumulatedLuminance: (m,)

    assert len(envmap.shape) == 3, envmap.shape
    assert envmap.shape[2] == 3, envmap.shape
    assert envmap.dtype == np.float32, envmap.dtype
    assert n >= 1, n

    luminance = 0.2125 * envmap[:, :, 0] + 0.7154 * envmap[:, :, 1] + 0.0721 * envmap[:, :, 2]
    integralMap = np.cumsum(np.cumsum(luminance, axis=1), axis=0)

    tot = 2 ** n  # m can be smaller than tot
    outputM = torch.zeros((1,), dtype=torch.int32)
    outputRegion = torch.zeros(tot, 4, dtype=torch.int32)
    outputCentroid = torch.zeros(tot, 2, dtype=torch.float32)
    outputAccumulatedLuminance = torch.zeros(tot, dtype=torch.float32)

    outputM, outputRegion, outputCentroid, outputAccumulatedLuminance = doMedianCut(
        outputM,
        outputRegion,
        outputCentroid, 
        outputAccumulatedLuminance,
        torch.from_numpy(integralMap),
        n,
    )
    m = int(outputM[0])
    outputRegion = outputRegion[:m, :]
    outputCentroid = outputCentroid[:m, :]
    outputAccumulatedLuminance = outputAccumulatedLuminance[:m]

    assert torch.all(0 <= outputRegion[:, 0])
    assert torch.all(outputRegion[:, 0] <= outputRegion[:, 1])
    # assert torch.all(outputRegion[:, 1] < int(envmap.shape[1]))
    outputRegion[:, 1] = torch.clamp(outputRegion[:, 1], max=int(envmap.shape[1] - 1))
    assert torch.all(0 <= outputRegion[:, 2])
    assert torch.all(outputRegion[:, 2] <= outputRegion[:, 3])
    # assert torch.all(outputRegion[:, 3] < int(envmap.shape[0]))
    outputRegion[:, 3] = torch.clamp(outputRegion[:, 3], max=int(envmap.shape[0] - 1))

    # also filter out small value hdr region
    ind = np.where(outputAccumulatedLuminance > smallHdrSumThre)[0]
    m = int(ind.shape[0])
    outputRegion = outputRegion[ind, :]
    outputCentroid = outputCentroid[ind, :]
    outputAccumulatedLuminance = outputAccumulatedLuminance[ind]

    return (
        m,
        outputRegion.numpy(),
        outputCentroid.numpy(),
        outputAccumulatedLuminance.numpy(),
    )


def mainDoMedianCut():
    import os
    import sys
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
    sys.path.append(projRoot + "src/versions/")
    from codes_py.toolbox_show_draw.draw_v1 import drawPoint, drawBoxXDXD
    from codes_py.toolbox_graphics.tonemap_v1 import tonemap_srgb_to_rgb
    from matplotlib.cm import get_cmap
    envmapRoot = projRoot + "remote_fastdata/envmaps/"
    import cv2
    envmap = cv2.imread(
        envmapRoot + "sausalito_office_overcast_2.exr",
        flags=cv2.IMREAD_UNCHANGED,
    )
    assert len(envmap.shape) == 3
    assert int(envmap.shape[2]) in [3]
    envmap = np.stack([envmap[:, :, 2], envmap[:, :, 1], envmap[:, :, 0]], 2)
    """
    if envmap.shape[2] == 3:
        envmap = np.concatenate([envmap, np.ones_like(envmap[:, :, :1])], 2)
    """
    n = 9
    luminance = 0.2125 * envmap[:, :, 0] + 0.7154 * envmap[:, :, 1] + 0.0721 * envmap[:, :, 2]
    outputRegion, outputCentroid, outputAccumulatedLuminance = doMedianCut_np_wrapper(
        envmap, n
    )
    m = 2 ** n
    """
    cmap = get_cmap("inferno")
    envmap = np.clip(tonemap_srgb_to_rgb(torch.from_numpy(envmap)).numpy(), a_min=0, a_max=1)
    for j in range(m):
        if j % 100 == 0:
            print(j)
        envmap = drawBoxXDXD(envmap, outputRegion[j, :], lineWidthFloat=2, rgb=np.array(cmap(float(j) / m)[:3], dtype=np.float32))
        envmap = drawPoint(envmap, outputCentroid[j:j+1, :], color=np.array(cmap(float(j) / m)[:3], dtype=np.float32))

    cv2.imwrite("out.png", envmap[:, :, ::-1] * 255)
    cv2.imwrite("outLum.png", outputAccumulatedLuminance * 255)
    """
    ldr = np.clip(tonemap_srgb_to_rgb(torch.from_numpy(luminance)).numpy(), a_min=0, a_max=1)
    cv2.imwrite("ldr.png", ldr * 255)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == "__main__":
    mainDoMedianCut()
