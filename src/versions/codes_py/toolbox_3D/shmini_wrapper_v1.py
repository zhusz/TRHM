# (local)
import numpy as np
import torch
from shmini_v1.csrc.sh_main import applyRotation, projectEnvironment


def applyRotation_np_wrapper(L, q, sh_input):
    # input
    # L: int (so that N == (L + 1) ** 2)
    # sh_input: (m, N) or (m, N, 3)

    # output
    # sh_output: (m, N) or (m, N, 3)

    assert q.dtype == np.float32
    assert q.shape == (4,)
    assert sh_input.dtype == np.float32
    m = sh_input.shape[0]
    N = (L + 1) ** 2
    assert sh_input.ndim in [2, 3]
    if sh_input.ndim == 2:
        sh_input_input = np.tile(sh_input[:, :, None], (1, 1, 3))
    elif sh_input.ndim == 3:
        sh_input_input = sh_input.copy()
    else:
        raise NotImplementedError("Unknown sh_input.ndim: %d" % sh_input.ndim)
    sh_output = torch.zeros(m, N, 3, dtype=torch.float32)
    sh_output = applyRotation(
        L, torch.from_numpy(q), torch.from_numpy(sh_input_input), sh_output
    )
    sh_output = sh_output.numpy()

    if sh_input.ndim == 2:
        sh_output = sh_output[:, :, 0]
    elif sh_input.ndim == 3:
        sh_output = sh_output
    else:
        raise NotImplementedError("Unknown sh_input.ndim: %d" % sh_input.ndim)

    return sh_output


# https://github.com/zhusz/ICLR22-DGS/blob/reimplemented/src/versions/codes_py/toolbox_3D/rotations_v1.py#L179
# MIT license
def getQuatBatchNP(rotationAxis, degrees):  # functional
    assert len(rotationAxis.shape) == 2 and rotationAxis.shape[1] == 3
    assert len(degrees.shape) == 1 and rotationAxis.shape[0] == degrees.shape[0]

    norms = np.linalg.norm(rotationAxis, ord=2, axis=1)
    assert norms.min() > 0.0
    normalized = np.divide(rotationAxis, norms[:, None])

    halfRadian = degrees * 0.5 * np.pi / 180.0
    vCos = np.cos(halfRadian)
    vSin = np.sin(halfRadian)
    quat = np.concatenate([vCos[:, None], vSin[:, None] * normalized], 1)
    return quat


def check_applyRotation_np_wrapper():
    rotationAxis = np.array([[0, 0, 1]], dtype=np.float32)
    degrees = np.array([180], dtype=np.float32)
    q = getQuatBatchNP(rotationAxis, degrees)[0]
    sh_input = np.array(
        [0.433024, +0.5, 0, 0, 0, 0, -0.121034, 0, -0.209616], dtype=np.float32
    )
    sh_output = applyRotation_np_wrapper(2, q, sh_input[None, :])
    import ipdb

    ipdb.set_trace()
    print(1 + 1)


def projectEnvironment_np_wrapper(L, to_be_fit):
    # input
    # L: int
    # to_be_fit: (H, W) or (H, W, 3)

    # output
    # result: ((L + 1) * (L + 1), ) or ((L + 1) * (L + 1), 3)

    assert to_be_fit.dtype == np.float32
    assert int(to_be_fit.ndim) in [2, 3]
    if to_be_fit.ndim == 2:
        to_be_fit_input = np.tile(to_be_fit[:, :, None], (1, 1, 3))
    elif to_be_fit.ndim == 3:
        assert to_be_fit.shape[2] == 3
        to_be_fit_input = to_be_fit
    else:
        raise NotImplementedError("Unknown to_be_fit.dim(): %d" % to_be_fit.ndim)

    N = (L + 1) ** 2
    result_output = torch.zeros(N, 3, dtype=torch.float32)
    result_output = projectEnvironment(
        L, torch.from_numpy(to_be_fit_input), result_output
    )
    result_output = result_output.numpy()

    if to_be_fit.ndim == 2:
        result = result_output[:, 0]
    elif to_be_fit.ndim == 3:
        result = result_output
    else:
        raise NotImplementedError("Unknown to_be_fit.dim(): %d" % to_be_fit.ndim)

    return result


def check_projectEnvironment_np_wrapper():
    import os
    import sys

    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
    sys.path.append(projRoot + "src/versions/")
    from codes_py.third.sh import eval_sh

    L = 2
    deg = L
    sh = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    W = 512
    H = 256
    stepH = np.pi / float(H)
    stepW = 2.0 * np.pi / float(W)
    xi = np.linspace(0.5 * stepW, 2.0 * np.pi - 0.5 * stepW, W).astype(np.float32)
    yi = np.linspace(0.5 * stepH, np.pi - 0.5 * stepH, H).astype(np.float32)
    x, y = np.meshgrid(xi, yi)
    dirs = np.stack(
        [np.cos(x) * np.sin(y), np.sin(x) * np.sin(y), np.cos(y)], -1
    )  # (H, W, 3(xyz))
    envmap = eval_sh(L, sh, dirs)[:, :, 0]
    envmap[envmap < 0] = 0  # fitting the relu-out version
    result = projectEnvironment_np_wrapper(L, envmap)

    """
    import imageio
    imageio.plugins.freeimage.download()
    with open("./debug.exr", "wb") as f:
        imageio.imwrite(f, envmap, format="exr")
    """

    import ipdb

    ipdb.set_trace()
    print(1 + 1)


if __name__ == "__main__":
    check_applyRotation_np_wrapper()
