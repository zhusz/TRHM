# Written with the help of codebase of https://github.com/zhusz/ICLR22-DGS/tree/reimplemented, released
# under the following license:

# MIT License
#
# Copyright (c) 2022 Shizhan Zhu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


def to_heatmap(x, vmin=None, vmax=None, cmap=None):
    x = x.astype(np.float32)
    upperBound = x[np.isfinite(x)].max() if vmax is None else vmax
    lowerBound = x[np.isfinite(x)].min() if vmin is None else vmin
    x = (x - lowerBound) / (upperBound - lowerBound)
    cm = plt.get_cmap(cmap, 2**16)
    return cm(x)[..., :3]


def getPltDraw(f):
    fig = plt.figure()
    ax = plt.gca()

    f(ax)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(
        (h, w, 4)
    )  # It is ARGB!
    buf = buf[:, :, [1, 2, 3, 0]]
    assert buf[:, :, -1].min() == 255
    render = buf[:, :, :3].astype(np.float32) / 255.0
    # plt.clf()
    plt.close()

    return render


def drawBoxXDXD(img0, bboxXDXD0, lineWidthFloat=None, rgb=np.array([1., 0., 0.]), txt=None):  # functional, non-batched
    assert len(bboxXDXD0.shape) == 1 and bboxXDXD0.shape[0] == 4
    if lineWidthFloat is None:
        lineWidthFloat = math.sqrt(img0.shape[0] ** 2 + img0.shape[1] ** 2) * 3. / 2000.
    lineWidth = int(math.ceil(lineWidthFloat))
    b = np.around(bboxXDXD0).astype(np.int32)
    imgNew0 = img0.copy()
    cv2.rectangle(imgNew0, (b[0], b[2]), (b[1], b[3]), rgb.tolist(), lineWidth)
    if txt is not None:
        fontMin = 0.5
        fontMax = 1.
        heightFloat = b[3] - b[2]
        fontSize = max(heightFloat / 100., fontMin)
        fontSize = min(fontSize, fontMax)
        fontHeight = int(fontSize * 25.)
        fontWidth = int(fontHeight * 0.7 * len(txt))
        cv2.rectangle(imgNew0, (b[0], b[2] - fontHeight), (b[0] + fontWidth, b[2]), rgb.tolist(), -1)
        cv2.putText(imgNew0, txt, (b[0], b[2]), cv2.FONT_HERSHEY_TRIPLEX, fontSize, [0, 0, 0])
    return imgNew0


def drawPoint(img0, data0, vData0=None, vThre=0.01, radius=3, color=[1., 0., 0.]):  # functional, non-batched
    newImg0 = img0.copy()
    assert len(img0.shape) == 3 and img0.shape[2] == 3
    assert len(data0.shape) == 2 and data0.shape[1] == 2
    nPts = data0.shape[0]
    if vData0 is not None:
        assert len(vData0.shape) == 1 and vData0.shape[0] == nPts

    for j in range(nPts):
        if vData0 is None or vData0[j] > vThre:
            # assert data0[j, 0] != 0
            # assert data0[j, 1] != 0
            cv2.circle(newImg0,
                       (int(data0[j, 0]), int(data0[j, 1])),
                       int(radius + 0.5),
                       [float(x * 1.) for x in color],
                       -1)
    return newImg0
