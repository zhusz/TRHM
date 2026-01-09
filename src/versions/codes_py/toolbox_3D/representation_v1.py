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

import numpy as np
import skimage.measure


def voxSdfSign2mesh_skmc(voxSdfSign, goxyz, sCell, **kwargs):
    level = kwargs.get("level", 0.5)
    assert len(voxSdfSign.shape) == 3
    assert len(goxyz) == 3
    if type(sCell) is float:
        sCellX = sCell
        sCellY = sCell
        sCellZ = sCell
    else:
        assert len(sCell) == 3
        sCellX = float(sCell[0])
        sCellY = float(sCell[1])
        sCellZ = float(sCell[2])

    # vert0, faceClockWise0 = mcubes.marching_cubes(voxSdfSign.transpose((1, 0, 2)), 0.5)  # since voxSdfSign == 1 is unoccupied
    vert0, face0, normal0, value0 = skimage.measure.marching_cubes(
        voxSdfSign.transpose((1, 0, 2)),
        level=level,
        spacing=[1.0, 1.0, 1.0],
        method="lewiner",
    )

    # face0 = faceClockWise0[:, [0, 2, 1]]  # It seems this function does not need to do wrapping.
    # It has one sign gap.
    vert0 = vert0.astype(np.float32)
    face0 = face0.astype(np.int32)

    vert0[:, 0] = (vert0[:, 0] - 0.5) * sCellX + float(goxyz[0])
    vert0[:, 1] = (vert0[:, 1] - 0.5) * sCellY + float(goxyz[1])
    vert0[:, 2] = (vert0[:, 2] - 0.5) * sCellZ + float(goxyz[2])

    return vert0, face0
