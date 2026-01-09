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

# import meshzoo
import numpy as np

# from meshzoo import uv_sphere as meshzoo_uv_sphere


def create_cube_mesh(
    r,
):  # This can only create tiny cube surface. For large cube, turn to meshzoo_augmented_v1.py
    v0 = np.array([-r, -r, -r], dtype=np.float32)
    v1 = np.array([-r, -r, +r], dtype=np.float32)
    v2 = np.array([-r, +r, -r], dtype=np.float32)
    v3 = np.array([-r, +r, +r], dtype=np.float32)
    v4 = np.array([+r, -r, -r], dtype=np.float32)
    v5 = np.array([+r, -r, +r], dtype=np.float32)
    v6 = np.array([+r, +r, -r], dtype=np.float32)
    v7 = np.array([+r, +r, +r], dtype=np.float32)

    f = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 5],
            [0, 5, 4],
            [0, 4, 2],
            [2, 4, 6],
            [4, 5, 7],
            [4, 7, 6],
            [2, 6, 3],
            [3, 6, 7],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    f = f[:, [0, 2, 1]]  # Change to the conventional counter-clock-wise case.

    return np.stack([v0, v1, v2, v3, v4, v5, v6, v7], 0), f


def create_cuboid_mesh(min0, max0):
    xl, yl, zl = min0[0], min0[1], min0[2]
    xr, yr, zr = max0[0], max0[1], max0[2]
    v0 = np.array([xl, yl, zl], dtype=np.float32)
    v1 = np.array([xl, yl, zr], dtype=np.float32)
    v2 = np.array([xl, yr, zl], dtype=np.float32)
    v3 = np.array([xl, yr, zr], dtype=np.float32)
    v4 = np.array([xr, yl, zl], dtype=np.float32)
    v5 = np.array([xr, yl, zr], dtype=np.float32)
    v6 = np.array([xr, yr, zl], dtype=np.float32)
    v7 = np.array([xr, yr, zr], dtype=np.float32)

    f = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 5],
            [0, 5, 4],
            [0, 4, 2],
            [2, 4, 6],
            [4, 5, 7],
            [4, 7, 6],
            [2, 6, 3],
            [3, 6, 7],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    f = f[:, [0, 2, 1]]  # Change to the conventional counter-clock-wise case.

    return np.stack([v0, v1, v2, v3, v4, v5, v6, v7], 0), f


def create_tetrahedron_mesh(r):  # , maxvol=0.1):
    # copied from https://github.com/nschloe/meshzoo/blob/master/examples/meshpy/tetrahedron.py
    # circumcircle radius
    # r = 5.0

    r = r * 2.0

    # boundary points
    points = []
    points.append((0.0, 0.0, r))
    # theta = arccos(-1/3) (tetrahedral angle)
    costheta = -1.0 / 3.0
    sintheta = 2.0 / 3.0 * np.sqrt(2.0)
    # phi = 0.0
    sinphi = 0.0
    cosphi = 1.0
    points.append((r * cosphi * sintheta, r * sinphi * sintheta, r * costheta))
    # phi = np.pi * 2.0 / 3.0
    sinphi = np.sqrt(3.0) / 2.0
    cosphi = -0.5
    points.append((r * cosphi * sintheta, r * sinphi * sintheta, r * costheta))
    # phi = - np.pi * 2.0 / 3.0
    sinphi = -np.sqrt(3.0) / 2.0
    cosphi = -0.5
    points.append((r * cosphi * sintheta, r * sinphi * sintheta, r * costheta))

    # boundary faces
    facets = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]

    return np.array(points, dtype=np.float32), np.array(facets, dtype=np.int32)


def composeSingleShape(
    xyz, radius, color, shapeName, ifSolidRatherThanBubble=True, **kwargs
):
    assert len(xyz) == 3
    assert type(radius) is float
    assert len(color) == 3

    if shapeName == "sphere":
        raise NotImplementedError
        # v0, t0 = meshzoo_uv_sphere(
        #     num_points_per_circle=kwargs.get("num_points_per_circle", 12),
        #     num_circles=kwargs.get("num_circles", 6),
        #     radius=radius,
        # )
        # t0 = t0[:, [0, 2, 1]]
    elif shapeName == "cube":
        v0, t0 = create_cube_mesh(r=radius)
        t0 = t0[:, [0, 2, 1]]
    elif shapeName == "tetrahedron":
        v0, t0 = create_tetrahedron_mesh(r=radius)
    else:
        raise NotImplementedError("Unknown shapeName %s" % shapeName)

    if not ifSolidRatherThanBubble:
        t0 = t0[:, [0, 2, 1]]  # if it is a bubble, you need to make it inside out.
    c0 = np.zeros_like(v0)

    # Modification on v0
    v0[:, 0] += xyz[0]
    v0[:, 1] += xyz[1]
    v0[:, 2] += xyz[2]
    v0 = v0.astype(np.float32)

    # Modification on t0
    # nVert = vert0.shape[0]
    # t0 += nVert
    f0 = t0  # [:, [0, 2, 1]]
    f0 = f0.astype(np.int32)

    # Modification on c0
    c0[:, 0] = color[0]
    c0[:, 1] = color[1]
    c0[:, 2] = color[2]
    c0 = c0.astype(np.float32)

    return v0, f0, c0


def create_cuboid_bone_mesh(rBone, cxyz, rxyz, color):
    assert len(cxyz) == 3
    if type(cxyz) is list:
        cxyz = np.array(cxyz, dtype=np.float32)
    assert len(rxyz) == 3
    if type(rxyz) is list:
        rxyz = np.array(rxyz, dtype=np.float32)
    signTable = np.array(
        [
            [-1, -1, -1],
            [-1, -1, +1],
            [-1, +1, -1],
            [-1, +1, +1],
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
        ],
        dtype=np.float32,
    )
    vfc0 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[1], cxyz + rxyz * signTable[0], color
    )
    vfc1 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[3], cxyz + rxyz * signTable[1], color
    )
    vfc2 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[2], cxyz + rxyz * signTable[3], color
    )
    vfc3 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[0], cxyz + rxyz * signTable[2], color
    )

    vfc4 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[5], cxyz + rxyz * signTable[4], color
    )
    vfc5 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[7], cxyz + rxyz * signTable[5], color
    )
    vfc6 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[6], cxyz + rxyz * signTable[7], color
    )
    vfc7 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[4], cxyz + rxyz * signTable[6], color
    )

    vfc8 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[4], cxyz + rxyz * signTable[0], color
    )
    vfc9 = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[5], cxyz + rxyz * signTable[1], color
    )
    vfca = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[6], cxyz + rxyz * signTable[2], color
    )
    vfcb = create_lineArrow_mesh(
        rBone, 0, cxyz + rxyz * signTable[7], cxyz + rxyz * signTable[3], color
    )

    vfc = combineMultiShapes_withVertRgb(
        [vfc0, vfc1, vfc2, vfc3, vfc4, vfc5, vfc6, vfc7, vfc8, vfc9, vfca, vfcb]
    )
    return vfc  # tuple of (vert0, face0, vertRgb0)


def create_cuboid_bone_mesh_from_cornerBound(rBone, cornerBound0, color):
    assert cornerBound0.shape == (8, 3)
    vfc0 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[0, :], cornerBound0[1, :], color
    )
    vfc1 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[1, :], cornerBound0[3, :], color
    )
    vfc2 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[3, :], cornerBound0[2, :], color
    )
    vfc3 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[2, :], cornerBound0[0, :], color
    )

    vfc4 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[4, :], cornerBound0[5, :], color
    )
    vfc5 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[5, :], cornerBound0[7, :], color
    )
    vfc6 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[7, :], cornerBound0[6, :], color
    )
    vfc7 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[6, :], cornerBound0[4, :], color
    )

    vfc8 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[0, :], cornerBound0[4, :], color
    )
    vfc9 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[1, :], cornerBound0[5, :], color
    )
    vfc10 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[2, :], cornerBound0[6, :], color
    )
    vfc11 = create_lineArrow_mesh(
        rBone, 0, cornerBound0[3, :], cornerBound0[7, :], color
    )

    vfc = combineMultiShapes_withVertRgb(
        [vfc0, vfc1, vfc2, vfc3, vfc4, vfc5, vfc6, vfc7, vfc8, vfc9, vfc10, vfc11]
    )
    return vfc  # tuple of (vert0, face0, vertRgb0)


def create_pyramidK_bone_mesh(rBone, apexXyz, baseK, color):
    assert len(apexXyz.shape) == 1 and apexXyz.shape[0] == 3
    assert len(baseK.shape) == 2 and baseK.shape[1] == 3
    K = baseK.shape[0]
    vfcList = []
    for k in range(K):
        # apex-base link
        vfcList.append(create_lineArrow_mesh(rBone, 0, apexXyz, baseK[k], color))
        # base-base link
        vfcList.append(
            create_lineArrow_mesh(rBone, 0, baseK[k], baseK[(k + 1) % K], color)
        )
    vfc = combineMultiShapes_withVertRgb(vfcList)
    return vfc


def create_lineArrow_mesh(rLine, rArrow, origin_xyz, terminal_xyz, color):
    assert len(terminal_xyz) == 3
    assert len(origin_xyz) == 3
    assert len(color) == 3
    if type(origin_xyz) is list:
        origin_xyz = np.array(origin_xyz).astype(np.float32)
    if type(origin_xyz) is list:
        origin_xyz = np.array(origin_xyz).astype(np.float32)
    if type(color) is list:
        color = np.array(color).astype(np.float32)
    # before rotation (pointing A is 000 B is 001, pointing toward the +z direction)
    #  line
    delta_xyz = terminal_xyz - origin_xyz
    L = (delta_xyz[0] ** 2 + delta_xyz[1] ** 2 + delta_xyz[2] ** 2).sum() ** 0.5
    v0 = np.array([-rLine, -rLine, 0], dtype=np.float32)
    v1 = np.array([-rLine, -rLine, L], dtype=np.float32)
    v2 = np.array([-rLine, +rLine, 0], dtype=np.float32)
    v3 = np.array([-rLine, +rLine, L], dtype=np.float32)
    v4 = np.array([+rLine, -rLine, 0], dtype=np.float32)
    v5 = np.array([+rLine, -rLine, L], dtype=np.float32)
    v6 = np.array([+rLine, +rLine, 0], dtype=np.float32)
    v7 = np.array([+rLine, +rLine, L], dtype=np.float32)
    v_line = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], 0)
    f_line = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 5],
            [0, 5, 4],
            [0, 4, 2],
            [2, 4, 6],
            [4, 5, 7],
            [4, 7, 6],
            [2, 6, 3],
            [3, 6, 7],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    if rArrow > 0:
        f_line = f_line[
            :, [0, 2, 1]
        ]  # Change to the conventional counter-clock-wise case.
        #  arrow
        v_arrow, f_arrow = create_tetrahedron_mesh(rArrow)
        v_arrow[:, 2] += L
        #  merge
        v, f = combineMultiShapes_plain([(v_line, f_line), (v_arrow, f_arrow)])
    else:
        v, f = v_line, f_line
    # after rotation
    p0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    p1 = np.array(delta_xyz, dtype=np.float32) / L
    c = np.cross(p0, p1)
    d = np.dot(p0, p1)
    np0 = np.linalg.norm(p0, ord=2)
    if np.abs(c).sum() == 0:
        R = (np.sign(d) * np.linalg.norm(p1, ord=2) / np0) * np.eye(3).astype(
            np.float32
        )
    else:
        Z = np.array(
            [[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]], dtype=np.float32
        )
        R = (
            np.eye(3).astype(np.float32)
            + Z
            + (np.matmul(Z, Z)) * (1 - d) / (np.linalg.norm(c, ord=2) ** 2) / (np0**2)
        )
    R = R.astype(np.float32)
    if np.linalg.det(R) > 0:  # The 1 case (the else is the -1 case)
        f = f[:, [0, 2, 1]]
    v = np.matmul(v, R.transpose()) + origin_xyz[None, :]
    c = np.tile(color, [v.shape[0], 1])
    return v, f, c


def batch_create_lineArrow_mesh(rLine, rArrow, origin_xyz, terminal_xyz, color, hat_color=None):
    # color is the stick color, hat_color is the arrow_hat color

    B = int(origin_xyz.shape[0])
    assert origin_xyz.shape == (B, 3), origin_xyz.shape
    assert terminal_xyz.shape == (B, 3)
    assert color.shape == (B, 3)
    if hat_color is None:
        hat_color = color
    else:
        assert hat_color.shape == (B, 3)
    
    def normalize(x, epsilon=1.e-4):
        x_norm = np.linalg.norm(x, ord=2, axis=-1)[:, None].clip(min=epsilon, max=np.inf)
        return x / x_norm
    
    # Determine the 3 directions of the arrow (the pointing direction called arrow_z, and the other two, called arrow_x and arrow_y)
    arrow_z = normalize(terminal_xyz - origin_xyz)
    arrow_x = normalize(np.cross(np.array([3.3, 3.4, 3.5], dtype=np.float32)[None, :], arrow_z))
    arrow_y = normalize(np.cross(arrow_z, arrow_x))
    v0 = origin_xyz - 0.5 * rLine * arrow_x - 0.5 * rLine * arrow_y   
    v2 = origin_xyz - 0.5 * rLine * arrow_x + 0.5 * rLine * arrow_y
    v4 = origin_xyz + 0.5 * rLine * arrow_x - 0.5 * rLine * arrow_y
    v6 = origin_xyz + 0.5 * rLine * arrow_x + 0.5 * rLine * arrow_y
    v1 = terminal_xyz - 0.5 * rLine * arrow_x - 0.5 * rLine * arrow_y   
    v3 = terminal_xyz - 0.5 * rLine * arrow_x + 0.5 * rLine * arrow_y
    v5 = terminal_xyz + 0.5 * rLine * arrow_x - 0.5 * rLine * arrow_y
    v7 = terminal_xyz + 0.5 * rLine * arrow_x + 0.5 * rLine * arrow_y
    v_line = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], 1).reshape((B * 8, 3))
    f0 = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 5],
            [0, 5, 4],
            [0, 4, 2],
            [2, 4, 6],
            [4, 5, 7],
            [4, 7, 6],
            [2, 6, 3],
            [3, 6, 7],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    ff = np.arange(B, dtype=np.int32)[:, None, None] * 8 + f0
    f_line = ff.reshape((B * 12, 3))
    c_line = np.tile(color[:, None, :], (1, 8, 1)).reshape((B * 8, 3))
    if rArrow > 0:
        v0 = terminal_xyz - 0.5 * rArrow * arrow_x - 0.5 * rArrow * arrow_y
        v1 = terminal_xyz - 0.5 * rArrow * arrow_x + 0.5 * rArrow * arrow_y
        v2 = terminal_xyz + 0.5 * rArrow * arrow_x - 0.5 * rArrow * arrow_y
        v3 = terminal_xyz + 0.5 * rArrow * arrow_x + 0.5 * rArrow * arrow_y
        v4 = terminal_xyz + 1.0 * rArrow * arrow_z
        v_arrow = np.stack([v0, v1, v2, v3, v4], 1).reshape((B * 5, 3))
        f0 = np.array([
            [0, 4, 1],
            [0, 2, 4],
            [2, 3, 4],
            [1, 4, 3],
            [0, 3, 2],
            [0, 1, 3],
        ], dtype=np.int32)
        ff = np.arange(B, dtype=np.int32)[:, None, None] * 5 + f0
        f_arrow = ff.reshape((B * 6, 3))
        c_arrow = np.tile(hat_color[:, None, :], (1, 5, 1)).reshape((B * 5, 3))
        v, f, c = combineMultiShapes_withVertRgb([(v_line, f_line, c_line), (v_arrow, f_arrow, c_arrow)])
    else:
        v, f, c = v_line, f_line, c_line
    # We do not care about mesh inside-out at this point. Just switch to "double" mode in meshlab to view

    return v, f, c


def mainCheckArrow():
    v, f, c = create_lineArrow_mesh(0.01, 0.04, [1, 1, 1], [2, 3, 4], color=[1, 0, 0])
    import os

    from codes_py.toolbox_3D.mesh_io_v1 import dumpPly

    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../../"
    dumpPly(
        projRoot + "cache/debuggingCenter/dump_codes_py_mesh_surgery/lineArrow.ply",
        v,
        f,
        c,
    )


def combineMultiShapes_plain(vt0List):  # plain means just the vert and face
    nVertList = np.array([x[0].shape[0] for x in vt0List], dtype=np.int32)
    nVertCumsumList = np.cumsum(nVertList)
    nVertCumsumList = np.concatenate(
        [
            np.array([0], dtype=np.int32),
            nVertCumsumList,
        ],
        0,
    )

    # combine
    vert0 = np.concatenate([x[0] for x in vt0List], 0)
    face0 = np.concatenate(
        [vt0List[j][1] + nVertCumsumList[j] for j in range(len(vt0List))], 0
    )

    return vert0, face0


def combineMultiShapes_withVertRgb(vtc0List):
    # stats on vtc0List
    nVertList = np.array([x[0].shape[0] for x in vtc0List], dtype=np.int32)
    nVertCumsumList = np.cumsum(nVertList)
    nVertCumsumList = np.concatenate(
        [
            np.array([0], dtype=np.int32),
            nVertCumsumList,
        ],
        0,
    )

    # combine
    vert0 = np.concatenate([x[0] for x in vtc0List], 0)
    face0 = np.concatenate(
        [vtc0List[j][1] + nVertCumsumList[j] for j in range(len(vtc0List))], 0
    )
    color0 = np.concatenate([x[2] for x in vtc0List], 0)

    return vert0, face0, color0


def combineMultiShapes_withWhatever(vtwhatever0List):
    nVertList = np.array([x[0].shape[0] for x in vtwhatever0List], dtype=np.int32)
    nVertCumsumList = np.cumsum(nVertList)
    nVertCumsumList = np.concatenate(
        [
            np.array([0], dtype=np.int32),
            nVertCumsumList,
        ],
        0,
    )

    # combine
    vert0 = np.concatenate([x[0] for x in vtwhatever0List], 0)
    face0 = np.concatenate(
        [
            vtwhatever0List[j][1] + nVertCumsumList[j]
            for j in range(len(vtwhatever0List))
        ]
    )
    out = [vert0, face0]
    for j in range(2, len(vtwhatever0List[0])):
        out.append(np.concatenate([x[j] for x in vtwhatever0List], 0))
    return tuple(out)


def composeMultiShape(
    point, radius, color, shapeName, ifSolidRatherThanBubble=True, **kwargs
):
    assert len(point.shape) == 2 and point.shape[-1] == 3
    assert type(radius) is float
    assert len(color) == 3

    if shapeName == "sphere":
        v0, t0 = meshzoo_uv_sphere(
            num_points_per_circle=kwargs.get("num_points_per_circle", 12),
            num_circles=kwargs.get("num_circles", 6),
            radius=radius,
        )
        t0 = t0[:, [0, 2, 1]]
    elif shapeName == "cube":
        v0, t0 = create_cube_mesh(r=radius)
        t0 = t0[:, [0, 2, 1]]
    elif shapeName == "tetrahedron":
        v0, t0 = create_tetrahedron_mesh(r=radius)
    else:
        raise NotImplementedError("Unknown shapeName %s" % shapeName)

    if not ifSolidRatherThanBubble:
        t0 = t0[:, [0, 2, 1]]  # if it is a bubble, you need to make it inside out.
    v0 = v0.astype(np.float32)
    f0 = t0.astype(np.int32)

    numPoint = int(point.shape[0])
    v = np.tile(v0[None, :, :], (numPoint, 1, 1))
    v = v + point[:, None, :]
    f = np.tile(f0[None, :, :], (numPoint, 1, 1))
    f = f + (np.arange(numPoint, dtype=np.int32) * v0.shape[0])[:, None, None]
    v = v.astype(np.float32).reshape((-1, 3))
    f = f.astype(np.int32).reshape((-1, 3))
    c = np.zeros_like(v)
    for k in range(3):
        c[:, k] = color[k]
    return v, f, c


def addPointCloudToMesh(
    vfcInput, point0, pointColor, pointShapeName, pointRadius, ifSolidRatherThanBubble
):
    if point0.shape[0] == 0:
        return vfcInput
    vfcOutput = combineMultiShapes_withVertRgb(
        [
            composeMultiShape(
                point0, pointRadius, pointColor, pointShapeName, ifSolidRatherThanBubble
            ),
            vfcInput,
        ]
    )
    return vfcOutput


def trimVert(vert0, face0):
    # remove irrelevant vertex.
    # This typically happens when face0 is a small subset of all the faces
    u = np.sort(np.unique(face0))
    vertOut0 = vert0[u]
    faceOut0 = np.searchsorted(u, face0).astype(face0.dtype)
    return vertOut0, faceOut0


def trimVertWithRgb(vert0, face0, vertRgb0):
    u = np.sort(np.unique(face0))
    vertOut0 = vert0[u]
    faceOut0 = np.searchsorted(u, face0).astype(face0.dtype)
    vertRgb0 = vertRgb0[u]
    return vertOut0, faceOut0, vertRgb0


def trimVertGeneral(vert0, face0, otherVert0s):
    # Note there is no need for otherFaces as they do not change.
    u = np.sort(np.unique(face0))
    vertOut0 = vert0[u]
    faceOut0 = np.searchsorted(u, face0).astype(face0.dtype)
    if type(otherVert0s) is dict:
        otherVertOut0s = {k: otherVert0s[k][u] for k in otherVert0s.keys()}
    return vertOut0, faceOut0, otherVertOut0s


if __name__ == "__main__":
    mainCheckArrow()
