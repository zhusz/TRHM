import numpy as np


def get_ray_directions_croppedWindowOnly(H, W, fx, fy, cx, cy, cropCenterX, cropCenterY, cropWidthHalf, cropHeightHalf):
    # H, W, fx, fy, cx, cy is the same as the function get_ray_directions - regarding the full map
    # the newly introduced crop**** variables are describing the cropped window 
    #   - as we do not need to work on the whole map (which wastes memory)
    #   And actually computing the fullmap's directions are just not useful at all, as they are not on the right pixel integer grid point locations of the cropped window
    #   And H W seems to be useless, but let us just input them here
    j, i = np.meshgrid(
        np.linspace(cropCenterX - cropWidthHalf, cropCenterX + cropWidthHalf, 2 * cropWidthHalf),
        np.linspace(cropCenterY - cropHeightHalf, cropCenterY + cropHeightHalf, 2 * cropHeightHalf),
    )
    directions = np.stack([(j - cx) / fx, (i - cy) / fy, np.ones_like(i)], -1)
    return directions.astype(np.float32)


# useful when computing "pixel_area" requested by nerfstudio
def get_ray_directions_with_shift(H, W, fx, fy, cx, cy, shift_x, shift_y):
    j, i = np.meshgrid(
        np.linspace(shift_x, W + shift_x - 1, W),
        np.linspace(shift_y, H + shift_y - 1, H),
    )   
    directions = np.stack([(j - cx) / fx, (i - cy) / fy, np.ones_like(i)], -1)
    return directions.astype(np.float32)


def get_ray_directions(H, W, fx, fy, cx, cy):
    """
    Get ray directions for all pixels in camera coordinate.

    In:
    * H: image height
    * W: image width
    * fx, fy: focal lengths
    * cx, cy: principal point

    Out:
    * directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    j, i = np.meshgrid(
        np.linspace(0, W - 1, W),
        np.linspace(0, H - 1, H),
    )
    # j = j.transpose()
    # i = i.transpose()

    # We generate directions using OpenCV notation.
    directions = np.stack([(j - cx) / fx, (i - cy) / fy, (np.ones_like(i))], -1)

    return directions.astype(np.float32)


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate.
    rays_d = directions @ c2w[:, :3].transpose()  # (H, W, 3)
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # The origin of all rays is the camera origin in world coordinate.
    # rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_o = np.tile(c2w[:, 3][None, None, :], (rays_d.shape[0], rays_d.shape[1], 1))

    rays_d = rays_d.reshape((-1, 3))
    rays_o = rays_o.reshape((-1, 3))

    return rays_o, rays_d


def readjustE(E, d, objCentroid0, readjustERadius):
    C0 = objCentroid0  # notational convenience

    # asserts
    m = int(E.shape[0])
    assert E.shape == (m, 3), E.shape
    assert E.dtype == np.float32, E.dtype
    assert d.shape == (m, 3), d.shape
    assert d.dtype == np.float32, d.dtype
    d_norm = np.linalg.norm(d, ord=2, axis=1)
    assert d_norm.min() >= 1.0 - 1.0e-4, d_norm.min()
    assert d_norm.max() <= 1.0 + 1.0e-4, d_norm.max()
    assert C0.shape == (3,), C0.shape
    assert C0.dtype == np.float32, C0.dtype
    assert type(readjustERadius) is float

    k = np.linalg.norm(C0[None, :] - E, ord=2, axis=1)  # the length of EC
    assert k.min() > 0  # make sure E and C are at two different locations
    kcos = ((C0[None, :] - E) * d).sum(1)  # the projection length of EC onto Ed.
    P = E + kcos[:, None] * d  # The projection point of C onto Ed
    if np.any(kcos > k):  # numerical problems
        ind_bad = np.where(kcos > k)[0]
        assert np.all(kcos[ind_bad] - k[ind_bad] < 0.0001)
        kcos[ind_bad] = k[ind_bad]
    assert np.all(kcos <= k), (kcos[kcos > k], k[kcos > k])
    ksin = (k ** 2 - kcos ** 2) ** 0.5  # the point-to-line distance between C and Ed

    t = np.nan * np.zeros((m,), dtype=np.float32)  # length of PQ. (Q is the output: the readjusted new E)
    # (Note, Q and E should be at the same side of P on the d direction)
    ind = np.where(readjustERadius > ksin)[0]  # Only these rays are meaningful - the rays are not too far from C
    t[ind] = (readjustERadius ** 2 - ksin[ind] ** 2) ** 0.5
    Q = P - t[:, None] * d

    return Q, ksin  # Q: the new E, ksin: the point-to-line distance between objCentroid0 and the ray
    # Note Q might contain lots of nans, while ksin would be all finite
