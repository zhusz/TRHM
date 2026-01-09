import torch


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
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W),
        torch.linspace(0, H - 1, H),
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    # We generate directions using OpenCV notation.
    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, (torch.ones_like(i))], -1)

    return directions


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
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

    # The origin of all rays is the camera origin in world coordinate.
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d
