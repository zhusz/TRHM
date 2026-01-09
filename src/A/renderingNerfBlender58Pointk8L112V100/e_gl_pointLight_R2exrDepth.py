import math
import os
import pickle

import bpy
import numpy as np
np.set_printoptions(suppress=True)


def rotMat02quat0(M0):
    assert M0.shape == (3, 3)
    assert M0.dtype == np.float32
    qw = ((1.0 + float(M0[0, 0]) + float(M0[1, 1]) + float(M0[2, 2])) ** 0.5) / 2.0
    assert qw > 0, M0
    factor = 0.25 / qw
    qx = (float(M0[2, 1]) - float(M0[1, 2])) * factor
    qy = (float(M0[0, 2]) - float(M0[2, 0])) * factor
    qz = (float(M0[1, 0]) - float(M0[0, 1])) * factor
    return (qw, qx, qy, qz)


def argmax4(a):
    if (a[0] >= a[1]) and (a[0] >= a[2]) and (a[0] >= a[3]):
        return 0
    elif (a[1] >= a[2]) and (a[1] >= a[3]):
        return 1
    elif (a[2] >= a[3]):
        return 2
    else:
        return 3


def rotMat02quat0New(M0):
    assert M0.shape == (3, 3)
    assert M0.dtype == np.float32
    # copied from scipy
    decision_w = float(M0[0, 0]) + float(M0[1, 1]) + float(M0[2, 2])
    decision_x = float(M0[0, 0])
    decision_y = float(M0[1, 1])
    decision_z = float(M0[2, 2])
    choice = argmax4((decision_x, decision_y, decision_z, decision_w))

    q_xyz = [None, None, None]
    q_w = None
    if choice != 3:
        i = choice
        j = (i + 1) % 3
        k = (j + 1) % 3
        q_xyz[i] = 1 - decision_w + 2 * float(M0[i, i])
        q_xyz[j] = float(M0[j, i]) + float(M0[i, j])
        q_xyz[k] = float(M0[k, i]) + float(M0[i, k])
        q_w = float(M0[k, j]) - float(M0[j, k])
    else:
        q_xyz[0] = float(M0[2, 1]) - float(M0[1, 2])
        q_xyz[1] = float(M0[0, 2]) - float(M0[2, 0])
        q_xyz[2] = float(M0[1, 0]) - float(M0[0, 1])
        q_w = 1 + decision_w

    norm = (q_xyz[0] ** 2 + q_xyz[1] ** 2 + q_xyz[2] ** 2 + q_w ** 2) ** 0.5
    qx = q_xyz[0] / norm
    qy = q_xyz[1] / norm
    qz = q_xyz[2] / norm
    qw = q_w / norm
    return (qw, qx, qy, qz)


def quat2matNP(quat):  # functional
    assert len(quat.shape) == 2 and quat.shape[1] == 4

    norm_quat = quat
    norm_quat = norm_quat / np.linalg.norm(norm_quat, ord=2, axis=1, keepdims=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.shape[0]

    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMats = np.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        1,
    ).reshape((B, 3, 3))
    return rotMats


def ELU02cam0(elu0):
    assert len(elu0) == 9
    ex = elu0[0]
    ey = elu0[1]
    ez = elu0[2]
    lx = elu0[3]
    ly = elu0[4]
    lz = elu0[5]
    ux = elu0[6]
    uy = elu0[7]
    uz = elu0[8]
    u = np.array([ux, uy, uz], dtype=np.float32)

    z = np.array([lx - ex, ly - ey, lz - ez], dtype=np.float32)
    z = z / np.linalg.norm(z)
    x = np.cross(z, u)
    y = np.cross(z, x)
    assert np.linalg.norm(x) > 0.0
    assert np.linalg.norm(y) > 0.0
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    M = np.array(
        [
            [x[0], x[1], x[2], -(x[0] * ex + x[1] * ey + x[2] * ez)],
            [y[0], y[1], y[2], -(y[0] * ex + y[1] * ey + y[2] * ez)],
            [z[0], z[1], z[2], -(z[0] * ex + z[1] * ey + z[2] * ez)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert M[-1, 0] == 0.0
    assert M[-1, 1] == 0.0
    assert M[-1, 2] == 0.0
    assert M[-1, 3] == 1.0
    return M  # which is cam0


def getRotationMatrixBatchNP(rotationAxis, degrees, **kwargs):  # functional
    assert len(rotationAxis.shape) == 2 and rotationAxis.shape[1] == 3
    assert len(degrees.shape) == 1 and rotationAxis.shape[0] == degrees.shape[0]

    norms = np.linalg.norm(rotationAxis, ord=2, axis=1)
    assert norms.min() > 0.0
    normalized = np.divide(rotationAxis, norms[:, None])

    halfRadian = degrees * 0.5 * np.pi / 180.0
    vCos = np.cos(halfRadian)
    vSin = np.sin(halfRadian)
    quat = np.concatenate([vCos[:, None], vSin[:, None] * normalized], 1)
    rotMats = quat2matNP(quat)
    return rotMats


class LgtDatasetCache(object):
    def __init__(self, projRoot):
        self.projRoot = projRoot
        self.cache = {}

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}
            with open(self.projRoot + "v/A/%s/A0_main.pkl" % dataset, "rb") as f:
                cache0["A0"] = pickle.load(f)
            # Add more if you wish for more meta info of thhis lighting dataset.
            self.cache[dataset] = cache0
        return self.cache[dataset]


class SceneDatasetCache(object):
    def __init__(self, projRoot):
        self.projRoot = projRoot
        self.cache = {}

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}
            with open(self.projRoot + "v/A/%s/A0_main.pkl" % dataset, "rb") as f:
                cache0["A0"] = pickle.load(f)
            # Add more if you wish for more meta info of thhis scene dataset.
            self.cache[dataset] = cache0
        return self.cache[dataset]



class CaptureDatasetCache(object):
    def __init__(self, projRoot):
        self.projRoot = projRoot
        self.cache = {}  # collecting A0b

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}
            with open(
                self.projRoot + "v/A/%s/A0b_precomputation.pkl" % dataset, "rb"
            ) as f:
                cache0["A0b"] = pickle.load(f)
            self.cache[dataset] = cache0
        return self.cache[dataset]


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

    with open(projRoot + "v/A/%s/A0_randomness.pkl" % dataset, "rb") as f:
        A0 = pickle.load(f)

    lgtDatasetCache = LgtDatasetCache(projRoot=projRoot)
    sceneDatasetCache = SceneDatasetCache(projRoot=projRoot)
    # captureDatasetCache = CaptureDatasetCache(projRoot=projRoot)

    # ============ set your set ============ #
    # only render a subset of the dataset
    # since this rendering is typically time consuming
    # flagChosen = A0["sceneIDList"] == 3
    flagChosen = np.ones((A0["m"],), dtype=bool)
    # flagChosen = np.zeros((A0["m"],), dtype=bool)
    # flagChosen[(A0["caseIDList"] == 8) & (A0["flagSplit"] == 2)] = True
    # print(flagChosen)
    # indChosen = np.where(flagChosen)[0]  # hotdog
    # print(indChosen)
    # ====================================== #

    # potentially parallel
    j1 = int(os.environ.get("J1", 0))
    j2 = int(os.environ.get("J2", int(A0["m"])))
    # only render j \in [j1, j2), and only if j \in indChosen
    # assert 0 <= j1 < j2 <= A0["m"], (0, j1, j2, A0["m"])
    assert 0 <= j1 < j2, (0, j1, j2, A0["m"])
    # You should allow j2 to be larger than m, since sometimes m is not divisible by (j2 - j1)

    count = 0
    tot = flagChosen[j1:j2].sum()
    currentScene = None
    currentLgt = None
    currentResolution = None
    currentFov = None
    for j in range(j1, j2):
        if flagChosen[j]:
            if count % 1 == 0:
                print(
                    "Rendering R2 for %s: j = %d (J1 = %d, J2 = %d), count = %d (tot = %d, progress = %.1f%%)"
                    % (
                        dataset,
                        j,
                        j1,
                        j2,
                        count,
                        tot,
                        float(count) / float(tot) * 100.0,
                    )
                )

            # R2exr
            outputFileName_2exr = projRoot + "v/R/%s/R2exr/%sCase%d/%s/r_%d.exr" % (
                dataset,
                dataset,
                A0["caseIDList"][j],
                ["train", "val", "test"][A0["flagSplit"][j] - 1],
                A0["caseSplitInsideIndexList"][j],
            )
            # R2exrDepth 
            outputFileName_2exrDepth = projRoot + "v/R/%s/R2exrDepth/%sCase%d/%s/r_%d.exr" % (
                dataset,
                dataset,
                A0["caseIDList"][j],
                ["train", "val", "test"][A0["flagSplit"][j] - 1],
                A0["caseSplitInsideIndexList"][j],
            )
            # R2exrNormal
            outputFileName_2exrNormal = projRoot + "v/R/%s/R2exrNormal/%sCase%d/%s/r_%d.exr" % (
                dataset,
                dataset,
                A0["caseIDList"][j],
                ["train", "val", "test"][A0["flagSplit"][j] - 1],
                A0["caseSplitInsideIndexList"][j],
            )
            # R2
            assert A0["nameList"][j] == "v/R/%s/R2/%sCase%d/%s/r_%d.png" % (
                dataset,
                dataset,
                A0["caseIDList"][j],
                ["train", "val", "test"][A0["flagSplit"][j] - 1],
                A0["caseSplitInsideIndexList"][j],
            )
            outputFileName = projRoot + A0["nameList"][j]
            # R2a
            outputFileName_2a = projRoot + "v/R/%s/R2a/%sCase%d/%s/r_%d_r2a.png" % (
                dataset,
                dataset,
                A0["caseIDList"][j],
                ["train", "val", "test"][A0["flagSplit"][j] - 1],
                A0["caseSplitInsideIndexList"][j],
            )
            # R2b  # Now all put to R2a for conveniency
            outputRootName_2b = projRoot + "v/R/%s/R2a/%sCase%d/%s/" % (
                dataset,
                dataset,
                A0["caseIDList"][j],
                ["train", "val", "test"][A0["flagSplit"][j] - 1],
            )
            outputFileName_2b = (
                outputRootName_2b + "r_%d_r2a.txt" % A0["caseSplitInsideIndexList"][j]
            )

            """
            flag = (os.path.isfile(outputFileName_2exr)
                and os.path.isfile(outputFileName)
                and os.path.isfile(outputFileName_2a)
                and os.path.isfile(outputFileName_2b)
            )
            if 1085 <= j <= 1090:
                print("%d: %s" % (j, flag))
                print(outputFileName_2exr)
                raise ValueError
            """

            if (
                os.path.isfile(outputFileName_2exr)
                # and os.path.isfile(outputFileName_2exrDepth)
                # and os.path.isfile(outputFileName_2exrNormal)
                # and os.path.isfile(outputFileName)
                # and os.path.isfile(outputFileName_2a)
                # and os.path.isfile(outputFileName_2b)
            ):
                count += 1
                continue

            flagToChange = False

            sceneDataset = A0["sceneDatasetList"][j]
            sceneID = int(A0["sceneIDList"][j])
            A0_scene = sceneDatasetCache.getCache(dataset=sceneDataset)["A0"]
            # sceneName = A0_scene["sceneNameRetrieveList"][sceneID]
            name = A0_scene["nameList"][sceneID]
            if flagToChange or currentScene != name:
                print("[Scene] Loading from %s" % (projRoot + name))
                bpy.ops.wm.open_mainfile(filepath=projRoot + name)
                bpy.context.scene.render.use_persistent_data = True
                if bpy.context.scene.world is None:
                    new_world = bpy.data.worlds.new("World")
                    bpy.context.scene.world = new_world
                bpy.context.scene.world.use_nodes = True
                bpy.context.scene.render.engine = "CYCLES"
                # global illumination
                bpy.context.scene.cycles.max_bounces = max(4, bpy.context.scene.cycles.max_bounces)
                bpy.context.scene.cycles.diffuse_bounces = max(4, bpy.context.scene.cycles.diffuse_bounces)
                bpy.context.scene.cycles.glossy_bounces = max(4, bpy.context.scene.cycles.glossy_bounces)
                bpy.context.scene.cycles.transmission_bounces = max(4, bpy.context.scene.cycles.transmission_bounces)
                bpy.context.scene.cycles.transparent_max_bounces = max(4, bpy.context.scene.cycles.transparent_max_bounces)
                bpy.context.scene.cycles.min_light_bounces = max(1, bpy.context.scene.cycles.min_light_bounces)
                bpy.context.scene.cycles.min_transparent_bounces = max(1, bpy.context.scene.cycles.min_transparent_bounces)
                bpy.context.scene.render.image_settings.color_mode = "RGBA"
                bpy.context.scene.render.film_transparent = True
                for obj in bpy.data.objects:
                    if (
                        ("Camera" in obj.name)
                        or ("Light" in obj.name)
                        or ("camera" in obj.name)
                        or ("light" in obj.name)
                    ):
                        bpy.data.objects.remove(obj)
                # Remove the backgrond lighting
                # This is critically important for point light to delete this manually
                # previously for envmap, this will be automatically overriden
                # so you do not need to manually delete this in that setting
                bpy.context.scene.world.node_tree.nodes.remove(
                    bpy.context.scene.world.node_tree.nodes["Background"]
                )
                # Add the point light
                #   Note: We never need to add new lights to the same scene - we only change their locations
                #   Since all the renderings are only coupled with a single point light.
                light_data = bpy.data.lights.new(name="theOnlyPointLight", type="POINT")
                light_data.energy = 10
                light_object = bpy.data.objects.new(
                    name="theOnlyPointLight", object_data=light_data
                )
                bpy.context.scene.collection.objects.link(light_object)
                # light_object.location = (1, 1, 1)
                # Change the location in this way: light_object.location = (X, Y, Z)
                # Add the camera
                camera = bpy.data.objects.new(
                    "Camera", bpy.data.cameras.new(name="Camera")
                )
                bpy.context.scene.collection.objects.link(camera)
                camera.data.lens_unit = "FOV"
                camera.rotation_mode = "XYZ"
                bpy.context.scene.camera = camera
                flagToChange = True
                currentScene = name

                # add depth
                bpy.context.scene.use_nodes = True
                tree = bpy.context.scene.node_tree
                links = tree.links
                # print(bpy.context.scene.view_layers["View Layer"].keys())
                # bpy.context.scene.view_layers["View Layer"].use_pass_normal = True
                # bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
                for k in ["View Layer", "RenderLayer"]:
                    if k in bpy.context.scene.view_layers.keys():
                        bpy.context.scene.view_layers[k].use_pass_normal = True
                # assert "Render Layers" in tree.nodes.keys()
                for n in tree.nodes:
                    # if n.name != "Render Layers":
                    tree.nodes.remove(n)
                # render_layers = tree.nodes["Render Layers"]
                render_layers = tree.nodes.new("CompositorNodeRLayers")
                render_layers.label = "Custom Outputs"
                render_layers.name = "Custom Outputs"
                depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
                depth_file_output.label = "Depth Output"
                depth_file_output.name = "Depth Output"
                links.new(
                    render_layers.outputs["Depth"],
                    depth_file_output.inputs[0],
                )
                normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
                normal_file_output.label = "Normal Output"
                normal_file_output.name = "Normal Output"
                links.new(
                    render_layers.outputs["Normal"],
                    normal_file_output.inputs[0],
                )
                del tree, links, render_layers, depth_file_output
            del name

            # lgtDataset = A0["lgtDatasetList"][j]
            # lgtID = int(A0["lgtIDList"][j])
            # A0_lgt = lgtDatasetCache.getCache(dataset=lgtDataset)["A0"]
            # name = A0_lgt["nameList"][lgtID]
            # lgtStrength = 1.0  # float(A0["lgtStrengthList"][j])
            # if flagToChange or currentLgt != (name, lgtStrength):
            #     print("[Lgt] Loading from %s" % (projRoot + name))
            #     enode = bpy.context.scene.world.node_tree.nodes.new(
            #         "ShaderNodeTexEnvironment"
            #     )
            #     enode.image = bpy.data.images.load(projRoot + name)
            #     bpy.context.scene.world.node_tree.links.new(
            #         enode.outputs["Color"],
            #         bpy.context.scene.world.node_tree.nodes["Background"].inputs[
            #             "Color"
            #         ],
            #     )
            #     assert (
            #         lgtStrength == 1.0
            #     )  # currently we ban this value to be different than 1.0
            #     bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            #         1
            #     ].default_value = lgtStrength
            #     # Currently if you want to use a different bounce number, you can only
            #     # create a new blend file. In the future we might extend A0.keys() to accomendate this.
            #     # bpy.context.scene.cycles.diffuse_bounces = 12
            #     # bpy.context.scene.cycles.max_bounces = 12
            #     flagToChange = True
            #     currentLgt = (name, lgtStrength)
            # del name
            # lighting will be determined by mimickingSceneDatasetList, mimickingSceneIDList, mimickingGroupIDList, mimickingOlatIDList
            # Everything becomes point light now
            # We only retrieve the location of the point light, and then set to light_object.location
            lgtE = (float(A0["lgtE"][j, 0]), float(A0["lgtE"][j, 1]), float(A0["lgtE"][j, 2]))
            lgtEnergy = float(A0["lgtEnergy"][j])
            if flagToChange or currentLgt != (
                lgtE,
                lgtEnergy,
            ):
                # fetch the point light location
                specified_light_location = A0["lgtE"][j]
                # specified_light_location = captureDatasetCache.getCache(
                #     mimickingSceneDataset
                # )["A0b"][mimickingSceneID]["leaf_lgt_collectors"][mimickingGroupID][
                #     mimickingOlatID
                # ][
                #     "EWorld0"
                # ]
                # set the point light location
                light_object.location = (
                    float(specified_light_location[0]),
                    float(specified_light_location[1]),
                    float(specified_light_location[2]),
                )
                light_object.data.energy = lgtEnergy
                # set currentLgt (for potentially saving time) but almost sure this is not going to save any time...
                currentLgt = (
                    (float(A0["lgtE"][j, 0]), float(A0["lgtE"][j, 1]), float(A0["lgtE"][j, 2])),
                    lgtEnergy,
                )

            winWidth0 = int(A0["winWidth"][j])
            winHeight0 = int(A0["winHeight"][j])
            if flagToChange or currentResolution != (winWidth0, winHeight0):
                bpy.context.scene.render.resolution_x = winWidth0
                bpy.context.scene.render.resolution_y = winHeight0
                flagToChange = True
                currentResolution = (winWidth0, winHeight0)
            fovRad0 = float(A0["fovRad"][j])
            if flagToChange or currentFov != fovRad0:
                camera.data.angle = fovRad0
                flagToChange = True
                currentFov = fovRad0

            # Now do camera extrinsic pose and rendering
            E0 = A0["E"][j]
            L0 = A0["L"][j]
            U0 = A0["U"][j]
            q1 = rotMat02quat0New(
                np.linalg.inv(ELU02cam0(np.concatenate([E0, L0, U0], 0))[:3, :3])
            )
            camera.rotation_mode = "QUATERNION"
            camera.location[0] = E0[0]
            camera.location[1] = E0[1]
            camera.location[2] = E0[2]
            camera.rotation_quaternion[0] = -q1[1]
            camera.rotation_quaternion[1] = q1[0]
            camera.rotation_quaternion[2] = q1[3]
            camera.rotation_quaternion[3] = -q1[2]
            # LE0 = L0 - E0
            # LE0_normed = LE0 / float(max(np.linalg.norm(LE0, ord=2), 1.0e-4))
            # U0_normed = U0 / float(max(np.linalg.norm(U0, ord=2), 1.0e-4))
            # theta0_projection = np.dot(LE0_normed, U0_normed)
            # theta0 = -float(
            #     math.asin(theta0_projection)
            # )  # top is +np.pi / 2, bottom is -np.pi / 2
            # # Rotate LE0 according to the same rotation of rotating U0 to [0, 0, 1]
            # rotatingAngle0 = np.array(math.acos(U0[2]), dtype=np.float32)
            # if rotatingAngle0 == 0:
            #     LE0_backrotated_normed = LE0.copy()
            #     phi0_tmp = np.cross(
            #         LE0_backrotated_normed, np.array([0, 0, 1], dtype=np.float32)
            #     )
            #     phi0 = (
            #         math.atan2(phi0_tmp[1], phi0_tmp[0]) + np.pi / 2.0
            #     )  # right-ward is 0-degree, simily to the x-y system
            # else:
            #     rotatingAxis0 = np.cross(U0, np.array([0, 0, 1], dtype=np.float32))
            #     rotatingMat0 = getRotationMatrixBatchNP(
            #         rotatingAxis0[None], rotatingAngle0[None]
            #     )[0]
            #     phi0 = None
            #     raise NotImplementedError
            # camera.location[0] = E0[0]
            # camera.location[1] = E0[1]
            # camera.location[2] = E0[2]
            # camera.rotation_euler[0] = np.pi / 2.0 - theta0
            # camera.rotation_euler[1] = 0.0
            # camera.rotation_euler[2] = -np.pi / 2.0 + phi0

            # R2exr
            bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
            bpy.context.scene.render.filepath = outputFileName_2exr
            bpy.context.scene.node_tree.nodes["Depth Output"].base_path = os.path.dirname(outputFileName_2exrDepth)
            bpy.context.scene.node_tree.nodes["Depth Output"].file_slots[0].path = os.path.basename(outputFileName_2exrDepth)
            bpy.context.scene.node_tree.nodes["Depth Output"].format.file_format = "OPEN_EXR"
            bpy.context.scene.node_tree.nodes["Normal Output"].base_path = os.path.dirname(outputFileName_2exrNormal)
            bpy.context.scene.node_tree.nodes["Normal Output"].file_slots[0].path = os.path.basename(outputFileName_2exrNormal)
            bpy.context.scene.node_tree.nodes["Normal Output"].format.file_format = "OPEN_EXR"
            bpy.ops.render.render(write_still=True)
            # bpy.context.scene.render.image_settings.file_format = "PNG"

            # # R2
            # bpy.context.scene.render.filepath = outputFileName
            # bpy.context.scene.render.film_transparent = True
            # bpy.ops.render.render(write_still=True)

            # # R2a
            # bpy.context.scene.render.film_transparent = False
            # bpy.context.scene.render.filepath = outputFileName_2a
            # bpy.ops.render.render(write_still=True)
            # bpy.context.scene.render.film_transparent = True

            # # R2b
            # os.makedirs(outputRootName_2b, exist_ok=True)
            # with open(outputFileName_2b, "w") as f:
            #     f.write("sceneDataset: %s\n" % A0["sceneDatasetList"][j])
            #     f.write("sceneID: %d\n" % A0["sceneIDList"][j])
            #     f.write("lgtDataset: %s\n" % A0["lgtDatasetList"][j])
            #     f.write("lgtID: %d\n" % A0["lgtIDList"][j])
            #     f.write("lgtStrength: %.3f\n" % A0["lgtStrengthList"][j])
            #     f.write("flagSplit: %d\n" % A0["flagSplit"][j])
            #     f.write("caseID: %d\n" % A0["caseIDList"][j])
            #     f.write(
            #         "caseSplitInsideIndex: %d\n" % A0["caseSplitInsideIndexList"][j]
            #     )
            #     f.write("index: %d\n" % j)
            #     f.write("outputFileName_R2: %s\n" % outputFileName)
            #     f.write("outputFileName_R2a: %s\n" % outputFileName_2a)
            #     f.write("outputFileName_R2b: %s\n" % outputFileName_2b)

            count += 1


main()
