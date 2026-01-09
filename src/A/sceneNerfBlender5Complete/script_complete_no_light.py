import math
import os
import pickle

import bpy
import numpy as np


def getAllMaterialDict(allMaterialDict, k_material, material):
    # for k_material, material in materialsOrGroup.items():
    allMaterialDict[k_material] = material
    # validContains = ["NodeGroup", "Group", "Colors", "Missing Datablock"]
    for k_shader, shader in material.node_tree.nodes.items():
        if hasattr(shader, "node_tree"):
            # assert any([(x in k_shader) for x in validContains]), k_shader
            getAllMaterialDict(allMaterialDict, k_material + "-" + k_shader, shader)


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"

    sceneNameList = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]
    m = len(sceneNameList)

    # 0-chair (no emission - no need to change at all) and 7-ship (do the change on your own) contains
    #   texture loads, and it is better if we keep the original form
    #   It is not clear why it loses the link
    for j in [1, 2, 3, 4, 5, 6]:
        bpy.ops.wm.open_mainfile(
            filepath=projRoot
            + "remote_fastdata/blend_files_complete_contains_emitter/%s.blend"
            % sceneNameList[j]
        )

        sceneName = sceneNameList[j]

        for k_obj, obj in bpy.data.objects.items():
            # print(k_obj)
            if (sceneName == "lego") and (k_obj in ["Plane", "Plane.001"]):
                bpy.data.objects.remove(obj)
            if (sceneName == "mic") and (k_obj.startswith("Area.00")):
                bpy.data.objects.remove(obj)
            if (sceneName == "mic") and (k_obj == "Lamp"):
                bpy.data.objects.remove(obj)
                # print("---- %s removed ----" % k_obj)

        # Extract whatever inside NodeGroup and also put to allMaterialDict.
        allMaterialDict = {}
        for k_material, material in bpy.data.materials.items():
            getAllMaterialDict(allMaterialDict, k_material, material)

        print(bpy.data.materials.keys())
        print(len(bpy.data.materials.keys()))
        print(allMaterialDict.keys())
        print(len(allMaterialDict.keys()))

        for k_material, material in allMaterialDict.items():
            for k_shader, shader in material.node_tree.nodes.items():
                if k_shader == "Emission":
                    print(shader.inputs["Strength"].default_value)
                    shader.inputs["Strength"].default_value = 0.0
                    print(shader.inputs["Strength"].default_value)
                    print((k_shader, k_material))

        bpy.ops.wm.save_as_mainfile(
            filepath=projRoot
            + "remote_fastdata/blend_files_complete/%s.blend" % sceneNameList[j]
        )


main()
