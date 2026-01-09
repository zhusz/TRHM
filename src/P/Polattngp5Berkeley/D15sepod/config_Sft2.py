# (tngp2)
# Sft2 means hdr2 are tightened
from .config_Sf58adf2ndf01357V44 import getConfigFunc as getConfigFuncParent
from Bprelight4.testDataEntry.testDataEntryPool import nerfSynthetic_hdrScalingList


hdr_clipped_clip_table = {
    0: nerfSynthetic_hdrScalingList[0] * 1.0,  # 2.0,
    1: nerfSynthetic_hdrScalingList[1] * 1.0,  # 4.0 / 2,
    2: nerfSynthetic_hdrScalingList[2] * 4.0 / 2,
    3: nerfSynthetic_hdrScalingList[3] * 1.0,  # 2.0,
    4: nerfSynthetic_hdrScalingList[4] * 4.0 / 2,
    5: nerfSynthetic_hdrScalingList[5] * 1.0,  # 10.0,
    6: nerfSynthetic_hdrScalingList[6] * 4.0 / 2,
    7: nerfSynthetic_hdrScalingList[7] * 1.0,
}


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    caseID = config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["caseID"]
    config.hdr_clipped_clip = hdr_clipped_clip_table[caseID % 8]

    return config
