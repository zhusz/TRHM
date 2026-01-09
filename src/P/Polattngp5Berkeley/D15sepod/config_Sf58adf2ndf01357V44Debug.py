from .config_Sf58adf2ndf01357V44 import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["debugReadTwo"] = True
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["debugReadHowMany"] = 2

    return config
