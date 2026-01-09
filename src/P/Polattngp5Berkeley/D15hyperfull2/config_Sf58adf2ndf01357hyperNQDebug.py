from .config_Sf58adf2ndf01357hyperNQ import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict["renderingNerfBlender58EnvolatGray16x32k8distillV100"]["debugReadTwo"] = True
    config.datasetConfDict["renderingNerfBlender58EnvolatGray16x32k8distillV100"]["debugReadHowMany"] = 1

    return config
