from .config_SgCap7V4 import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict["capture7jf"]["debugReadTwo"] = True
    config.datasetConfDict["capture7jf"]["debugReadHowMany"] = 2

    return config
