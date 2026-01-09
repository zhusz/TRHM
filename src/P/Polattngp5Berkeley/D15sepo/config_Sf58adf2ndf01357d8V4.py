from .config_Sf58adf2ndf01357V4 import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.wl["lossSuppressHdr3"] = 0.01 / 10

    return config
