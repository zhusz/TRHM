from .config_Sf58adf2 import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.if_normal3_tied_to_normal2 = False

    return config
