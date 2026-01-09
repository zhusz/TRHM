from .config_S58Pointk8L112V100bszh import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["loadDepthTag"] = "R2neurecondepth"

    return config
