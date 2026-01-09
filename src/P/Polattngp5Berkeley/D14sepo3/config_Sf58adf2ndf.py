from .config_Sf58adf2n import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.if_normal3_tied_to_normal2 = False

    caseID = config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["caseID"]
    # config.density_fix = caseID in [0, 1, 2, 3, 4, 5, 6]
    config.density_fix = caseID in [0, 1, 2, 4, 5, 6]

    if config.density_fix:
        config.wl["lossTie"] = 0
        config.wl["lossTieHighlight"] = 0

    return config
