from .config_Sf58a import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    caseID = config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["caseID"]
    config.needDensityFixer = (caseID % 8 == 7)  # only the ship
    """
    config.fireDensityFixerCondition = (
        lambda xyz: ((xyz[..., 0] ** 2 + xyz[..., 1] ** 2) > 0.9 ** 2) | (xyz[..., 2] > -0.1)  # the ship
    ) if (caseID % 8 == 7) else (
        None
    )
    """

    # config.density_fix = (caseID % 8) in [0, 1, 2, 3, 4, 6]
    # This should be kept to be False

    return config
