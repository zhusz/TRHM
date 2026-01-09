from collections import OrderedDict
from .config_SgCap7V4 import getConfigFunc as getConfigFuncParent
from .config_SgCap7V4 import get_trailing_number, originalBoundTable


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict["capture7jf"]["debugReadTwo"] = True
    config.datasetConfDict["capture7jf"]["debugReadTwoHowMany"] = 1

    for (dataset, datasetConf) in config.datasetConfDict.items():
        assert dataset == datasetConf["dataset"]

    return config
