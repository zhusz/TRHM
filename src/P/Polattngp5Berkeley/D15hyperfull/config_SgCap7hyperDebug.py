from .config_SgCap7hyper import getConfigFunc as getConfigFuncParent


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    config.datasetConfDict["capture7jfLfhyper"]["batchSizeBg"] = 2
    config.datasetConfDict["capture7jfLfhyper"]["batchSizeMg"] = 2
    config.datasetConfDict["capture7jfLfhyper"]["mbtot"] = 698702
    config.datasetConfDict["capture7jfLfhyper"]["debugReadTwo"] = True
    config.datasetConfDict["capture7jfLfhyper"]["debugReadTwoHowMany"] = 5

    config.memory_bank_read_out_freq = 0

    return config
