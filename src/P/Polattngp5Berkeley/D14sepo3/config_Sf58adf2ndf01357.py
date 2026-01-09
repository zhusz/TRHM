from .config_Sf58adf2ndf import getConfigFunc as getConfigFuncParent

highlight_case_list = [0, 1, 3, 5, 7]  # No 3 now


def getConfigFunc(P, D, S, R, **kwargs):
    config = getConfigFuncParent(P, D, S, R, **kwargs)

    # All of the following are a copy of the original specs, just that now the "highlight_case_list"
    #   has been changed

    caseID = config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["caseID"]

    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["ifLoadDepth"] = False
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["loadDepthTag"] = None

    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["ifPreloadHighlight"] = (caseID % 8) in highlight_case_list
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["batchSizeHighlight"] = 128 if ((caseID % 8) in highlight_case_list) else 0
    config.datasetConfDict["renderingNerfBlender58Pointk8L112V100"]["batchSizeSublight"] = 128 * 7 if ((caseID % 8) in highlight_case_list) else 0

    config.if_highlight_case = (caseID in highlight_case_list)

    wl = {}
    wl["lossMain"] = 1.0
    wl["lossMask"] = 0.0
    wl["lossTie"] = 0.001 * 1.0
    wl["lossBackface"] = 0.01 * 1.0  # Now we use mean instead of sum
    wl["lossTie23"] = 0.001  # * (1.0 if (caseID % 8) in highlight_case_list else 100.0)
    s = (0.06 / 10 / 10) if ((caseID % 8) in highlight_case_list) else 0  # V2
    wl["lossMainHighlight"] = s * wl["lossMain"]
    wl["lossMaskHighlight"] = s * wl["lossMask"]
    wl["lossTieHighlight"] = s * wl["lossTie"]
    wl["lossBackfaceHighlight"] = s * wl["lossBackface"]
    wl["lossTie23Highlight"] = s * wl["lossTie23"]

    wl["lossHighlightNdotH"] = (0.1 * 2) if (caseID in highlight_case_list) else 0  # V2
    wl["lossSuppressHdr3"] = 0.01 / 20  # V2
    config.wl = wl
    config.sloss = s
    config.ifShrinkNdotHloss = False

    if caseID % 8 == 7:
        config.density_activation = "softplus"
    else:
        config.density_activation = "exp"

    if config.density_fix:
        config.wl["lossTie"] = 0
        config.wl["lossTieHighlight"] = 0

    return config
