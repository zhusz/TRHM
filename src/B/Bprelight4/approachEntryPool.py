from codes_py.toolbox_framework.framework_util_v4 import splitPDSRI


def getApproachEntryDict():
    approachEntryList = [
        # (approachNickName, methodologyName (PDSRI), scriptTag, approachShownName)
    ]
    # From now on, any non-external-models (PDSRI) will be automatically generated, as long as:
    # (1) It is not presented in the approachEntryList above
    # (2) It follows the format of PDSRI(realNumber,no"W";if"W",itWillAppend"0080")OnTag(e.g.1Eldr)
    # Then it would be automatically generated as one item from above
    # This can greatly accelerate the research progress
    # Call the getApproachEntry function below to achieve this

    approachEntryList += [
        ("ironNb58R%dOn1G4plain" % r, "ironNb58R%d" % r, "1G4plain", None)
        for r in list(range(8)) + list(range(16, 24))
    ]

    approachEntryList += [
        ("inverseTranslucentNb58R%dOn1G5plain" % r, "inverseTranslucentNb58R%d" % r, "1G5plain", None)
        for r in list(range(8)) + list(range(16, 24))
    ]

    approachEntryList += [
        ("ironCap7R%dOn1G4plain" % r, "ironCap7R%d" % r, "1G4plain", None)
        for r in [1, 2, 3, 5, 6, 7, 8, 10]
    ]

    approachEntryList += [
        ("inverseTranslucentCap7R%dOn1G5plain" % r, "inverseTranslucentCap7R%d" % r, "1G5plain", None)
        for r in [1, 2, 3, 5, 6, 7, 8, 10]
    ]

    approachEntryDict = {
        approachEntry[0]: {
            "approachNickName": approachEntry[0],
            "methodologyName": approachEntry[1],
            "scriptTag": approachEntry[2],
            "approachShownName": approachEntry[3]
            if approachEntry[3]
            else approachEntry[0],
        }
        for approachEntry in approachEntryList
    }

    return approachEntryDict


def getApproachEntry(approachNickName):
    approachEntryDict = getApproachEntryDict()
    if approachNickName not in approachEntryDict.keys():  # Then it should follow the PDSRIOn format
        for k in ["P", "D", "S", "R", "I", "On"]:
            assert k in approachNickName, k
        index = approachNickName.rfind("On")
        P, D, S, R, I = splitPDSRI(approachNickName[:index])
        if I.endswith("W"):
            I = I[:-1] + "0080"
        scriptTag = approachNickName[(index + 2):]
        approachEntry = {
            "approachNickName": approachNickName,
            "methodologyName": P + D + S + R + I,
            "scriptTag": scriptTag,
            "approachShownName": approachNickName,
        }
    else:
        approachEntry = approachEntryDict[approachNickName]
    return approachEntry
        