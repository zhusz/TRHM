import argparse
import os
import random
import sys
import typing

import numpy as np

import torch

projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
sys.path.append(projRoot + "src/versions/")
sys.path.append(projRoot + "src/B/")
sys.path.append(projRoot + "src/P/")

from .approachEntryPool import getApproachEntry


def main():
    # set your set - testDataNickName
    testDataNickName = os.environ["DS"]

    # random seed and precision
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # specify
    approachEntry = getApproachEntry(os.environ["MC"])
    if approachEntry["scriptTag"] == "1Eldrf":
        from .csvPsnrEntry.Bprelight_1Eldrf_fineLdr import testApproachOnTestData
    elif approachEntry["scriptTag"] == "1Eldrfenvmap":
        from .csvPsnrEntry.Bprelight_1Eldrfenvmap_fineLdrenvmap import testApproachOnTestData
    else:
        raise NotImplementedError(
            "Unknown approach script tag: %s" % approachEntry["scriptTag"]
        )
    testApproachOnTestData(testDataNickName, approachEntry)


if __name__ == "__main__":
    main()
