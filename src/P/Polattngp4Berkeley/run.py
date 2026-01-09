import os

import sys

projRoot = os.path.dirname(os.path.realpath(__file__)) + "/" + "../../../"
sys.path.append(projRoot + "src/versions/")
sys.path.append(projRoot + "src/B/")
sys.path.append(projRoot + "src/P/")

from configs_registration import getConfigGlobal


def main():
    fullPathSplit = os.path.dirname(os.path.realpath(__file__)).split("/")
    P = fullPathSplit[-1]
    D = os.environ["D"]
    S = os.environ["S"]
    R = os.environ["R"]
    getConfigFunc = getConfigGlobal(P, D, S, R)["getConfigFunc"]
    config = getConfigFunc(P, D, S, R)

    DTrainer = getConfigGlobal(P, D, S, R)["exportedClasses"]["DTrainer"]

    if (
        ("RANK" in os.environ)
        and ("WORLD_SIZE" in os.environ)
        and (int(os.environ["WORLD_SIZE"]) > 1)
    ):
        rank = int(os.environ["RANK"])
        numMpProcess = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        numMpProcess = 0

    print("Rank: %d, WORLD_SIZE: %d" % (rank, numMpProcess))
    # return

    trainer = DTrainer(
        config,
        rank=rank,
        numMpProcess=numMpProcess,
        ifDumpLogging=True,
    )

    if "I" in list(os.environ.keys()):
        iter_label = int(os.environ["I"][1:])
    else:
        iter_label = None

    trainer.initializeAll(
        iter_label=iter_label,
        hook_type=None,
        ifLoadToCuda=True,
        if_need_metaDataLoading=True,
    )
    trainer.train()

if __name__ == "__main__":
    main()
