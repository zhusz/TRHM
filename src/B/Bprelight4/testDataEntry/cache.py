import pickle
import os
import cv2
import numpy as np


class LgtDatasetCacheUnio(object):
    def __init__(self, unioRoot, uconf):
        self.unioRoot = unioRoot
        self.uconf = uconf
        self.cache = {}

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}
            A0_path = self.unioRoot + "v/A/%s/A0_main.pkl" % dataset
            if unio.isfile(A0_path, self.uconf):
                with unio.open(A0_path, self.uconf, "rb") as f:
                    cache0["A0"] = pickle.load(f)
            else:
                cache0["A0"] = A0_path
            # Add more if you wish for more meta info of thhis lighting dataset.
            self.cache[dataset] = cache0
        return self.cache[dataset]


class SceneDatasetCacheUnio(object):
    def __init__(self, unioRoot, uconf, **kwargs):
        self.unioRoot = unioRoot
        self.uconf = uconf
        self.cache = {}

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}

            with unio.open(
                self.unioRoot + "v/A/%s/A0_main.pkl" % dataset, self.uconf, "rb"
            ) as f:
                cache0["A0"] = pickle.load(f)

            self.cache[dataset] = cache0
        return self.cache[dataset]


class LgtDatasetCache(object):
    def __init__(self, projRoot):
        self.projRoot = projRoot
        self.cache = {}
        self.cacheR = {}
        self.cacheExr = {}  # actually it could be merged into self.cacheR via self.cacheR["exr"] but we keep it single out

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}
            A0_path = self.projRoot + "v/A/%s/A0_main.pkl" % dataset
            if os.path.isfile(A0_path):
                with open(A0_path, "rb") as f:
                    cache0["A0"] = pickle.load(f)
            else:
                raise ValueError("The path does not exist: %s" % A0_path)
                cache0["A0"] = A0_path
            self.cache[dataset] = cache0
        return self.cache[dataset]

    def getR(self, dataset, index):  # getCache is mainly for cache-loading header files (A), while getR is mainly for cache-loading per-sample files (R)
        raise NotImplementedError

    def getExr(self, dataset, index):  # a specialized class method
        A0 = self.getCache(dataset)["A0"]
        if (dataset not in self.cacheExr.keys()):
            self.cacheExr[dataset] = {}
        if (index not in self.cacheExr[dataset].keys()):
            fn = self.projRoot + A0["nameList"][index]
            assert fn.endswith(".exr")
            assert os.path.isfile(fn), fn
            hdr = cv2.imread(fn, flags=cv2.IMREAD_UNCHANGED)
            assert hdr.dtype == np.float32, hdr.dtype
            assert len(hdr.shape) == 3, hdr.shape
            assert hdr.shape[2] == 3, hdr.shape
            hdr = np.stack([hdr[:, :, 2], hdr[:, :, 1], hdr[:, :, 0]], 2)
            self.cacheExr[dataset][index] = hdr
        return self.cacheExr[dataset][index]


class SceneDatasetCache(object):
    def __init__(self, projRoot, **kwargs):
        self.projRoot = projRoot
        self.cache = {}

    def getCache(self, dataset):
        if dataset not in self.cache.keys():
            cache0 = {}

            with open(
                self.projRoot + "v/A/%s/A0_main.pkl" % dataset, "rb"
            ) as f:
                cache0["A0"] = pickle.load(f)

            self.cache[dataset] = cache0
        return self.cache[dataset]
