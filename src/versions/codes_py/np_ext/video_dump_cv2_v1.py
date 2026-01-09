
import cv2
import numpy as np


def dumpVideo(fn, cloth, message):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(fn, fourcc, 24.0, (cloth.shape[2], cloth.shape[1]))
    assert len(cloth.shape) == 4
    assert cloth.shape[3] == 3
    T = int(cloth.shape[0])
    for t in range(T):
        if message is not None:
            print("%s Writing t = %d / T = %d" % (message, t, T))
        videoWriter.write((cloth[t, :, :, ::-1] * 255).astype(np.uint8))
    videoWriter.release()
