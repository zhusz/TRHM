import os


class LoggerDummy(object):
    def __init__(self):
        pass

    def info(self, text):
        pass

    def close(self):
        pass


class LoggerManifold(object):
    def __init__(self, pathdir_manifold, uconf, **kwargs):
        self.update_count = int(kwargs["update_count"])
        self.refresh_count = int(kwargs["refresh_count"])

        self.pathdir_manifold = pathdir_manifold
        self.uconf = uconf
        assert unio.isdir(pathdir_manifold, uconf)
        self.fileObj = None
        self.count = None
        self.currentID = 0
        self.refresh()

    def refresh(self):
        if self.fileObj:
            self.close()

        fns = unio.ls(self.pathdir_manifold, self.uconf)
        fns = [fn.split("/")[-1] for fn in fns]
        if len(fns) == 0:
            nextID = 0
        else:
            ids = []
            for fn in fns:
                assert fn.startswith("trainingLog")
                assert fn.endswith(".txt")
                ids.append(int(fn[len("trainingLog_") : -len(".txt")]))
            ids = sorted(ids)
            for i, id in enumerate(ids):
                assert id == i
            nextID = len(ids)

        self.fileObj = unio.open(
            self.pathdir_manifold + "trainingLog_%06d.txt" % nextID, self.uconf, "a"
        )
        self.count = 0
        self.currentID = nextID

    def update(self):
        if self.fileObj:
            self.close()

        self.fileObj = unio.open(
            self.pathdir_manifold + "trainingLog_%06d.txt" % self.currentID,
            self.uconf,
            "a",
        )

    def info(self, text, add=1):
        if (self.count > 0) and (
            (self.count % self.update_count == 0)
            or (
                (self.count < self.update_count)
                and (self.count % int(self.update_count / 10) == 0)
            )
        ):
            self.update()
        if self.count >= self.refresh_count:
            self.refresh()

        self.fileObj.write(text + "\n")
        self.count += add
        print(text)

    def close(self):
        self.fileObj.close()


class Logger(object):
    def __init__(self, pathdir, **kwargs):
        self.update_count = int(kwargs["update_count"])
        self.refresh_count = int(kwargs["refresh_count"])

        self.pathdir = pathdir
        assert os.path.isdir(pathdir)
        self.fileObj = None
        self.count = None
        self.currentID = 0
        self.refresh()

    def refresh(self):
        if self.fileObj:
            self.close()

        fns = sorted(os.listdir(self.pathdir))
        fns = [fn.split("/")[-1] for fn in fns]
        if len(fns) == 0:
            nextID = 0
        else:
            ids = []
            for fn in fns:
                assert fn.startswith("trainingLog"), fn
                assert fn.endswith(".txt")
                ids.append(int(fn[len("trainingLog_") : -len(".txt")]))
            ids = sorted(ids)
            for i, id in enumerate(ids):
                assert id == i
            nextID = len(ids)

        self.fileObj = open(
            self.pathdir + "trainingLog_%06d.txt" % nextID, "a"
        )
        self.count = 0
        self.currentID = nextID

    def update(self):
        if self.fileObj:
            self.close()

        self.fileObj = open(
            self.pathdir + "trainingLog_%06d.txt" % self.currentID,
            "a",
        )

    def info(self, text, add=1):
        if (self.count > 0) and (
            (self.count % self.update_count == 0)
            or (
                (self.count < self.update_count)
                and (self.count % int(self.update_count / 10) == 0)
            )
        ):
            self.update()
        if self.count >= self.refresh_count:
            self.refresh()

        self.fileObj.write(text + "\n")
        self.count += add
        print(text)

    def close(self):
        self.fileObj.close()
