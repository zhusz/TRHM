# Written with the help of codebase of https://github.com/zhusz/ICLR22-DGS/tree/reimplemented, released
# under the following license:

# MIT License
#
# Copyright (c) 2022 Shizhan Zhu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dominate
import imageio
import numpy as np
from dominate.tags import *

import unio


class HTML(object):
    def __init__(self, uconf, web_dir, title, reflesh=0):

        self.uconf = uconf

        assert web_dir.endswith("/")
        if unio.isdir(web_dir, uconf):
            unio.rmdir(web_dir, uconf)
        unio.mkdirs(web_dir + "images/", uconf)

        self.title = title
        self.web_dir = web_dir
        self.img_dir = self.web_dir + "images/"

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_header_h4(self, str):
        with self.doc:
            h4(str)

    def add_header_h5(self, str):
        with self.doc:
            h5(str)

    def add_text(self, s):
        with self.doc:
            p(s)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        with p():
                            with a(im):  # zhuzhu
                                br()
                                img(style="width:%dpx" % width, src=link)
                            br()
                            if not (txt is None):
                                p(txt)

    def save(self):
        html_file = "%sindex.html" % self.web_dir
        with unio.open(html_file, self.uconf, "w") as f:
            f.write(self.doc.render())


def storeImagesAndReturnLinks(uconf, str_store, str_link, strs_imgs, imgsContents):
    links = []
    assert len(strs_imgs) == len(imgsContents)
    for i in range(len(strs_imgs)):
        if len(imgsContents[i].shape) in [2, 3]:  # It is png
            if len(imgsContents[i].shape) == 2:
                assert imgsContents[i].shape[0] > 5
                assert imgsContents[i].shape[1] > 5
            if "float32" in str(imgsContents[i].dtype):
                tmp = imgsContents[i].copy()
                tmp[tmp < 0] = 0.0
                tmp[tmp > 1] = 1.0
                tmp = (tmp * 255.0).astype(np.uint8)
                # io.imsave(str_store + strs_imgs[i] + '.png', tmp)
                with unio.open(str_store + strs_imgs[i] + ".png", uconf, "wb") as f:
                    imageio.imwrite(f, tmp, format="png")
            else:
                # io.imsave(str_store + strs_imgs[i] + '.png', imgsContents[i])
                with unio.open(str_store + strs_imgs[i] + ".png", uconf, "wb") as f:
                    imageio.imwrite(f, imgsContents[i], format="png")
            links.append(str_link + strs_imgs[i] + ".png")
        elif len(imgsContents[i].shape) == 4:  # It is gif
            print(strs_imgs[i])
            raise NotImplementedError("GIF Manifold dump is not yet implemented.")
            # gifWrite(str_store + strs_imgs[i] + ".gif", imgsContents[i])
            # links.append(str_link + strs_imgs[i] + ".gif")
        else:
            raise ValueError(
                "Shape of imgsContents[i] is wierd %s" % str(imgsContents[i].shape)
            )

    return links


class HTMLStepper(object):
    def __init__(
        self, uconf, logDir, singleHtmlSteps, htmlName
    ):  # consider to use htmlName to represent different experiments.
        # Static member
        self.uconf = uconf
        assert unio.isdir(logDir, uconf)
        if not logDir.endswith("/"):
            logDir += "/"
        self.logDir = logDir
        self.htmlName = htmlName
        self.singleHtmlSteps = singleHtmlSteps

        # dynamics
        self.htmlStepCount = float("inf")
        self.currentFileID = None
        self.html = None

    def step2(
        self, summary0=None, txt0=None, brInds=(0,), headerMessage=None, subMessage=None
    ):
        assert brInds[0] == 0
        if txt0 is not None:
            assert len(summary0) == len(txt0)
        if self.htmlStepCount >= self.singleHtmlSteps:  # reset
            assert unio.isdir(self.logDir, self.uconf)
            unio.mkdirs(self.logDir + "html_" + self.htmlName + "/", self.uconf)
            tmp = [
                int(x.split("/")[-1][5:])
                for x in unio.ls(
                    self.logDir + "html_" + self.htmlName + "/", self.uconf
                )
                if x.split("/")[-1].startswith("html_")
            ]
            if tmp:  # safe if  # tmp is a non-empty set
                maxHtmlFileID = max(tmp)
            else:  # tmp is an empty set
                maxHtmlFileID = -1
            self.currentFileID = maxHtmlFileID + 1
            self.html = HTML(
                self.uconf,
                self.logDir
                + "html_"
                + self.htmlName
                + "/"
                + "html_%06d/" % self.currentFileID,
                self.htmlName,
            )  # zhuzhu
            self.htmlStepCount = 0

        if headerMessage is not None:
            self.html.add_header(headerMessage)
        if subMessage is not None:
            self.html.add_header_h4(subMessage)
        if summary0 is not None:
            links = storeImagesAndReturnLinks(
                self.uconf,
                self.logDir
                + "html_"
                + self.htmlName
                + "/html_%06d/images/%s_%06d_%06d_"
                % (
                    self.currentFileID,
                    self.htmlName,
                    self.currentFileID,
                    self.htmlStepCount,
                ),
                "images/%s_%06d_%06d_"
                % (self.htmlName, self.currentFileID, self.htmlStepCount),
                list(summary0.keys()),
                list(summary0.values()),
            )
            for o in range(len(brInds)):
                head = brInds[o]
                if head >= len(summary0):
                    break
                if o == len(brInds) - 1 or brInds[o + 1] >= len(summary0):
                    tail = len(summary0)
                else:
                    tail = brInds[o + 1]
                if txt0 is None:
                    self.html.add_images(
                        list(summary0.keys())[head:tail],
                        [None for _ in range(head, tail)],
                        links[head:tail],
                    )
                else:
                    self.html.add_images(
                        list(summary0.keys())[head:tail],
                        [txt0[i] for i in range(head, tail)],
                        links[head:tail],
                    )
        self.html.save()
        self.htmlStepCount += 1
