# coding=utf-8

import os
import numpy as np


class Compose(object):
    def __init__(self, process_list):
        self.process_list = process_list

    def __call__(self, img):
        for process in self.process_list:
            img = process(img)
        return img


class ThresholdProcess(object):
    def __init__(self, threshold, fg, bg):
        self.threshold = threshold
        self.fg = fg
        self.bg = bg

    def __call__(self, img):
        img = np.where(img > self.threshold, self.fg, self.bg)
        return img
