# coding=utf-8

import numpy as np


class KNN(object):
    def __init__(self, dist_func, train_data, train_label, top_k=3, transform=None):
        self.transform = transform
        self.top_k = top_k
        self.dist_func = dist_func
        self.train_data = train_data if transform is None else transform(train_data)
        self.train_label = train_label

    def predict(self, x):
        x = x if self.transform is None else self.transform(x)

        all_dist = []
        for train_sample in self.train_data:
            all_dist.append(self.dist_func(x, train_sample))
        all_dist = np.array(all_dist)
        top_k_index = np.argsort(all_dist)[:self.top_k]

        top_k_cls = self.train_label[top_k_index]

        cls_count = {}
        for cls in top_k_cls:
            if cls in cls_count:
                cls_count[cls] += 1
            else:
                cls_count[cls] = 1

        max_count = -1
        max_cls = -1
        for k, v in cls_count.items():
            if v > max_count:
                max_cls = k
                max_count = v

        if max_count == 1:
            max_cls = top_k_cls[0]

        return max_cls
