# coding=utf-8

import os
import knn
import img_tools
import dist_tools
import mnist_tools

import numpy as np

train_img_path = './data/train-images-idx3-ubyte.gz'
train_label_path = './data/train-labels-idx1-ubyte.gz'

test_img_path = './data/t10k-images-idx3-ubyte.gz'
test_label_path = './data/t10k-labels-idx1-ubyte.gz'

train_img = mnist_tools.parse_mnist_file(train_img_path)
test_img = mnist_tools.parse_mnist_file(test_img_path)

train_label = mnist_tools.parse_mnist_file(train_label_path)
test_label = mnist_tools.parse_mnist_file(test_label_path)

print test_img.shape
print len(test_img)
# print test_img.dtype

process = img_tools.Compose([img_tools.ThresholdProcess(127., 1., 0.)])

# img_0 = process(test_img)
# # print img_0
# # print img_0.dtype
# print img_0[0]
#
# a1 = np.array([0, 0])
# a2 = np.array([0, 0])
# print dist_tools.l1_dist(img_0[0], img_0[0])
# print dist_tools.l2_dist(img_0[0], img_0[0])
# print dist_tools.l2_dist(img_0[0], img_0[1])

test_knn = knn.KNN(dist_tools.l2_dist, train_img, train_label, transform=process)

test_num = 0
hit_num = 0

for i in range(len(test_img)):
    if i % 500 == 0:
        print i, '/', len(test_img)
    gt_cls = test_label[i]
    pred_cls = test_knn.predict(test_img[i])
    if gt_cls == pred_cls:
        hit_num += 1
    test_num += 1

print hit_num, '/', test_num
print float(hit_num) / test_num