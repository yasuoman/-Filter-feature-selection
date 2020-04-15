import numpy as np
import math


# 针对标签样本都是二值的互信息,label和sample是对称的
def binary_mutula_information(label, sample):
    d = dict()  # 用字典来计数
    # 统计其中00,01,10,11各自的个数
    binary_mi_score = 0.0
    label = np.asarray(label)
    sample = np.asarray(sample)
    if label.size != sample.size:
        print('error,input array length is not equal')
        exit()
    x = [1 - np.sum(label) / label.size, np.sum(label) / label.size]
    # np.sum(label)/label.size表示1在label中的概率,
    # 前者就是0在label中的概率
    y = [1 - np.sum(sample) / sample.size, np.sum(sample) / sample.size]

    for i in range(label.size):
        if (label[i], sample[i]) in d:
            d[label[i], sample[i]] += 1
        else:
            d[label[i], sample[i]] = 1

    # 遍历字典，得到各自的px,py,pxy，并求和
    for key in d.keys():
        px = x[key[0]]
        py = y[key[1]]
        pxy = d[key] / label.size
        binary_mi_score = binary_mi_score + pxy * math.log(pxy / (px * py))

    return binary_mi_score


y = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
x = [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
a = binary_mutula_information(x, y)
print(a)
