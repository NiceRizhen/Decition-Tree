# coding=utf-8
# the implementation of the decition tree algothyme!

import numpy as np
import math

# data 的行数-1即为在list里的下标
list_name = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
wmdata = np.loadtxt(open("wmdata.csv"), dtype=np.str, delimiter=",", skiprows = 1)

# 转置， 使每一行为同一属性
wmdata = wmdata.T

C = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
D = []

def data_process(data):
    res = data
    for i,d in enumerate(data[-3, :]):
        if(d < 0.5):
            res[-3, i] = 'small'
        else:
            res[-3, i] = 'big'

    for i,d in enumerate(data[-2, :]):
        if(d < 0.5):
            res[-3, i] = 'small'
        else:
            res[-3, i] = 'big'

    return res

# 输入的data为某一属性的数据，输出该属性下的类别与个数
def get_type(data):
    if(len(data) is 0 or data is None):
        return None

    type = []
    sta = {}

    for i in data:
        if i not in type:
            type.append(i)
            sta[i] = 0
        else:
            sta[i] += 1

    return sta

# 输入data为某属性的数据, ent为信息熵
def compute_gain(data, ent):
    sta = get_type(data)
    num = len(data)

    staz = sta
    for i in staz:
        staz[i] = 0

    for i in data:
        staz[i] = staz[i] + 1

    res = 0
    for i in sta.keys():
        # z 表示该分类下正例的个数，all表示该分类样本的个数
        z = staz[i]
        all = sta[i]

        # 计算每一类的信息熵
        res = res - (all/num) * ((z/all) * math.log(z/all,2)\
                                 + ((all-z)/all) * math.log((all-z)/all, 2))

    gain = ent - res

    return gain

# 传入
def compute_ent(data):
    res = 0
    num = len(data)

    sta = get_type(data)
    for n in sta.values():
        res = res - (n/num) * math.log(n/num, 2)

    return res

def build_tree():
    new_data = data_process(wmdata)

    ent = compute_ent(new_data[-1])