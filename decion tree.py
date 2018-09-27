# coding=utf-8
# the implementation of the decition tree algothyme!

import math
import numpy as np

# data 的行数-1即为在list里的下标
abbr_list = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
wmdata = np.loadtxt(open("wmdata.csv"), dtype=np.str, delimiter=",", skiprows = 1)

# 树结点的定义
class node:
    def __init__(self, type=None):
        self.type = type
        self.childlist = []
        self.attrlist = []

    def add_child(self, child, attr):
        self.childlist.append(child)
        self.attrlist.append(attr)

# 对数据进行预处理，这里针对数据集修改
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

# 计算样本里的正例、反例个数
def get_type(s_list):
    z = 0
    f = 0

    for i in s_list:
        if wmdata[i][-1] == wmdata[s_list[0]][-1]:
            z += 1
        else:
            f += 1

    return [z, f]

# 找到最优划分属性和每种属性的gain
def compute_gain(s_list, a_list):
    gain_dict = {}

    ent = compute_ent(s_list)

    for attr in a_list:
        i = abbr_list.index(attr) + 1

        abbr_type = []
        abbr_num = {}
        for s in s_list:
            if wmdata[s][i] not in abbr_type:
                abbr_type.append(wmdata[s][i])
                abbr_num[wmdata[s][i]] = 1
            else:
                abbr_num[wmdata[s][i]] += 1

        all = 0
        gain = 0
        for num in abbr_num.values():
            all += num

        for num in abbr_num.values():
            gain = gain - (num/all) * math.log(num/all, 2)

        gain_dict[attr] = ent - gain

    # 找到最优划分属性
    res = None
    max = -math.inf
    for gain in gain_dict.keys():
        if gain_dict[gain] > max :
            res = gain
            max = gain_dict[gain]

    return  res, gain_dict

# 计算当前样本集的信息熵
def compute_ent(s_list):

    z, f = get_type(s_list)
    res = - z/(z+f) * math.log(z/(z+f), 2)\
          - f/(z+f) * math.log(f/(z+f), 2)

    return res

# 判断样本集中的样本是否同一类别
def is_same_type(s_list):
    t = wmdata[s_list[0]][-1]

    for i in s_list:
        if wmdata[i][-1] != t:
            return False

    return True

# 找出当前样本中个数较多的类别
def get_most_type(s_list):
    type_list = []
    type_num = {}

    # 记录出现的类别和个数
    for i in s_list:
        t = wmdata[i][-1]
        if t not in type_list:
            type_list.append(t)
            type_num[t] = 1
        else:
            type_num[t] += 1

    max = 0
    res = None
    for i in type_num.keys():
        if type_num[i] > max:
            res = i
            max = type_num[i]

    return res

# 核心部分，通过样本集和属性集生成决策树
def treeGenerate(s_list, a_list):

    this = node()

    if is_same_type(s_list):
        this.type = wmdata[s_list[0]][-1]
        return this

    if a_list is None:
        this.type = get_most_type(s_list)
        return this

    gain, gain_dict = compute_gain(s_list, a_list)

    this.type = gain

    index = abbr_list.index(gain) + 1
    a_list_child = a_list.remove(gain)
    for attr_type in gain_dict.keys():
        s_list_child = []
        for i in s_list:
            if wmdata[i][index] == attr_type:
                s_list_child.append(i)
        if len(s_list_child) > 0:
            this.add_child(treeGenerate(s_list_child, a_list_child), attr_type)

    return this

if __name__ == "__main__":
    s_list = list(range(len(wmdata)))
    a_list = abbr_list

    tree = treeGenerate(s_list, a_list)

    print(tree.childlist)