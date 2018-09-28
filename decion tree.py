# coding=utf-8
# The implementation of the decition tree algorithm!
import math
import numpy as np

# 定义属性集合, 用numpy读取数据
abbr_list = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
wmdata = np.loadtxt(open("wmdata.csv"), dtype=str, delimiter=",", skiprows = 1)

# 树结点的定义
class node:
    def __init__(self, type=None):
        self.type = type
        self.childlist = []
        self.attrlist = []

    def add_child(self, child, attr):
        self.childlist.append(child)
        self.attrlist.append(attr)

# 对数据进行预处理
# 本函数可以自行设置
def data_process(data):
    res = data
    for i,d in enumerate(data[ : , -3]):
        if(d < 0.5):
            res[-3, i] = 'small'
        else:
            res[-3, i] = 'big'

    for i,d in enumerate(data[ : , -2]):
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
    sample_num = len(s_list)
    ent = compute_ent(s_list)

    # 用于判断正反例
    flag = wmdata[s_list[0]][-1]

    for attr in a_list:
        i = abbr_list.index(attr) + 1

        abbr_type = []
        abbr_num = {}

        # 计算每种属性的信息增益,并放入gain_dict中
        # 先计算每种属性不同类别的正反例个数，放在abbr_num[abbr][]里
        for s in s_list:
            if wmdata[s][i] not in abbr_type:
                abbr_type.append(wmdata[s][i])
                if (wmdata[s][-1] == flag):
                    abbr_num[wmdata[s][i]] = [1, 0]
                else:
                    abbr_num[wmdata[s][i]] = [0, 1]
            else:
                if (wmdata[s][-1] == flag):
                    abbr_num[wmdata[s][i]][0] += 1
                else:
                    abbr_num[wmdata[s][i]][1] += 1

        # print(abbr_num)
        # 再计算该属性总的信息增益
        ent_abbr_list = []
        for num in abbr_num.values():
            z = num[0]
            f = num[1]

            if z != 0 and f != 0:
                gain = -z/(z+f) * math.log(z/(z+f), 2) - f/(z+f) * math.log(f/(z+f), 2)
            else:
                gain = 0
            ent_abbr_list.append([gain, z+f])

        ent_abbr = 0
        for i in ent_abbr_list:
            ent_abbr += (i[1]/sample_num) * i[0]

        gain_dict[attr] = ent - ent_abbr

    # 找到最优划分属性和这个属性对应的属性类别
    res = None
    max = -math.inf
    for gain in gain_dict.keys():
        if gain_dict[gain] > max:
            res = gain
            max = gain_dict[gain]

    index = abbr_list.index(res) + 1
    attr_type_list = []
    for s in s_list:
        if wmdata[s][index] not in attr_type_list:
            attr_type_list.append(wmdata[s][index])

    return  res, attr_type_list

# 计算当前样本集的信息熵
# 返回最优划分属性和该属性的类别
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

# 判断是否当前所有属性都单一取值
def is_same_attr(s_list, a_list):
    for a in a_list:
        index = abbr_list.index(a) + 1
        attr = wmdata[s_list[0]][index]

        for s in s_list:
            if wmdata[s][index] != attr:
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

# 遍历生成的决策树
def traverse_tree(tree, floor, attr = None, attr_type = None, root = False):
    if root is True:
        print('The', floor, 'floor:', tree.type)
    else:
        print('The', floor, 'floor: if', attr, 'is', attr_type, ',then', tree.type)
    if(len(tree.childlist) > 0):
        for i, child in enumerate(tree.childlist):
            traverse_tree(child, floor+1, attr = tree.type, attr_type=tree.attrlist[i])

# 核心部分，通过样本集和属性集生成决策树
def treeGenerate(s_list, a_list):
    this = node()

    if is_same_type(s_list):
        this.type = wmdata[s_list[0]][-1]
        return this

    if len(a_list) <= 0 or is_same_attr(s_list, a_list):
        this.type = get_most_type(s_list)
        return this

    gain, attr_type_dict = compute_gain(s_list, a_list)
    # print(gain_dict)
    this.type = gain

    index = abbr_list.index(gain) + 1
    a_list.remove(gain)
    # print(a_list)

    # 根据划分属性的不同类别，继续向下划分
    for attr_type in attr_type_dict:
        s_list_child = []
        for i in s_list:
            if wmdata[i][index] == attr_type:
                s_list_child.append(i)

        if len(s_list_child) > 0:
            this.add_child(treeGenerate(s_list_child, a_list), attr_type)

    return this

if __name__ == "__main__":
    s_list = list(range(len(wmdata)))
    a_list = abbr_list

    tree = treeGenerate(s_list, a_list)
    traverse_tree(tree, 1, root=True)