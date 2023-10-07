# 该文件本来是负责画点图 "P_M" ，但是后来代码被合并到 "绘图.py" 文件中，所以弃用。

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np

# pdf文档中有Type3 字体，不被认可，所以下面下面两行是将其改成Type 1字体
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# 数组前n项求平均
def cummean(arr):
    cumsum_arr = np.cumsum(arr)
    return cumsum_arr / np.arange(1, len(arr) + 1)


def remove_zero(a, b):
    ra, rb = [], []
    for ia, ib in zip(a, b):
        if ib == 0:
            continue
        else:
            ra.append(ia)
            rb.append(ib)
    return ra, rb


k = 1

for month in range(4, 5):

    file_path = "../result/mon_" + str(month) + "/"
    fig = plt.figure(figsize=(8, 4))

    df1 = pd.read_table(file_path + '1_Lysolve_' + str(k) + '.txt', sep=',', names=["Energy", "QLen", "X", "P_M", "Battery", "charge_flag"])
    df2 = pd.read_table(file_path + '2_now_opt_' + str(k) + '.txt', sep=',', names=["Energy", "QLen", "X", "P_M", "Battery", "charge_flag"])
    df3 = pd.read_table(file_path + '3_now_process_' + str(k) + '.txt', sep=',', names=["Energy", "QLen", "X", "P_M", "Battery", "charge_flag"])
    df4 = pd.read_table(file_path + '4_full_process_' + str(k) + '.txt', sep=',', names=["Energy", "QLen", "X", "P_M", "Battery", "charge_flag"])

    name = "P_M"
    energy_1 = df1[name].values
    energy_2 = df2[name].values
    energy_3 = df3[name].values
    energy_4 = df4[name].values
    df1['Ours'] = energy_1
    df1['Strict'] = energy_2
    df1['Active'] = energy_3
    df1['Lazy'] = energy_4

    x = [i for i in range(1009)]
    # print(len(energy_2))

    # df1.plot(y=['Lysolve', 'Strict', 'Active', 'Lazy'])
    rx, ry = remove_zero(x, energy_1)
    s1 = plt.scatter(rx, ry, s=5, c='b', alpha=0.65)
    rx, ry = remove_zero(x, energy_2)
    s2 = plt.scatter(rx, ry, s=5, c='orange', alpha=0.65)
    rx, ry = remove_zero(x, energy_3)
    s3 = plt.scatter(rx, ry, s=5, c='g', alpha=0.65)
    rx, ry = remove_zero(x, energy_4)
    s4 = plt.scatter(rx, ry, s=5, c='r', alpha=0.65)


    plt.xlim((0, 1008))

    # 控制哪些轴上有刻度线
    plt.tick_params(top=False, bottom=True, left=True, right=False)

    # ax为两条坐标轴的实例
    ax = plt.gca()

    # 把y轴的主刻度设置为 yaxis_locator[i] 的倍数
    ax.xaxis.set_major_locator(MultipleLocator(144))

    # 设置上边和右边无边框
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # 设置x边和y边为黑色
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # 坐标轴刻度值操作
    plt.tick_params(axis='x', pad=10, direction='in')
    plt.tick_params(axis='y', pad=10, direction='in')

    # 画网格线
    # plt.gcf().set_facecolor(np.ones(3) * 240 / 255)  # 生成画布的大小
    # plt.grid(linestyle='-.')  # 生成网格
    plt.grid(True)
    # 设置网格线格式：
    plt.grid(color='grey',
             linestyle='--',
             linewidth=1,
             alpha=0.3)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(name)

    legend1 = ax.legend((s1, s2, s3, s4), (u'LyDROO', u'Strict', u'Active', u'Lazy'),
                        # loc="right",
                        # frameon=False,  # 是否有边框
                        # framealpha=0,  # 是否透明，0表示透明
                        # bbox_to_anchor=(0.06, 0.475),  # 基准的值
                        bbox_to_anchor=(0.825, 0.52),  # simu_5
                        edgecolor='k'
                        )
    ax.add_artist(legend1)


    # plt.legend(loc='best', frameon=False)
    # frameon：是否显示图例边框
    # plt.legend(loc='best', prop={'size': 23}, fontsize=18, frameon=False)

    fig.subplots_adjust(left=0.08, top=0.9, bottom=0.13, right=0.95)
    plt.savefig(file_path + name + '.jpg', dpi=600)
