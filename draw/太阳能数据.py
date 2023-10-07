# 该文件用来绘制每个月的太阳能强度图，因为后来将12个月的数据绘制到一张图上，所以该文件弃用

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np
from com_args import read_charge_file
mpl.use('TkAgg')  # 阻止警告


# pdf文档中有Type3 字体，不被认可，所以下面下面两行是将其改成Type 1字体
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


for month in range(1, 13):

    file_path = "../simu_data/mon_" + str(month)
    solar_data = read_charge_file(file_path)

    fig = plt.figure(figsize=(6, 5))

    x = [i for i in range(1008)]
    plt.plot(x, solar_data, color='indianred', linestyle='-')  # indianred  crimson

    plt.xlim((0, 1008))  # 设置x轴的刻度范围
    plt.tick_params(top=False, bottom=True, left=True, right=False)  # 控制哪些轴上有刻度线
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(MultipleLocator(144))  # 把y轴的主刻度设置为 yaxis_locator[i] 的倍数

    # ax.spines['right'].set_color('none')   # 设置上边和右边无边框
    # ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('black')  # 设置x边和y边为黑色
    ax.spines['bottom'].set_color('black')

    # 坐标轴刻度值操作
    plt.tick_params(axis='x', pad=10, direction='in')
    plt.tick_params(axis='y', pad=10, direction='in')

    # 画网格线
    # plt.gcf().set_facecolor(np.ones(3) * 240 / 255)  # 生成画布的大小
    # plt.grid(linestyle='-.')  # 生成网格
    plt.grid(True)
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha=0.3)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title(month)

    # plt.legend(loc='best', frameon=False)
    # frameon：是否显示图例边框
    # plt.legend(loc='best', prop={'size': 23}, fontsize=18, frameon=False)

    # fig.subplots_adjust(left=0.1, top=0.95, bottom=0.11, right=0.97)
    plt.savefig("../simu_data/pic/" + str(month) + '.jpg', dpi=600)
