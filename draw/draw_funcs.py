import numpy as np
from com_args import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

mpl.use('TkAgg')  # 阻止警告

# pdf文档中有Type3 字体，不被认可，所以下面下面两行是将其改成Type 1字体
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# alg_names = ["Ours", "Strict", "Active", "Lazy", "XGBoost"]
alg_names = ["Ours", "Strict", "Active", "Lazy"]
table_names = ["Energy", "QLen", "X", "P_M", "Battery", "charge_flag", "lose_photo"]
colors = ["b", "orange", 'g', 'r', 'k']


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


def find_end_index(df_list, all_index=1009):
    end_index = [all_index, all_index, all_index, all_index]
    for i in range(4):
        y = df_list[i]["Battery"].values
        for j in range(all_index):
            if y[j] < B_max * K_safe:
                end_index[i] = j
                break
    return end_index


def draw_subplot(subplot_index, file_path, col_name, isStop=True, isLegend=False, all_index=1009):
    # 读取四个算法的结果
    df_list = []
    for i in range(4):
        df = pd.read_table(file_path + str(i + 1) + '_' + alg_names[i] + '.txt', sep=',', names=table_names)
        df_list.append(df)

    # 找到每个算法的截止时间点，即电量第一次低于安全电压的时间点
    if isStop:
        end_index = find_end_index(df_list)
    else:
        end_index = [all_index, all_index, all_index, all_index]

    # 绘制四个算法的折线图
    for alg_i in range(4):
        x = [i for i in range(end_index[alg_i])]
        if col_name == "Battery":
            y = df_list[alg_i][col_name].values[:end_index[alg_i]]
        else:
            y = cummean(df_list[alg_i][col_name].values)[:end_index[alg_i]]

        plt.plot(x, y, color=colors[alg_i], label=alg_names[alg_i])

        plt.xlim((0, all_index-1))
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
        plt.title("Mon " + str(subplot_index))

        if subplot_index == 1 or isLegend:
            plt.legend(loc='best', frameon=False)
