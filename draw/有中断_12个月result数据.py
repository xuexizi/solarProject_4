# 该文件用来绘制 2*6 的大图，也就是在基准参数下，12个月内的四种算法分别的结果表现
# 绘图数据包括 pic_names = ["Battery", "QLen", "Energy"]
#       其中 "Energy", "QLen" 是取得平均值，
#       "Battery" 是直接画的折线图。

import matplotlib as mpl
import matplotlib.pyplot as plt
from draw.draw_funcs import draw_subplot

mpl.use('TkAgg')  # 阻止警告


if __name__ == "__main__":
    pic_names = ["Battery", "QLen", "Energy"]
    for i in range(len(pic_names)):
        fig = plt.figure(figsize=(20, 6))

        # pdf文档中有Type3 字体，不被认可，所以下面下面两行是将其改成Type 1字体
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        isStop = False
        col_name = pic_names[i]
        for month in range(1, 13):
            plt.subplot(2, 6, month)
            file_path = "../result/mon_" + str(month) + "/"
            draw_subplot(month, file_path, col_name, isStop=isStop)

        # 设置子图之间的距离
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)

        fig.subplots_adjust(left=0.03, top=0.93, bottom=0.07, right=0.98)
        if isStop:
            plt.savefig("../result/all_pic/stop_" + col_name + ".pdf", dpi=600)
        else:
            plt.savefig("../result/all_pic/" + col_name + ".pdf", dpi=600)
