# 该文件负责绘制算法结果图，每张图都是分开的，并不是画在一个大图上
# 因为用的函数是 df.plot(y=)，四个算法的x取值范围需要一致，所以没办法绘制带有stop的图像，
#       因此stop的图在文件 "绘图_低于安全电量就停止.py" 中。
# 绘图数据包括 pic_names = ["Energy", "QLen", "Battery", "P_M", "lose_photo"]，
#       其中 "Energy", "QLen" 是取得平均值，
#       "P_M", "lose_photo" 是点图，
#       "Battery" 是直接画的折线图。

# 部分代码有点累赘，但是懒得改，先这样了
from draw.draw_funcs import *


# 这个文件用了 df.plot(y=)，四个算法的x取值范围需要一致，所以没办法绘制带有stop的图像
def draw_plot(df, alg_names, file_name, title_name):
    fig = plt.figure(figsize=(6, 4))

    for alg_i, alg_name in enumerate(alg_names):
        x = [i for i in range(1009)]
        y = df[alg_names[alg_i]].values
        plt.plot(x, y, color=colors[alg_i], label=alg_names[alg_i])
    # df.plot(y=alg_names)

    plt.xlim((0, 1008))  # 设置x轴的刻度范围 1008
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
    plt.title(title_name)

    plt.legend(loc='best', frameon=False)
    # frameon：是否显示图例边框
    # plt.legend(loc='best', prop={'size': 23}, fontsize=18, frameon=False)

    fig.subplots_adjust(left=0.1, top=0.85, bottom=0.1, right=0.95)
    plt.savefig(file_name + '.jpg', dpi=600)
    # plt.close()


def draw_pic_1(file_path):

    df_list = []
    for i, alg_name in enumerate(alg_names):
        df = pd.read_table(file_path + str(i+1) + '_' + alg_name + '.txt', sep=',', names=table_names)
        df_list.append(df)

    pic_tables = ["Energy", "QLen", "Battery", "P_M", "lose_photo"]
    pic_names = ["Average power consumption", "Average queue length", "Remaining battery power",
                 "The number of processed photos", "The number of missing photos"]
    # for table in pic_tables:
    for table_i, table in enumerate(pic_tables):
        df = pd.DataFrame()

        average_tables = ["Energy", "QLen"]  # 需要取平均
        if table in average_tables:
            for i, alg_name in enumerate(alg_names):
                df[alg_name] = cummean(df_list[i][table].values)
        else:
            for i, alg_name in enumerate(alg_names):
                df[alg_name] = df_list[i][table].values

        plot_tables = ["Energy", "QLen", "Battery"]  # 画折线图
        if table in plot_tables:
            draw_plot(df, alg_names, file_path + table, pic_names[table_i])

        else:  # 画scatter点图
            fig = plt.figure(figsize=(6, 4))
            # plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot

            x = [i for i in range(1009)]  # 1009, 4033
            s_list = []
            for i, alg_name in enumerate(alg_names):
                rx, ry = remove_zero(x, df[alg_name])
                s = plt.scatter(rx, ry, s=5, c=colors[i], alpha=0.65)
                s_list.append(s)

            plt.xlim((0, 1008))  # 1008, 4032

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

            # plt.xlabel('X')
            # plt.ylabel('Y')
            plt.title(pic_names[table_i])

            # (u'LySolve', u'Strict', u'Active', u'Lazy', u'XGBoost')
            legend1 = ax.legend(s_list, alg_names,
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

            fig.subplots_adjust(left=0.1, top=0.88, bottom=0.1, right=0.92)
            plt.savefig(file_path + table + '.jpg', dpi=600)
            plt.close()


if __name__ == "__main__":
    # for month in range(1, 13):
    #     file_path = "../result/mon_" + str(month) + "/"

    # for simu_i in range(4, 5):
    #     file_path = "../result/mon_4/modify/simu_" + str(simu_i) + "/"
    #     draw_pic_1(file_path)

    file_path = "../result/mon_4/"
    draw_pic_1(file_path)
