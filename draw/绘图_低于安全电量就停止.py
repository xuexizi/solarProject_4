# 该文件的效果和 "绘图.py" 是一样的，但是可以绘制四个算法在不同时刻停止的x

from draw.draw_funcs import *


# table_names = ["Energy", "QLen", "X", "P_M", "Battery", "charge_flag", "lose_photo"]
# alg_names = ["Ours", "Strict", "Active", "Lazy"]
# colors = ["b", "orange", 'g', 'r']

pic_names = ["Energy", "QLen", "Battery"]


def draw_pic_2(file_path):

    df_list = []
    for i in range(4):
        df = pd.read_table(file_path + str(i + 1) + '_' + alg_names[i] + '.txt', sep=',', names=table_names)
        df_list.append(df)

    end_index = find_end_index(df_list)

    for index in range(3):
        name = pic_names[index]
        fig = plt.figure(figsize=(8, 5))

        for alg_i in range(4):
            x = [i for i in range(end_index[alg_i])]
            if name == "Battery":
                y = df_list[alg_i][name].values[:end_index[alg_i]]
            else:  # "Energy", "QLen" 需要取平均
                y = cummean(df_list[alg_i][name].values)[:end_index[alg_i]]

            plt.plot(x, y, color=colors[alg_i], label=alg_names[alg_i])

            plt.xlim((0, 1008))
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
            plt.title(name)

            plt.legend(loc='best', frameon=False)

        fig.subplots_adjust(left=0.1, top=0.95, bottom=0.11, right=0.97)
        plt.savefig(file_path + "stop_" + name + '.pdf', dpi=600)


if __name__ == "__main__":
    # for month in range(1, 13):
    #     file_path = "../result/mon_" + str(month) + "/"
    for simu_i in range(2, 24):
        file_path = "../result/mon_4/modify/simu_" + str(simu_i) + "/"
        draw_pic_2(file_path)
