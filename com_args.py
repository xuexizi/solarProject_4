import subprocess
from memory import MemoryDNN
import numpy as np
import cvxpy as cp
import pandas as pd
import os
import multiprocessing
import time
import queue

solve_index = 6  # 执行次数，保存结果文件序列
simu_filename = 'simu_data/simu_' + str(solve_index) + '.txt'
PHOTO_RANGE = (50, 110)  # 每个时间片中照片个数范围，三个等级的均值：50/80/110 ————（30-70）（50-110）（80-140）
FACE_RANGE = (3, 7)  # 每张照片人脸个数范围，三个等级的均值：3/5/7 ————（1-5）（3-7）（5-9）

T = 6 * 24 * 3  # 时间片个数
delta = 10  # 每个时间片时长（10分钟）

L_max = 600  # 也就是论文中的L*，队列最大长度
L_opt = 0.5 * L_max

B_max = 1300  # 电池容量
B_start = B_max * 1
K_safe = 0.2
K_charge_full = 0.6

V = 2000  # 李雅普诺夫偏移惩罚系数
Eta = 1.5  # 处理器运行功率系数

# 这里的 Mu_max 和 P_idle 没有关系，我们可以将其看成是两个芯片。当x_t=0时，两个芯片之间的数据通道是关闭的，所以求解功率时分别考虑kf^3
P_idle = 0.5  # 系统空闲时（包括存储照片）的运行功率
Mu_max = 2  # 最大处理能力（影响求解是否成功）：1.8GHz。
Mu_min = 0.5 * Mu_max  # 最低处理能力（影响求解是否成功）：启动程序的最低要求
E_u_idle_t = P_idle * delta

T_s = 0.005  # 每张图片利用全部设备处理能力 Mu_max 来清点人脸数所需要的时长（0.3秒）
T_r = 0.003  # 每个人脸在利用全部资源 Mu_max 来识别的时间（0.18秒）
T_p = 0.015  # 最快启动程序所需时间（0.9秒钟）


# 生成模拟数据：每时刻照片信息
def generate_data(filename):
    # 照片数据
    for i in range(T):
        photo_num = np.random.randint(*PHOTO_RANGE)
        face_list = np.random.randint(*FACE_RANGE, size=photo_num)
        photo_info = [photo_num, *face_list]
        column_names = [i for i in range(len(photo_info))]
        df = pd.DataFrame(columns=column_names, index=[])
        df.loc[0] = photo_info
        df.index = df.index + 1
        df.to_csv(filename, mode='a', index=False, header=False)


# 读取照片数据
def read_photo_file(filename):
    column_names = [i for i in range(0, T)]  # 列数不小于实际数据的列数
    data = pd.read_csv(filename, names=column_names, header=None, engine='python')  # 读取大文件时，需要更换python引擎
    photo_info = []
    for i in range(T):
        per_T_photo = np.array(data.iloc[i]).astype(int)
        photo_info.append(per_T_photo[1:per_T_photo[0] + 1])
    return photo_info


# 读取solar数据，读取后需要除以60，原因是本来的solar指的是watt/m^2。但是 1、我们的太阳能板没有那么大 2、因为太阳直射角不同，所以有损耗
def read_charge_file(file_dir):
    # file_dir = "simu_data/mon_1"  # file directory
    all_csv_list = os.listdir(file_dir)  # get csv list
    all_data_frame = []
    for single_csv in all_csv_list:
        single_data_frame = pd.read_csv(os.path.join(file_dir, single_csv),
                                        sep='\t',
                                        header=0,
                                        usecols=["--Timestamp---", "Solar"],
                                        converters={"Solar": lambda x: round(int(x) / 60, 2)},
                                        skiprows=lambda x: x > 0 and x % 2 == 0
                                        )
        if single_csv == all_csv_list[0]:
            all_data_frame = single_data_frame
        else:
            all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
    return all_data_frame["Solar"].values


# 需要考虑的输入变量
# Y[t]= max(Y[t-1] + L[t] - L_opt, 0)  与队列长度有关
# L[t] = min(L[t-1] + C_t, L_max)  队列长度
# C_t  t时刻到达的照片数目
# L_t_queue  照片队列
# E_c_t  当前时段（太阳能）充了多少电
# B_t  电池剩余量
def manual_solve(Alg_index, m_min, Y_t, Q_t, L_t, C_t, L_t_queue, E_c_t, B_t, isPrint=False):
    pm_list, mu_list, res_list, E_u_t_list = [], [], [], []

    # 由于 p=m，所以当 m 用枚举确定下来，就只剩下一个变量 mu_t，此时只需要根据所有约束找到 mu_t 的最小值即可
    # 注意：这里不考虑 m==0 的情况，这是 x_t = 0 应该考虑的事情
    for m in range(L_t, m_min, -1):
        N_sum = sum(list(L_t_queue.queue)[:m])

        # 首先排除掉不满足时间约束的m，该约束确保了 mu <= Mu_max
        T_sum = T_s * m + T_r * N_sum + T_p
        if Alg_index == 3:  # 照片来了就处理意味着，有几张照片就要启动几次
            T_sum = T_s * m + T_r * N_sum + T_p * m
        if T_sum > delta:
            continue

        # 然后去掉不满足能量约束导致的无法求解
        mu = max(1.0 * Mu_max / delta * T_sum, Mu_min)
        E_u_work_t = Eta * (mu ** 2) * Mu_max * T_sum
        E_u_t = E_u_idle_t + E_u_work_t  # 耗费电量 = 空闲耗费电量 + 工作耗费电量
        if B_t + E_c_t - E_u_t <= K_safe * B_max:
            continue

        if Alg_index == 1:
            result = Y_t * min(L_t - m + C_t, L_max) + V * E_u_work_t
        else:
            result = E_u_work_t
        pm_list.append(m)
        mu_list.append(mu)
        res_list.append(result)
        E_u_t_list.append(E_u_t)

        # 选择可求解的最大p和m，只针对 照片来了就处理/队满处理 2种情况
        # 这里有个问题：当队满求解时，会出现不满足能量约束的情况，所以需要增加一个是否求解成功的约束，即 len(mu_list) > 0
        if (Alg_index == 3 or Alg_index == 4) and len(mu_list) > 0:
            break

    # 这里的res=-1，对应在now_opt中的在[m_t_cons, L[t]]之间没有可行解
    if len(res_list) == 0:
        return -1, 0, 0, 0

    opt_index = np.argmin(res_list)
    # 当因为时，选择可求解的最大值
    if Alg_index == 3 or Alg_index == 4:
        opt_index = -1

    if isPrint:
        print("res = ", res_list)
        print("pm = ", pm_list)
        print("mu = ", mu_list)
    return res_list[opt_index], mu_list[opt_index], pm_list[opt_index], E_u_t_list[opt_index]


# generate_data(simu_filename)
