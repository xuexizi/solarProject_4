from com_args import *

res_dir = ""
charge_filename = ""
photo_filename = ""


def charging(t, photo_data, C_t, E_c_t, L_t_queue, L, lose_photo, Y, B):
    B[t + 1] = min(B[t] + E_c_t, B_max)
    # 将照片入队
    for item in photo_data:
        if L_t_queue.full():
            break
        L_t_queue.put(item)
    L[t + 1] = min(L[t] + C_t, L_max)  # 其实就是 L_t_queue 长度
    lose_photo[t] = max(L[t] + C_t - L_max, 0)  # 存储t时刻丢失的照片数目
    Y[t + 1] = max(Y[t] + L[t + 1] - L_opt, 0)


def x_is_zero(t, photo_data, C_t, E_c_t, L_t_queue, L, lose_photo, B, Y, E_t):
    # 将t时刻所有照片入队
    for item in photo_data:
        if L_t_queue.full():
            break
        L_t_queue.put(item)
    L[t + 1] = min(L[t] + C_t, L_max)  # 其实就是 L_t_queue 长度
    lose_photo[t] = max(L[t] + C_t - L_max, 0)  # 存储t时刻丢失的照片数目
    B[t + 1] = min(B[t] - E_u_idle_t + E_c_t, B_max)  # 电池电量
    Y[t + 1] = max(Y[t] + L[t + 1] - L_opt, 0)
    E_t[t] = E_u_idle_t


def x_is_one(t, pm_t, photo_data, C_t, E_c_t, E_u_t, Q, L_t_queue, L, lose_photo, B, Y, E_t, P_M):
    Q[t + 1] = Q[t]
    # 照片先出队，再入队
    for i in range(pm_t):
        L_t_queue.get()
    for item in photo_data:
        if L_t_queue.full():
            break
        L_t_queue.put(item)
    L[t + 1] = min(L[t] - pm_t + C_t, L_max)
    lose_photo[t] = max(L[t] + C_t - L_max, 0)  # 存储t时刻丢失的照片数目
    B[t + 1] = min(B[t] - E_u_t + E_c_t, B_max)
    Y[t + 1] = max(Y[t] + L[t + 1] - L_opt, 0)
    E_t[t] = E_u_t
    P_M[t] = pm_t


def lazy():
    # 1、读取数据：照片数据
    photo_data = read_photo_file(photo_filename)
    # 2、读取充电功率
    charge_P = read_charge_file(charge_filename)

    L_t_queue = queue.Queue(maxsize=L_max)  # 照片队列
    L = np.zeros(T + 1, dtype=int)  # 队列内照片数
    Q = np.zeros(T + 1, dtype=int)  # 已经完成清点的照片数
    B = np.zeros(T + 1)  # 电池剩余量
    Y = np.zeros(T + 1)  # 虚拟队列
    B[0] = B_start  # 初始电量

    E_t = np.zeros(T + 1)  # 记录每个时刻的功耗
    X = np.zeros(T + 1, dtype=int)  # 记录每个时刻的x_t
    P_M = np.zeros(T + 1, dtype=int)  # 记录每个时刻的 p_t 和 m_t
    charge_flag = np.zeros(T + 1, dtype=int)
    lose_photo = np.zeros(T + 1, dtype=int)  # 丢失照片数目
    for t in range(T):
        print("t = ", t)
        print("当前电量 = ", B[t])

        P_i_t = charge_P[t]  # 当前时段（太阳能）充电功率
        C_t = len(photo_data[t])  # t时刻到达的照片数目
        E_c_t = P_i_t * delta

        # 沉浸式充电，直到电量达到 (1 - K_safe) * B_max
        if t > 0 and charge_flag[t - 1] == 1:
            if B[t] < K_charge_full * B_max:
                charge_flag[t] = 1
                charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
                continue

        # 当电量不符合约束，此时不需要求解，下一时刻进入休眠充电状态
        if B[t] < K_safe * B_max:
            charge_flag[t] = 1
            charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
            continue

        x_t = 0  # 默认不处理照片
        res, mu_t, pm_t, E_u_t = 0, 0, 0, 0
        if L[t] == L_max:  # 如果队列满了再处理，处理尽可能多的照片
            x_t = 1
            res, mu_t, pm_t, E_u_t = manual_solve(4, 0, Y[t], Q[t], L[t], C_t, L_t_queue, E_c_t, B[t])
            P_M[t] = pm_t
            # 当没有可行解时
            if res == -1:
                x_t = 0

        X[t] = x_t


        if x_t == 0 or t == 0:
            x_is_zero(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, B, Y, E_t)
        else:
            x_is_one(t, pm_t, photo_data[t], C_t, E_c_t, E_u_t, Q, L_t_queue, L, lose_photo, B, Y, E_t, P_M)

    # 存储 E_t、L、X、P_M
    print(sum(E_t), sum(L))
    np.savetxt(res_dir + '4_Lazy.txt',
               np.column_stack((E_t, L, X, P_M, B, charge_flag, lose_photo)),
               delimiter=',',
               fmt='%.3f')


def active():
    # 1、读取数据：照片数据
    photo_data = read_photo_file(photo_filename)
    # 2、读取充电功率
    charge_P = read_charge_file(charge_filename)

    L_t_queue = queue.Queue(maxsize=L_max)  # 照片队列
    L = np.zeros(T + 1, dtype=int)  # 队列内照片数
    Q = np.zeros(T + 1, dtype=int)  # 已经完成清点的照片数
    B = np.zeros(T + 1)  # 电池剩余量
    Y = np.zeros(T + 1)  # 虚拟队列
    B[0] = B_start  # 初始电量

    E_t = np.zeros(T + 1)  # 记录每个时刻的功耗
    X = np.zeros(T + 1, dtype=int)  # 记录每个时刻的x_t
    P_M = np.zeros(T + 1, dtype=int)  # 记录每个时刻的 p_t 和 m_t
    charge_flag = np.zeros(T + 1, dtype=int)
    lose_photo = np.zeros(T + 1, dtype=int)  # 丢失照片数目
    for t in range(T):
        print("t = ", t)
        print("当前电量 = ", B[t])

        P_i_t = charge_P[t]  # 当前时段（太阳能）充电功率
        C_t = len(photo_data[t])  # t时刻到达的照片数目
        E_c_t = P_i_t * delta

        # 沉浸式充电，直到电量达到 (1 - K_safe) * B_max
        if t > 0 and charge_flag[t - 1] == 1:
            if B[t] < K_charge_full * B_max:
                charge_flag[t] = 1
                charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
                continue

        # 当电量不符合约束，此时不需要求解，下一时刻进入休眠充电状态
        if B[t] < K_safe * B_max:
            charge_flag[t] = 1
            charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
            continue

        x_t = 1
        res, mu_t, pm_t, E_u_t = manual_solve(3, 0, Y[t], Q[t], L[t], C_t, L_t_queue, E_c_t, B[t])
        P_M[t] = pm_t
        # 当求解结果为0，即电量不足时
        if res == -1:
            x_t = 0
        X[t] = x_t

        if x_t == 0 or t == 0:
            x_is_zero(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, B, Y, E_t)
        else:
            x_is_one(t, pm_t, photo_data[t], C_t, E_c_t, E_u_t, Q, L_t_queue, L, lose_photo, B, Y, E_t, P_M)

    # 存储 E_t、L、X、P_M
    print(sum(E_t), sum(L))
    np.savetxt(res_dir + '3_Active.txt',
               np.column_stack((E_t, L, X, P_M, B, charge_flag, lose_photo)),
               delimiter=',',
               fmt='%.3f')


def strict():
    # 1、读取数据：照片数据
    photo_data = read_photo_file(photo_filename)
    # 2、读取充电功率
    charge_P = read_charge_file(charge_filename)

    L_t_queue = queue.Queue(maxsize=L_max)  # 照片队列
    L = np.zeros(T + 1, dtype=int)  # 队列内照片数
    Q = np.zeros(T + 1, dtype=int)  # 已经完成清点的照片数
    B = np.zeros(T + 1)  # 电池剩余量
    Y = np.zeros(T + 1)  # 虚拟队列
    B[0] = B_start  # 初始电量

    E_t = np.zeros(T + 1)  # 记录每个时刻的功耗
    X = np.zeros(T + 1, dtype=int)  # 记录每个时刻的x_t
    P_M = np.zeros(T + 1, dtype=int)  # 记录每个时刻的 p_t 和 m_t
    charge_flag = np.zeros(T + 1, dtype=int)  # 休眠充电flag
    lose_photo = np.zeros(T + 1, dtype=int)  # 丢失照片数目
    for t in range(T):
        print("t = ", t)
        print("当前电量 = ", B[t])

        P_i_t = charge_P[t]  # 当前时段（太阳能）充电功率
        C_t = len(photo_data[t])  # t时刻到达的照片数目
        E_c_t = P_i_t * delta

        # 沉浸式充电，直到电量达到 (1 - K_safe) * B_max
        if t > 0 and charge_flag[t - 1] == 1:
            if B[t] < K_charge_full * B_max:
                charge_flag[t] = 1
                charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
                continue

        # 当电量不符合约束，此时不需要求解，下一时刻进入休眠充电状态
        if B[t] < K_safe * B_max:
            charge_flag[t] = 1
            charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
            continue

        # 队列长度约束
        m_t_cons = int(L[t] + C_t - (t + 1) * L_opt + sum(L[:t]))
        m_t_cons = min(m_t_cons, L[t])
        x_t = 0  # 默认不处理照片
        res, mu_t, pm_t, E_u_t = 0, 0, 0, 0
        if m_t_cons > 0:
            x_t = 1
            res, mu_t, pm_t, E_u_t = manual_solve(2, m_t_cons, Y[t], Q[t], L[t], C_t, L_t_queue, E_c_t, B[t])

            if res == -1:
                res, mu_t, pm_t, E_u_t = manual_solve(3, 0, Y[t], Q[t], L[t], C_t, L_t_queue, E_c_t, B[t])
                if res == -1:
                    x_t = 0
            P_M[t] = pm_t

        X[t] = x_t

        if x_t == 0 or t == 0:
            x_is_zero(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, B, Y, E_t)
        else:
            x_is_one(t, pm_t, photo_data[t], C_t, E_c_t, E_u_t, Q, L_t_queue, L, lose_photo, B, Y, E_t, P_M)

    print(sum(E_t), sum(L))
    np.savetxt(res_dir + '2_Strict.txt',
               np.column_stack((E_t, L, X, P_M, B, charge_flag, lose_photo)),
               delimiter=',',
               fmt='%.3f')


def Ours(index):
    # 1、读取数据：照片数据
    photo_data = read_photo_file(photo_filename)
    # 2、读取充电功率
    charge_P = read_charge_file(charge_filename)

    # 3、构建 DNN 模型，并读取预训练模型参数
    # mem = MemoryDNN(net=[5, 16, 8, 1])
    # mem.read_model()

    L_t_queue = queue.Queue(maxsize=L_max)  # 照片队列
    L = np.zeros(T + 1, dtype=int)  # 队列内照片数
    Q = np.zeros(T + 1, dtype=int)  # 已经完成清点的照片数
    B = np.zeros(T + 1)  # 电池剩余量
    Y = np.zeros(T + 1)  # 虚拟队列
    # Pred = np.zeros(T + 1)  # DNN预测x_t的结果：大于0.5视为1，否则视为0
    # Real_outcome = np.zeros(T + 1, dtype=int)  # 实际x_t
    B[0] = B_start  # 初始电量

    E_t = np.zeros(T + 1)  # 记录每个时刻的功耗
    X = np.zeros(T + 1, dtype=int)  # 记录每个时刻的x_t
    P_M = np.zeros(T + 1, dtype=int)  # 记录每个时刻的 p_t 和 m_t
    charge_flag = np.zeros(T + 1, dtype=int)   # 记录每个时刻是否在休眠充电
    lose_photo = np.zeros(T + 1, dtype=int)  # 丢失照片数目
    C = np.zeros(T + 1, dtype=int)  # 存储C_t

    for t in range(T):
        print("t = ", t)
        print("当前电量 = ", B[t])

        P_i_t = charge_P[t]  # 当前时段（太阳能）充电功率
        C_t = len(photo_data[t])  # t时刻到达的照片数目
        C[t] = C_t  # 这个也可以直接从photo文件中读取，但是为了方便，我就直接存储了（用于x_t的学习参数）
        E_c_t = P_i_t * delta

        # 沉浸式充电，直到电量达到 (1 - K_safe) * B_max
        if t > 0 and charge_flag[t-1] == 1:
            if B[t] < K_charge_full * B_max:
                charge_flag[t] = 1
                charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
                continue

        # 当电量不符合约束，此时不需要求解，下一时刻进入休眠充电状态
        if B[t] < K_safe * B_max:
            charge_flag[t] = 1
            charging(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, Y, B)  # 这里因为是浅拷贝，所以会直接改变列表
            continue

        v_list = [0, 0]  # 存储x_t分别为0/1时的最优值
        v_list[0] = Y[t] * min(L[t] + C_t, L_max)
        v_list[1], mu_t, pm_t, E_u_t = manual_solve(1, 0, Y[t], Q[t], L[t], C_t, L_t_queue, E_c_t, B[t])
        if v_list[1] == -1:  # 当没有可行解时，其实也就是即将进入休眠充电状态
            x_t = 0
        else:
            x_t = np.argmin(v_list)
        X[t] = x_t

        if x_t == 0 or t == 0:
            x_is_zero(t, photo_data[t], C_t, E_c_t, L_t_queue, L, lose_photo, B, Y, E_t)
        else:
            x_is_one(t, pm_t, photo_data[t], C_t, E_c_t, E_u_t, Q, L_t_queue, L, lose_photo, B, Y, E_t, P_M)

    # 存储 E_t、L、X、P_M
    print(sum(E_t), sum(L))
    np.savetxt(res_dir + '1_Ours.txt',
               np.column_stack((E_t, L, X, P_M, B, charge_flag, lose_photo)),
               delimiter=',',
               fmt='%.3f')
    # 存储x_t的学习参数和结果
    t_slot = [i for i in range(1009)]
    np.savetxt("learn_xt/" + str(index) + '.txt',
               np.column_stack((t_slot[:1008], L[:1008], B[:1008], charge_P[:1008], Y[:1008], C[:1008], X[:1008])),
               delimiter=',',
               fmt='%.3f'
               )

    # 保存 DNN 模型
    # mem.save_model()
    # mem.plot_cost()

    # print("pred = ", Pred)
    # print("real = ", Real_outcome)


if __name__ == "__main__":

    for index in range(4, 5):
        res_dir = "result/mon_" + str(index) + "/"
        charge_filename = "simu_data/mon_" + str(index)
        photo_filename = 'simu_data/simu_1.txt'

        print("------------Ours------------------")
        Ours(index)
        print("------------strict------------------")
        strict()
        print("------------active------------------")
        active()
        print("------------lazy------------------")
        lazy()
