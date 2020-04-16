import numpy as np
import pandas as pd
from data_tools import ATMData

class DE():
    def __init__(self, DATA, F, pop_size, max_iter, prop_mut):

        self.DATA = DATA
        self.DATA_ROOT = self.DATA.data_folder_root

        self.F = F
        self.V, self.U = None, None
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.prop_mut = prop_mut
        # 加钞上下界
        self.lb = np.zeros(self.DATA.ATM_number, dtype=int)
        self.ub = (self.DATA.max_add_money).astype(int)

        self.COST = np.zeros(pop_size)

        self.generation_best_add_money = []
        self.generation_best_COST = []

        self.all_history_COST = []
        self.history_best_add_money = []
        self.history_best_COST = []

        self.create_pop()  ## 完成初始值的设置，以及初代数据的更新


    ## 该函数完全可重用
    def create_pop(self):
        '''
        创建size = self.pop_size的初始数组
        加钞的原则按照ML原则来加，取加钞范围内的整数
        :return: 初始种群
        '''

        #         X
        self.add_money_pop = np.zeros((self.pop_size, self.DATA.ATM_number, self.DATA.days), dtype=int)

        # 对于每个种群，计算 COST，并更新最小下标，最小的代价
        for i in range(self.pop_size):

            # 初始值
            add_money_data = np.zeros((self.DATA.ATM_number, self.DATA.days), dtype=int)

            # 随机化初始值，得到第一代随机值
            for t in range(self.DATA.days):
                add_money_data[:, t] = self.lb + \
                                       np.random.random(self.DATA.ATM_number)  * (self.ub - self.lb)
                # add_money_data[:, t] = self.lb + np.random.randint(0, 121, self.DATA.ATM_number) * 10000
            self.add_money_pop[i, :, :] = add_money_data

            ##   适应度函数计算
            self.COST[i] = ATMData(self.DATA, self.add_money_pop[i, :, :]).update_COST()

        generation_best_index = self.COST.argmin()  # 选择 COST 最小的下标
        self.generation_best_add_money.append(self.add_money_pop[generation_best_index, :, :].copy())  ## 得到 pBestX
        self.generation_best_COST.append(self.COST[generation_best_index])   ## 得到 pBestY
        self.all_history_COST.append(self.COST)  ## 当代的最优值加入到 最优值历史表中

        return self.add_money_pop

    def mutation(self):
        '''
        V[i] = x[best] + F * (x[r2] - x[r3])
        :return:V[i]

        一次变异是针对种群中所有个体进行的
        每个个体都要经过 COST 函数的计算
        '''
        ATM_number = self.DATA.ATM_number # atm 机器数量
        days = self.DATA.days # 天数 8 天
        add_money_pop = self.add_money_pop # 初始化种群
        lb = np.zeros((ATM_number, days), dtype=int) # 初始值取 0
        ub = np.zeros((ATM_number, days), dtype=int) # 初始值取 0
        for t in range(days): # t 天进行迭代
            lb[:, t] = self.lb # lb [第一天数据，第二天数据, 第 t天数据]
            ub[:, t] = self.ub

        mask = np.zeros((self.pop_size, ATM_number, days)) #
        r1 = np.zeros(self.pop_size) # r 也是 popSize * 1 的矩阵
        r2 = np.zeros(self.pop_size)
        r3 = np.zeros(self.pop_size)

        for i in range(self.pop_size): # 对于种群中每个个体，进行跌打
            r = np.arange(0, self.pop_size)
            np.random.shuffle(r)
            r1[i], r2[i], r3[i] = r[0], r[1], r[2]
        r1, r2, r3 = r1.astype(int), r2.astype(int), r3.astype(int)
        # self.V.shape = (pop_size, atm_number, days)
        # self.V = add_money_pop[r1, :, :] + self.F * (add_money_pop[r2, :, :]
        # - add_money_pop[r3, :, :])

        ## V 更新的公式   add_money_pop[r3, :, :] X 的第 r2 维数据，也就是第 Xi 个样本  add_money_pop[r2, :, :] 标识第 r2 个样本

        ## 这里的 V 的维度和 X 的维度是一样的  popSize * atm * days
        # self.generation_best_add_money[-1] 上一步的 Xi
        self.V = self.generation_best_add_money[-1] + \
                 self.F * (add_money_pop[r2, :, :] - add_money_pop[r3, :, :])
        self.V = self.V.astype(int)
        # self.V = (self.V / 10000 + 0.5).astype(int) * 10000

        for i in range(self.pop_size): # 迭代每个个体
            for t in range(days): # 迭代每天
                mask[i, :, t] = self.lb + np.random.random(self.DATA.ATM_number)  * (self.ub - self.lb)
                # mask[i, :, t] = self.lb + np.random.randint(0, 121, self.DATA.ATM_number) * 10000
        mask = mask.astype(int)

        self.V = np.where(self.V < lb, lb, self.V)
        self.V = np.where(self.V > ub, mask, self.V)
        return self.V

    def crossover(self):
        '''
        如果rand < CR  取V否则取X
        :return:
        '''
        ATM_number = self.DATA.ATM_number
        days = self.DATA.days
        random_atm = np.random.randint(0, ATM_number, size=self.pop_size)
        random_days = np.random.randint(0, days, size=self.pop_size)

        mask = np.random.rand(self.pop_size, ATM_number, days) < self.prop_mut
        self.U = np.where(mask, self.V, self.add_money_pop)
        # TODO U继承V
        for i in range(self.pop_size):
            self.U[i, random_atm[i], random_days[i]] = self.V[i, random_atm[i], random_days[i]].copy()
        return self.U

    def selection(self):
        '''
        选择较好的基因遗传下去
        :return: 新一轮的种群
        '''
        COST_X = np.zeros(self.pop_size)
        COST_U = np.zeros(self.pop_size)


        for i in range(self.pop_size):
            COST_X[i] = ATMData(self.DATA, self.add_money_pop[i, :, :]).update_COST()
            COST_U[i] = ATMData(self.DATA, self.U[i, :, :]).update_COST()

        update = COST_U < COST_X
        self.add_money_pop[update, :, :] = self.U[update, :, :].copy()
        self.COST = np.where(COST_X < COST_U, COST_X, COST_U)

        return self.add_money_pop, self.COST

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter

        # max_iter 迭代次数，终止条件设置
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()

            # 记录当前迭代最好的解
            generation_best_index = self.COST.argmin()
            self.generation_best_add_money.append(self.add_money_pop[generation_best_index, :, :].copy())
            self.generation_best_COST.append(self.COST[generation_best_index])
            self.all_history_COST.append(self.COST)
            print('第{}次迭代结束，共{}次迭代'.format(i, self.max_iter))
            print("-------------------------------------")
            print("当前最小的COST为:{:.2f}".format(self.generation_best_COST[-1]))

        # 寻找全局最优解
        global_best_index = np.array(self.generation_best_COST).argmin()
        global_best_add_money = (self.generation_best_add_money[global_best_index] / 10000 + 0.5).astype(int) * 10000
        global_best_COST = ATMData(self.DATA, global_best_add_money).update_COST()

        return global_best_add_money, global_best_COST


if __name__ == '__main__':
    import os
    import time
    from data_tools import LoadData
    DATA_ROOT = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'DATA')

    DATA = LoadData(DATA_ROOT)

    F = 0.8
    pop_size = 50
    max_iter = 100
    prop_mut = 0.8

    start_time = time.time()
    de = DE(DATA, F, pop_size, max_iter, prop_mut)
    add_money_best, COST_best = de.run()
    end_time = time.time()
    print('程序运行了{:.3f}s'.format(end_time - start_time))

    print("1")
