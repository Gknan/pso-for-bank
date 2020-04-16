#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from sko.tools import func_transformer
import pandas as pd
from data_tools import ATMData

class PSO():
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """


    def create_pop(self):
        '''
        创建size = self.pop_size的初始数组
        加钞的原则按照ML原则来加，取加钞范围内的整数
        :return: 初始种群
        '''
        # X   的维度 是 ATM_number * days 所以 V 的维度也是这样的
        self.add_money_pop = np.zeros((self.pop_size, self.DATA.ATM_number, self.DATA.days), dtype=int)  # X

        # V 的初始化
        # self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        self.add_v_pop = np.zeros((self.pop_size, self.DATA.ATM_number, self.DATA.days), dtype=int)  # V

        for i in range(self.pop_size):
            add_money_data = np.zeros((self.DATA.ATM_number, self.DATA.days), dtype=int)
            # 初始化 V
            # self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))
            # add_v_data = np.zeros((self.DATA.ATM_number, self.DATA.days), dtype=int)

            # 这里的速度需要重新设置初始值
            v_high = self.ub - self.lb
            add_v_data = np.random.uniform(low=-v_high, high=v_high, size=(self.DATA.ATM_number, self.DATA.days))
            for t in range(self.DATA.days):
                add_money_data[:, t] = self.xlb + \
                                       np.random.random(self.DATA.ATM_number) * (self.xub - self.xlb)

                #       v_high = self.ub - self.lb
                #         self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))
                # add V data self.ub - self.lb   V 的值有正有负数
                # add_v_data[:, t] = self.lb + \
                #                        np.random.random(self.DATA.ATM_number) * (self.ub - self.lb)

                # add_money_data[:, t] = self.lb + np.random.randint(0, 121, self.DATA.ATM_number) * 10000
            self.add_money_pop[i, :, :] = add_money_data

            # update V
            self.add_v_pop[i, :, :] = add_v_data

            # cal fitness 只需要 X 的值就够了，不需要 V 的值
            self.COST[i] = ATMData(self.DATA, self.add_money_pop[i, :, :]).update_COST()

        generation_best_index = self.COST.argmin()
        self.generation_best_add_money.append(self.add_money_pop[generation_best_index, :, :].copy())
        self.generation_best_COST.append(self.COST[generation_best_index])
        self.all_history_COST.append(self.COST)

        return self.add_money_pop

    # def __init__(self, func, DATA, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):
    def __init__(self, DATA, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):

        self.DATA = DATA
        self.DATA_ROOT = self.DATA.data_folder_root

        # 上下界需要重新核对 这里的霞姐是 0，上界限是 最大进而，但是对于 速度而言，一般不能太快，一般是 -1~1 之间
        self.xlb = np.zeros(self.DATA.ATM_number, dtype=int) # 加钞金额的下界
        self.xub = (self.DATA.max_add_money).astype(int) # 加钞金额的上界限

        self.dim = self.DATA.ATM_number  # 初始速度的下界限

        self.COST = np.zeros(pop)

        self.generation_best_add_money = [] # ptestx
        self.generation_best_COST = [] # pbesty

        self.all_history_COST = [] # gbest_y of every iteration
        self.history_best_add_money = [] #gbestx
        self.history_best_COST = [] #gbesty

        # self.func = func_transformer(func)
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb) # 初始速度的下界限
        self.ub = np.ones(self.dim) if ub is None else np.array(ub) # 速度的上界限
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        # 速度和位置都是在一定范围内 随机初始化的；计算的适应度函数是我们的 COST 函数表达式
        # 更新 X 根据的是
        # 流程是，
        # 1，初始化种群以及对应的 V X
        # 2, 计算适应度函数对应的值，我们选的是 COST 最小的，去 COST 最小对应的 个体 的 X 作为pBest
        # 3，根据 pBest 对应的 COST 值更新 gBest，就是 cost(pBest) > cost(gbest) 时，gBest = pBest
        # 4, 进入下一代，根据 v2= w*v1 + c * r1 * (pBest1 - x1) + c2 * r2 * (gBest - x1) 更新每个个体的 V
        # 5，根据 X2 = X1 + V1 更新每个个体第二代的 X；得到 V X 后，X 值带入 COST 函数，计算每个个体对应的
        # COST值，COST 值最小对应的 X 是的是当代的 pBest 再根据当代的 pBest 更新 gBest
        # self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim)) # 位置
        # v_high = self.ub - self.lb
        # self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        # self.Y = self.cal_y()  # y = f(x) for all particles
        # self.pbest_x = self.X.copy()  # personal best location of every particle in history
        # self.pbest_y = self.Y.copy()  # best image of every particle in history
        # self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        # self.gbest_y = np.inf  # global best y for all particles
        # self.gbest_y_hist = []  # gbest_y of every iteration
        # self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

        self.create_pop()

    def update_V(self):
        '''
        v2= w*v1 + c * r1 * (pBest1 - x1) + c2 * r2 * (gBest - x1)
        :return:

        V 的维度 (self.pop_size, self.DATA.ATM_number, self.DATA.days)
        '''
        r1 = np.random.rand(self.pop, self.dim)  # r 使用 size_pop * 1
        r2 = np.random.rand(self.pop, self.dim)
        # V 的更新公式，直接可用
        # self.V = self.w * self.V + \
        #          self.cp * r1 * (self.pbest_x - self.X) + \
        #          self.cg * r2 * (self.gbest_x - self.X)

        self.add_v_pop = self.w * self.add_v_pop + \
                 self.cp * r1 * (self.generation_best_add_money - self.add_money_pop) + \
                 self.cg * r2 * (self.history_best_add_money - self.add_money_pop)

    def update_X(self):
        # 这里的 X 就是 self.add_money_pop
        # self.X = self.X + self.V
        self.add_money_pop = self.add_money_pop + self.add_v_pop


        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        # self.Y = self.func(self.X).reshape(-1, 1)
        # return self.Y
        # self.Y = self.func(self.X).reshape(-1, 1)
        # return self.Y

        COST_X = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            COST_X[i] = ATMData(self.DATA, self.add_money_pop[i, :, :]).update_COST()

        return self.COST

        # return self.Y


    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        # self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        # self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

        self.generation_best_add_money = np.where(self.generation_best_COST > self.COST, self.add_money_pop, self.generation_best_add_money)
        self.generation_best_COST = np.where(self.generation_best_COST > self.COST, self.COST, self.generation_best_COST)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        # if self.gbest_y > self.Y.min():
        #     self.gbest_x = self.X[self.Y.argmin(), :].copy()
        #     self.gbest_y = self.Y.min()
        if self.history_best_COST > self.COST.min():
            self.history_best_add_money = self.add_money_pop[self.COST.argmin(), :].copy()
            self.history_best_COST = self.COST.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.add_money_pop)
        self.record_value['V'].append(self.add_v_pop)
        self.record_value['Y'].append(self.COST)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

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
            # self.gbest_y_hist.append(self.gbest_y)
        # return self

if __name__ == '__main__':
    import os
    import time
    from data_tools import LoadData
    DATA_ROOT = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'DATA')

    DATA = LoadData(DATA_ROOT)

    # DATA, pop = 40, max_iter = 150, lb = None, ub = None, w = 0.8, c1 = 0.5, c2 = 0.5


    pop = 50
    max_iter = 150
    w = 0.8
    c1 = 0.5
    c2 = 0.5

    start_time = time.time()
    pso = PSO(DATA, pop, max_iter, w, c1, c2)
    add_money_best, COST_best = pso.run()
    end_time = time.time()
    print('程序运行了{:.3f}s'.format(end_time - start_time))

    print("1")