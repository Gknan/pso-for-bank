import numpy as np
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

    def __init__(self, DATA, pop=40, max_iter=150, w=0.8, c1=0.5, c2=0.5):
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.max_iter = max_iter  # max iter

        self.DATA = DATA
        self.DATA_ROOT = self.DATA.data_folder_root

        # 上下界需要重新核对 这里的上届是 0，上界限是 最大进而，但是对于 速度而言，一般不能太快，一般是 -1~1 之间
        # 每次加钞完成后都要根据上下界进行排除，去掉加钞小于下界和大于上界的值
        self.xlb = np.zeros(self.DATA.ATM_number, dtype=int) # 加钞金额的下界
        self.xub = (self.DATA.max_add_money).astype(int) # 加钞金额的上界限

        self.dim = self.DATA.ATM_number  # # dimension of particles, which is the number of variables of func
        self.day = self.DATA.days # 第三个维度

        # self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        # self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        #
        # self.lb = -np.ones() if lb is None else np.array(lb)
        # self.ub = np.ones() if ub is None else np.array(ub)
        self.lb = -10000
        self.ub = 10000

        # X 应该和 V 在一个数量级，要不然，很容易导致问题
        # X 的开始一定要随机
        # self.X = np.zeros((self.pop, self.dim, self.day), dtype=int)  # X
        self.X = np.zeros((self.pop, self.dim, self.day))  # X
        # V 的初始化
        # self.V = np.zeros((self.pop, self.dim, self.day))  # V # speed of particles
        self.V = np.zeros((self.pop, self.dim, self.day))
        for i in range(self.pop):
            # x = np.zeros((self.dim, self.day), dtype=int)
            # 初始加钞金额
            # x = np.zeros((self.dim, self.day))
            # 这里的速度需要重新设置初始值
            # v_high = self.ub - self.lb
            # v = np.random.uniform(low=-v_high, high=v_high,
            #                       size=(self.dim, self.day))
            # 初始化速度在 lb 与 ub 之间的随机值
            v = np.random.uniform(low=self.lb, high=self.ub, size=(self.dim, self.day))

            # 缩小 x 的值，和 v 一个数量级或者只高 一个数量级
            # self.scale = self.xub - self.xlb # 0~10
            # self.scaleub = 10000  # 把 0~ scale 映射到 0~5 上    cur - 0/ scale = x - 0/ scalebu
            # 钞值单位按 万来看
            # 第 i 个种群 第 t 天的加钞金额
            # for t in range(self.day):
            #     x[:, t] = (self.xlb + np.random.random(self.dim) * (self.xub - self.xlb)) / self.scaleub

            # 初始加钞金额
            x = np.zeros((self.dim, self.day), dtype=int)
            # 机型不同，xlb 和 xub 不同，所以 xlb = [72 * 1]
            for j in range(self.dim): # 对于 第 i 台机器
                x[j] = np.random.uniform(low=self.xlb[j], high=self.xub[j], size=self.day)

            # X  50 * 72 * 7   x 72 * 7
            # self.X[i, :, :] = x
            self.X[i, :, :] = x
            # update V
            self.V[i, :, :] = v

        # self.COST = np.zeros(pop)
        # self.COST = np.zeros(pop)
        self.Y = self.cal_y()  # y = f(x) for all particles

        ## 因为 pBest 每个粒子都有一个历史的最值，所以 pBest 比 gBest  要高一维度
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((self.dim, self.day))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles # y 越小越好，初始为 正无穷
        self.gbest_y_hist = []  # gbest_y of every iteration

        # 初始化 gbest_x gbest_y
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
        ## 对于种群中每个个体，进行 V 公式的更新
        for i in range(self.pop): # 对于种群中每个个体，进行迭代
            # update random
            r1 = np.random.rand()  # r 使用 size_pop * 1
            r2 = np.random.rand()
            self.V[i] = self.w * self.V[i] + \
                     self.cp * r1 * (self.pbest_x[i] - self.X[i]) + \
                     self.cg * r2 * (self.gbest_x - self.X[i])

        # 速度更新上下界控制
        # 速度小于下界，选择下界；大于上界，选择上届
        # self.V = np.where(self.V < lb, lb, self.V) 上下界一放缩，V全部集中到 -100 和 +100
        # self.V = np.where(self.V < self.lb, self.lb, self.V)
        # self.V = np.where(self.V > self.ub, self.ub, self.V)

        for i in range(self.pop):
            for j in range(self.dim):
                for k in range(self.day):
                    if self.V[i, j, k] < self.lb:
                        self.V[i, j, k] = self.lb
                    if self.V[i, j, k] > self.ub:
                        self.V[i, j, k] = self.ub
        # r1 = np.random.rand(self.pop, self.dim)
        # r2 = np.random.rand(self.pop, self.dim)
        # self.V = self.w * self.V + \
        #          self.cp * r1 * (self.pbest_x - self.X) + \
        #          self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V

        for i in range(self.pop):
            for j in range(self.dim):
                for k in range(self.day):
                    if self.X[i,j,k] < self.xlb[j]:
                        self.X[i,j,k] = self.xlb[j]
                    if self.X[i,j,k] > self.xub[j]:
                        self.X[i,j,k] = self.xub[j]
        # for i in range(self.dim):
        #     # X 上下界控制 bound control
        #     # self.X = np.where(self.X < self.xlb[i], self.xlb[i], self.V)
        #     # self.X = np.where(self.X > self.xub[i], self.xub[i], self.V)
        #     self.X = np.where(self.X < self.xlb[i], self.xlb[i], self.X)
        #     self.X = np.where(self.X > self.xub[i], self.xub[i], self.X)
        # if self.has_constraints:
        #     self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = np.zeros(self.pop)
        # self.Y = self.func(self.X).reshape(-1, 1)

        for i in range(self.pop):
            self.Y[i] = ATMData(self.DATA, self.X[i, :, :]).update_COST()

        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        # 更新每个群体中的历史最佳值
        for i in range(self.pop):
            # 当前计算的 Y 比 历史的最佳  best_y 小，选择当前的 X 作为 pbest_X,柔则 不变
            self.pbest_x[i] = np.where(self.pbest_y[i] > self.Y[i], self.X[i], self.pbest_x[i])
            # 更新 同理
            self.pbest_y[i] = np.where(self.pbest_y[i] > self.Y[i], self.Y[i], self.pbest_y[i])

    def update_gbest(self):
        '''
        global best
        :return:
        '''

        # 更新全局 gbest，若 之前的全局最佳 y 大于 当前代数求得 Y 中的
        # 最小值，更新 全局 gBest_X gBest_y
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :, :].copy()
            self.gbest_y = self.Y.min()
            # self.gbest_x = self.X[self.Y.argmin(), :].copy()
            # self.gbest_y = self.Y.min()

    def recorder(self):
        '''
        记录结果
        :return:
        '''
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
            print('第{}次迭代结束，共{}次迭代'.format(iter_num, self.max_iter))

            print("当前最小的COST为:{:.2f}".format(self.gbest_y))
            print("-------------------------------------")
        # 选中的 X 带入代价函数计算
        return self

    # fit = run
if __name__ == '__main__':
    import os
    import time
    from data_tools import LoadData
    DATA_ROOT = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'DATA')

    DATA = LoadData(DATA_ROOT)

    pop = 50
    max_iter = 1000
    w = 0.8
    c1 = 0.5
    c2 = 0.5

    start_time = time.time()
    pso = PSO(DATA, pop, max_iter, w, c1, c2)
    obj = pso.run()
    end_time = time.time()
    print('程序运行了{:.3f}s'.format(end_time - start_time))
    print('全局最小的COST为:{:.2f}'.format(obj.gbest_y))

    # cur - 0/ scale = x - 0/ scalebu 映射会真实的值
    print('全局最小COST对应的加钞方案为:')
    print(obj.gbest_x)

    print("1")