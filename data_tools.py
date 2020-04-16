'''
有关数据处理方面的问题，
需要将所需要加载的数据放到DATA文件下
包括：
1、预测ATM机8天的存取款量、
2、初始状态设置（各种系数的设置）
3、ATM机的位置信息设置
'''

import os
import numpy as np
import pandas as pd
from compute_tools import compute_clear_index, compute_cost_t, compute_shortage_money, compute_translation_cost


class LoadData():
    def __init__(self, data_folder_root):
        self.data_folder_root = data_folder_root

        # 用户存款  shape = (ATM_number, days)  header = None 否则默认第一行作为标头
        self.data_in = np.array(pd.read_csv(os.path.join(self.data_folder_root, 'IN_DATA.csv'), dtype=int, header=None))
        # 用户取款  shape = (ATM_number, days)
        self.data_out = np.array(pd.read_csv(os.path.join(self.data_folder_root, 'OUT_DATA.csv'), dtype=int, header=None))

        # ATM初始余额   shape = (ATM_number,)
        self.init_money = np.array(pd.read_csv(os.path.join(self.data_folder_root, "money.csv"), dtype=int, header=None)).flatten()
        # ATM初始清机系数 shape = (ATM_number,)
        self.init_clear_index = np.array(pd.read_csv(os.path.join(self.data_folder_root, 'clear_index.csv'), dtype=int, header=None)).flatten()
        # ATM最大加钞数  shape = (ATM_number,)
        self.max_add_money = np.array(pd.read_csv(os.path.join(self.data_folder_root, 'max_add_money.csv'), dtype=int, header=None)).flatten()
        # 各ATM机的清机周期 shape = (ATM_number,)
        self.clear_max = np.array(pd.read_csv(os.path.join(self.data_folder_root, 'clear_max.csv'), dtype=int, header=None)).flatten()
        # TODO:ATM 位置或者是两两之间的距离矩阵, shape = (ATM_number + 1, ATM_number + 1) 0下标为现金中心
        self.ATM_distance = np.array(pd.read_csv(os.path.join(self.data_folder_root, 'distance.csv'), dtype=int, header=None))
        # self.ATM_distance = np.zeros((self.ATM_number + 1, self.ATM_number + 1))

        self.ATM_number = self.data_in.shape[0]
        self.days = self.data_in.shape[1]

        # 占款利率 常数可调整
        self.holding_rate = 0.05
        # 点钞比率 常数可调整
        self.count_money_rate = 0.0
        # 出行比率 常数可调整
        self.translation_rate = 0.0

        # 清机惩罚项常数，确定后最好不调整
        self.clear_critisizm = 0
        # 缺钞惩罚项常数，确定后最好不调整
        self.shortage_critisizm = 5

        self.load_success()

    def load_success(self):
        print("Load data sucessfully!!!")


class ATMData():

    def __init__(self, DATA, add_money_data):
        # super().__init__(data_folder_root)

        self.DATA = DATA
        # 每天的加钞规划，shape = (ATM_number, days)
        self.add_money_data = add_money_data

        self.COST = 0
        self.translation_cost = np.zeros(self.DATA.days)
        self.cost_t = np.zeros(self.DATA.days)
        self.money = np.zeros((self.DATA.ATM_number, self.DATA.days))
        self.clear_index = np.zeros((self.DATA.ATM_number, self.DATA.days))
        self.shortage_money = np.zeros((self.DATA.ATM_number, self.DATA.days))

    def update_clear_index(self):
        self.clear_index = compute_clear_index(self.DATA.init_clear_index, self.add_money_data)
        return self.clear_index

    def update_shortage_money(self):
        self.money, self.shortage_money = compute_shortage_money(self.DATA.data_in, self.DATA.data_out,
                                                                 self.add_money_data, self.DATA.init_money)
        return self.money, self.shortage_money

    def update_translation_cost(self, method=2):
        '''
        计算出行成本，两种方式，第一种直接计算，第二种估计
        :param method: 计算方式
        :return: 当天出行成本，shape = (days)
        '''
        if method == 1:
            # TODO 完整计算
            pass
        elif method == 2:
            # 估计计算
            self.translation_cost = compute_translation_cost(self.add_money_data, self.DATA.ATM_distance)
            return self.translation_cost
        else:
            print("方式设置出错！")

    def update_cost_t(self):
        self.cost_t = compute_cost_t(self.money, self.shortage_money, self.add_money_data,
                                     self.DATA.init_money, self.DATA.clear_max,
                                     self.clear_index,self.translation_cost,
                                     self.DATA.holding_rate, self.DATA.count_money_rate,
                                     self.DATA.translation_rate,
                                     self.DATA.clear_critisizm, self.DATA.shortage_critisizm)
        return self.cost_t

    def update_COST(self):
        self.update_clear_index()
        self.update_shortage_money()
        self.update_translation_cost()
        self.update_cost_t()
        self.COST = np.sum(self.cost_t)
        return self.COST


if __name__ == '__main__':
    DATA_ROOT = os.path.join(os.getcwd(), 'DATA')

    DATA = LoadData(DATA_ROOT)
    add_money = np.array(pd.read_csv(os.path.join(DATA_ROOT, 'add_money.csv'), dtype=int))
    ATM = ATMData(DATA, add_money)
    COST = ATM.update_COST()
    print('1')
