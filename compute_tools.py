'''
包括简单的计算相关的函数
'''

import numpy as np


def step_function(t):
    '''
    阶跃函数
    :param t: 变量，如果是矩阵，常数，数组
    :return: 返回对变量中每个元素的阶跃函数值
    '''
    return t > 0


def compute_clear_index(init_clear_index, add_money_data):
    '''
    计算清机系数
    :param init_clear_index: 初始清机系数
    :param add_money_data: 加钞表
    :return: 清机系数表
    '''
    ATM_number = add_money_data.shape[0]
    days = add_money_data.shape[1]

    clear_index = np.zeros((ATM_number, days))
    add_money_data = 1 - step_function(add_money_data)

    for t in range(days):
        if t == 0:
            clear_index[:, 0] = init_clear_index * add_money_data[:, 0] + add_money_data[:, 0]
        else:
            clear_index[:, t] = clear_index[:, t - 1] * add_money_data[:, t] + add_money_data[:, t]

    return clear_index


def compute_shortage_money(data_in, data_out, add_money_data, init_money):
    '''
    计算每天的余额及缺钞量
    :param data_in: 用户往atm存钱数据，shape = (atm_number, 8)
    :param data_out: 用户从atm取钱数据，shape同上
    :param add_money_data: 银行给atm加钞数据，shape同上
    :param init_money: 初始存款, shape = (atm_number,)
    :return:atm的余额，atm的缺钞数
    '''
    atm_number = data_in.shape[0]
    days = data_in.shape[1]
    shortage_money = np.zeros((atm_number, days))
    money = np.zeros((atm_number, days))
    add_money = step_function(add_money_data)
    no_add = 1 - add_money

    for t in range(days):
        if t == 0:
            shortage_money[:, t] = data_out[:, 0] - data_in[:, 0] - \
                                   init_money * no_add[:, 0] - add_money_data[:, 0] * add_money[:, 0]
            money[:, t] = -shortage_money[:, t] * (shortage_money[:, t] < 0)
            shortage_money[:, t] = shortage_money[:, t] * (shortage_money[:, t] > 0)
        else:
            shortage_money[:, t] = data_out[:, t] - data_in[:, t] - \
                                   add_money_data[:, t] * add_money[:, t] - money[:, t - 1] * no_add[:, t]
            money[:, t] = -shortage_money[:, t] * (shortage_money[:, t] < 0)
            shortage_money[:, t] = shortage_money[:, t] * (shortage_money[:, t] > 0)

    return money, shortage_money


def compute_cost_t(money, shortage_money, add_money_data, init_money,
                   clear_max, clear_index, translation_cost,
                   holding_rate, count_money_rate, translation_rate,
                   clear_critisizm, shortage_critisizm):
    '''
    计算每个解状态的cost_t值
    :param money:ATM机的余额，shape = (ATM_number, days)
    :param shortage_money:ATM机的缺钞额，shape = (ATM_number, days)
    :param add_money_data:给ATM机的加钞规划，shape = (ATM_number, days)
    :param init_money:ATM机的初始余额，shape = (ATM_number,)
    :param clear_max:ATM机的清机周期，shape = (ATM_number,)
    :param clear_index:ATM机的清机系数，shape = (ATM_number, days)
    :param translation_cost:每天的出行成本，shape = (days,)
    :param holding_rate:ATM机的占款比率，常数
    :param count_money_rate:点钞比率，常数
    :param translation_rate:运钞出行比率，常数
    :param clear_critisizm:清机惩罚常数
    :param shortage_critisizm:缺钞惩罚常数
    :return:每一台ATM机每一天的cost值
    '''

    ATM_number = add_money_data.shape[0]
    days = add_money_data.shape[1]

    cost_atm_t = np.zeros((ATM_number, days))
    if_add_money = add_money_data > 0

    for t in range(days):
        if t == 0:
            cost_atm_t[:, 0] = holding_rate * money[:, t] \
                               + if_add_money[:, t] * count_money_rate * (init_money + add_money_data[:, t]) \
                               + shortage_money[:, t] * shortage_critisizm \
                               + step_function(clear_index[:, t] - clear_max) * clear_critisizm
        else:
            cost_atm_t[:, t] = holding_rate * money[:, t] \
                               + if_add_money[:, t] * count_money_rate * (money[:, t - 1] + add_money_data[:, t]) \
                               + shortage_money[:, t] * shortage_critisizm \
                               + step_function(clear_index[:, t] - clear_max) * clear_critisizm

    cost_t = np.sum(cost_atm_t, axis=0) + translation_rate * translation_cost

    return cost_t


def compute_translation_cost(add_money_data, ATM_distance):
    '''
    用估计方式简单计算出行成本
    :param add_money_data:加钞表
    :param ATM_distance:各个ATM机之间的距离
    :return:每天的出行成本
    '''
    days = add_money_data.shape[1]

    vehicle_index = 0.05
    route_index = 0.05

    translation_cost = np.zeros(days)
    if_add_money = step_function(add_money_data)

    for t in range(days):
        add_atm_numer = np.sum(if_add_money[:, t])
        vehicle_cost = add_atm_numer * vehicle_index
        route_cost = np.sum(if_add_money[:, t] * ATM_distance[1:, 0]) * route_index
        translation_cost[t] = vehicle_cost + route_cost

    return translation_cost
