import numpy as np
from OPT_ALG.DE import DE

import custom_pso as pso

def choose_ATM(DATA):
    '''
    用于加载数据后，选择需要加钞的ATM机
    :param DATA:加载初始数据类对象
    :return:一个合理的加钞表
    '''
    ATM_number = DATA.ATM_number
    days = DATA.days
    F = 0.5
    pop_size = 50
    max_iter = 200
    prop_mut = 0.3

    de = DE(DATA, F, pop_size, max_iter, prop_mut)


    # add_money_best, COST_best = de.run()

    add_money_best, COST_best = pso.run()

    return add_money_best, COST_best
