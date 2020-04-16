import pandas as pd
import numpy as np
import os

DATA_ROOT = os.path.join(os.getcwd(), 'DATA')
DATA_NAME = os.path.join(DATA_ROOT, os.listdir(DATA_ROOT)[0])

def load_data(DATA_NAME):
    DATA = pd.read_csv(DATA_NAME, encoding='gbk')
    return DATA


def find_data(DATA):
    data = None
    atm_index_list = DATA['设备编号'].unique()
    # print(len(atm_index_list))
    for atm_index in atm_index_list:
        data_i = DATA[DATA['设备编号'].isin([atm_index])]
        data_i = data_i.iloc[0:7, :]
        if data is None:
            data = data_i
        else:
            data = pd.concat([data, data_i], axis=0)
    data.reset_index(drop=True, inplace=True)
    return data


def tranlate_data(data):
    data = np.array(data.drop(['流水日期', '设备编号'], axis=1))
    data = data.reshape((-1, 7))
    return data


def save_data(data_t, label):
    pd_data = pd.DataFrame(data_t)
    # print(pd_data)
    # pd_data.to_csv(os.path.join(DATA_ROOT, label), index=False)
    pd_data.to_csv(os.path.join(DATA_ROOT, label), index=False, header=False)

def create_data_out(DATA_NAME):
    DATA = load_data(DATA_NAME)
    data = find_data(DATA)
    data_t = tranlate_data(data)
    save_data(data_t, 'OUT_DATA.csv')

def create_data_out2():
    data_out = np.zeros((46, 7), dtype=int) + 200000
    save_data(data_out, 'OUT_DATA.csv')

def create_data_in():
    data_in = np.zeros(46, dtype=int)
    save_data(data_in, 'IN_DATA.csv')


def create_money():
    money = np.zeros(46, dtype=int)
    money = money.astype(int)
    save_data(money, 'money.csv')


def create_clear_index():
    clear_index = np.random.randint(0, 7, size=72)
    save_data(clear_index, 'clear_index.csv')


# def create_max_add_money():
#     max_add_money = np.zeros(46, dtype=int) + 1200000
#     save_data(max_add_money, 'max_add_money.csv')


def create_clear_max():
    clear_max = np.zeros(46, dtype=int) + 7
    save_data(clear_max, 'clear_max.csv')


# def create_distance():
#     distance = np.zeros((47, 47), dtype=int) + 200
#     save_data(distance, 'distance.csv')

# def create_add_money():
#     add_money = np.zeros((46, 7), dtype=int)
#     save_data(add_money, 'add_money.csv')

if __name__ == '__main__':
    # create_data_out(DATA_NAME)
    create_data_out2()
    # create_data_in()
    # create_money()
    # create_max_add_money()
    # create_distance()
    # create_clear_max()
    # create_clear_index()
    # create_add_money()

