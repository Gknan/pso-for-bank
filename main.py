'''
这是项目入口，
该项目主要包括三部分：
1、项目数据导入
2、ATM选择规划
3、车辆路径规划
4、项目输出展示
'''

import os
from data_tools import LoadData
from util import choose_ATM

PROJECT_ROOT = os.getcwd()
DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'DATA')

DATA = LoadData(DATA_FOLDER_PATH)

add_money_plan, COST = choose_ATM(DATA)

# ROUTE = plan_route(add_money_plan)
#
# ans_show(ROUTE)
