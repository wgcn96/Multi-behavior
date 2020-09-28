# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/11/5'

if __name__ == '__main__':
    import pandas as pd
    data_sample = pd.read_csv('data_sample.csv', usecols=['user_id', 'item_id', 'time_stamp', 'action_type'])
    user_group = data_sample.groupby(['user_id', 'time_stamp'])


    def account(df, action_type):
        total = len(df)
        count = len(df[df[action_type] == 2])
        return count / total


    result = user_group.apply(account, action_type='action_type')
    result.index.values
