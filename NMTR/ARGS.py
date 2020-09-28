# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/12/5'


class ARGS:
    def __init__(self, embed_size, lr, regs, loss_func, dropout, loss_coefficient):
        self.embed_size = embed_size
        self.lr = lr
        self.regs = regs
        self.loss_func = loss_func
        self.dropout = dropout
        self.loss_coefficient = loss_coefficient


if __name__ == '__main__':
    args = ARGS(embed_size=64,
                lr=0.01,
                regs='[0, 0]',
                loss_func='square_loss',
                dropout=None)
    print('args', args)
