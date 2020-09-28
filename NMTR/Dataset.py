# -*- coding: utf-8 -*-

"""
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
"""

"""
reconstruct for multi-behavior dataset
Nov. 19, 2019
by Chen Wang  
"""

from time import time

import scipy.sparse as sp
import numpy as np
import random

random.seed(1234)
np.random.seed(1234)


class Dataset(object):
    """
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
    """

    def __init__(self, path, b_type=None, load_type='matrix'):
        """
        Constructor
        """
        self.b_type = b_type
        self.num_users, self.num_items = self.get_users_items_num(path)
        if load_type == 'matrix':
            self.trainMatrix = self.load_training_file_as_matrix(path)

        elif load_type == 'dict':
            self.trainDict, self.num_ipv, self.num_cart, self.num_buy,  = self.load_training_file_as_dict(path)
            max_rate = 0
            for u in self.trainDict.keys():
                rate = len(self.trainDict[u]['buy'])
                max_rate = max(rate, max_rate)
            self.max_rate = max_rate  # 用户最大的评分数

        self.testRatings = self.load_rating_file_as_list(path + "buy-test")

    def get_users_items_num(self, path): # TODO(wgcn96) : 行为数量不为三时的改动
        """
        watch through the three matrix and return the matrix length of users amd items
        :param path: the root path of the files
        :return: the max number of the users and items
        """

        filename_ipv = path + 'ipv-train'
        filename_cart = path + 'collect-train'
        filename_buy = path + 'buy-train'

        num_users_ipv, num_items_ipv = 0, 0
        num_users_cart, num_items_cart = 0, 0
        num_users_buy, num_items_buy = 0, 0

        with open(filename_ipv, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users_ipv = max(num_users_ipv, u)
                num_items_ipv = max(num_items_ipv, i)
                line = f.readline()

        with open(filename_cart, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users_cart = max(num_users_cart, u)
                num_items_cart = max(num_items_cart, i)
                line = f.readline()

        with open(filename_buy, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users_buy = max(num_users_buy, u)
                num_items_buy = max(num_items_buy, i)
                line = f.readline()

                # Construct matrix
        return max(num_users_ipv, num_users_cart, num_users_buy) + 1, \
               max(num_items_ipv, num_items_cart, num_items_buy) + 1

    def load_training_file_as_matrix(self, path):
        """
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        """

        if self.b_type == 'ipv':
            path = path + 'ipv-train'
        elif self.b_type == 'cart':
            path = path + 'collect-train'
        elif self.b_type == 'buy':
            path = path + 'buy-train'

        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)

        with open(path, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        print("already load the trainMatrix...")

        return mat

    def load_training_file_as_dict(self, path):
        """
        load multiple behavior dict
        :param path: the root path of the files
        :return: the dict
                    key -- user
                    value -- a dict --- key : behavior type (ipv/cart/buy)
                                    --- value : items iteration
        """
        filename_ipv = path + 'ipv-train'
        filename_cart = path + 'collect-train'
        filename_buy = path + 'buy-train'

        num_ipv, num_cart, num_buy = (0, 0, 0)

        trainDict = {}
        for i in range(self.num_users):
            trainDict[i] = {}
            trainDict[i]['ipv'], trainDict[i]['cart'], trainDict[i]['buy'] = [], [], []

        with open(filename_ipv, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                num_ipv += 1
                arr = line.split("\t")
                u, i, r = int(arr[0]), int(arr[1]), int(arr[2])
                if r > 0:
                    trainDict[u]['ipv'].append(i)
                line = f.readline()

        with open(filename_cart, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                num_cart += 1
                arr = line.split("\t")
                u, i, r = int(arr[0]), int(arr[1]), int(arr[2])
                if r > 0:
                    trainDict[u]['cart'].append(i)
                line = f.readline()

        with open(filename_buy, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                num_buy += 1
                arr = line.split("\t")
                u, i, r = int(arr[0]), int(arr[1]), int(arr[2])
                if r > 0:
                    trainDict[u]['buy'].append(i)
                line = f.readline()

        for i in trainDict.keys():
            if len(trainDict[i]['ipv']) == 0 and \
                    len(trainDict[i]['cart']) == 0 and \
                    len(trainDict[i]['buy']) == 0:
                # print('we will delete user {}'.format(i))
                del trainDict[i]

        print("already load the trainDict, the dict length {}".format(len(trainDict.keys())))
        return trainDict, num_ipv, num_cart, num_buy

    def load_rating_file_as_list(self, filename):
        """
        load the rating file as a binary list [user, item].
        Note it is a leave one out evaluation.
        :param filename:
        :return:
        """
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        print("already load the testRatings, the test rating length {}".format(len(ratingList)))
        return ratingList
