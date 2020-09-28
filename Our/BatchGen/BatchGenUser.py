# -*- coding: utf-8 -*-

"""
sampling the data, functions for single behavior model and multiple behavior model

for single behavior model: sampling and shuffling
for multiple behavior model: sampling_3 and shuffling_3

original author : Chen Gao
reconstructed by Chen Wang
Nov. 19th, 2019
"""

import numpy as np
from time import time
import random

random.seed(1234)
np.random.seed(1234)

_user_input = None
_item_input = None
_item_input_neg = None
_item_rate = None
_item_num = None
_labels = None
_labels_3 = None
_batch_size = None
_index = None
_dataset_ipv = None
_dataset_cart = None
_dataset_buy = None
_dataset = None


# input: dataset(Mat, List, Rating), batch_choice, num_negatives
# output: [_user_input_list, _item_input_list, _labels_list]
def sampling(args, dataset, num_negatives):
    if args.model == "FISM":
        _user_input, _item_input, _item_rate, _item_num, _labels = [], [], [], [], []
        num_users, num_items = dataset.num_users, dataset.num_items
        for u in dataset.trainDict.keys():
            item_rate = dataset.trainDict[u]['buy']
            item_num = len(item_rate)  # all rated items of user u
            for i in item_rate:
                item_rate_2 = filter(lambda x: x != i, item_rate)
                _user_input.append(u)
                _item_input.append(i)
                _item_rate.append(item_rate_2)
                _item_num.append(item_num - 1)
                _labels.append(1)
                # negative instances
                for t in range(num_negatives):
                    j = np.random.randint(num_items)
                    while j in item_rate:
                        j = np.random.randint(num_items)
                    _user_input.append(u)
                    _item_input.append(j)
                    _item_rate.append(item_rate)
                    _item_num.append(item_num)
                    _labels.append(0)

        # additional manipulation on _item_rate
        max_rate = max(map(lambda x: len(x), _item_rate))
        _item_rate_fixed = []
        for i in _item_rate:
            _item_rate_fixed.append(i + [num_items] * (max_rate - len(i)))

        return _user_input, _item_input, _item_rate_fixed, _item_num, _labels

    elif args.model == 'MF' and args.loss_func == 'BPR':
        _user_input, _item_input, _item_input_neg = [], [], []
        num_users, num_items = dataset.num_users, dataset.num_items
        for (u, i) in dataset.trainMatrix.keys():
            _user_input += [u] * (num_negatives + 1)
            _item_input += [i] * (num_negatives + 1)
            for _ in range(num_negatives + 1):
                j = np.random.randint(num_items)
                while (u, j) in dataset.trainMatrix.keys():
                    j = np.random.randint(num_items)
                _item_input_neg.append(j)
        print(len(_user_input), len(_item_input), len(_item_input_neg))
        return _user_input, _item_input, _item_input_neg

    else:
        _user_input, _item_input, _labels = [], [], []
        num_users, num_items = dataset.num_users, dataset.num_items

        if args.en_MC == 'yes':

            # TODO(gc): enable multi-channel sampling for single behavior models

            for u in dataset.trainDict.keys():
                log = dataset.trainDict[u]
                """
                for i in log['buy']:
                    _user_input += [u] * (num_negatives + 1)
                    _item_input += [i]
                    _labels += [1]
                    # uo_neg_prob = args.beta
                    uo_neg_prob = 1
                    cart_weight, ipv_weight = len(log['cart']) / 2.0, len(log['ipv']) / 3.0
                    # ipv_neg_prob = (1 - args.beta) * ipv_weight / (cart_weight + ipv_weight)+0.1
                    ipv_neg_prob = (1 - args.beta) * ipv_weight / (cart_weight + ipv_weight + 1)
                    _item_input += _sample_neg_accord_prob(log=log, num_negatives=num_negatives, num_items=num_items, prob=[uo_neg_prob, ipv_neg_prob])
                    _labels += [0] * num_negatives
                    """

                for i in log['buy']:
                    _user_input += [u] * (num_negatives + 1)
                    _item_input += [i]
                    _labels += [1]

                    cart_num, ipv_num = len(log['cart']) / 2.0, len(log['ipv']) / 3.0

                    if not (cart_num+ipv_num):
                        _item_input += _sample_unobserved(unob_sample_num=num_negatives, num_items=num_items, log=log)
                        _labels += [0] * num_negatives
                        continue

                    neg_list = []
                    for _ in range(num_negatives):
                        random_proba = random.random()
                        if random_proba <= args.beta:       # sample from unobserved items
                            neg_item_sample = _sample_unobserved(unob_sample_num=1, num_items=num_items, log=log)[0]
                        else:           # sample from observed items
                            random_proba = random.random()
                            if random_proba <= cart_num/(cart_num+ipv_num):     # sample from 'cart'
                                neg_item_sample = random.sample(log['cart'], 1)[0]
                                while neg_item_sample in log['buy']:
                                    neg_item_sample = random.sample(log['cart'], 1)[0]
                            else:                                               # sample from 'ipv'
                                neg_item_sample = random.sample(log['ipv'], 1)[0]
                                while neg_item_sample in log['buy']:
                                    neg_item_sample = random.sample(log['ipv'], 1)[0]
                        neg_list.append(neg_item_sample)

                    _item_input += neg_list
                    _labels += [0] * num_negatives

        else:
            for (u, i) in dataset.trainMatrix.keys():
                # positive instance
                _user_input.append(u)
                _item_input.append(i)
                _labels.append(1)
                # negative instances
                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while (u, j) in dataset.trainMatrix.keys():
                        j = np.random.randint(num_items)
                    _user_input.append(u)
                    _item_input.append(j)
                    _labels.append(0)
        assert (len(_user_input) == len(_item_input)) and (len(_user_input) == len(_labels))
        # print("total sample number: {}".format(len(_user_input)))
        return _user_input, _item_input, _labels


def shuffle(samples, batch_size, args):
    global _user_input
    global _item_input
    global _item_input_neg
    global _item_rate
    global _item_num
    global _labels
    global _batch_size
    global _index

    if args.model == 'FISM':
        _user_input, _item_input, _item_rate, _item_num, _labels = samples
        _batch_size = batch_size
        _index = list(range(len(_user_input)))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size

        print('num_batch:%d, all_entries:%d, batch_size:%d, labels:%d' % (
            num_batch, len(_user_input), _batch_size, len(_labels)))
        t1 = time()

        user_list = []
        item_list = []
        item_rate_list = []
        item_num_list = []
        labels_list = []
        for i in range(num_batch):
            user, item, item_rate, item_num, label = _get_train_batch_FISM(i)
            user_list.append(user)
            item_list.append(item)
            item_rate_list.append(item_rate)
            item_num_list.append(item_num)
            labels_list.append(label)

        print('shuffle time: %d' % (time() - t1))
        return user_list, item_list, item_rate_list, item_num_list, labels_list

    elif args.model in ['pure_GMF', 'pure_MLP', 'pure_NCF'] or (args.model == 'MF'and args.loss_func != 'BPR'):
        _user_input, _item_input, _labels = samples
        _batch_size = batch_size
        _index = list(range(len(_labels)))
        np.random.shuffle(_index)
        num_batch = len(_labels) // _batch_size
        user_list = []
        item_list = []
        labels_list = []
        for i in range(num_batch):
            user, item, label = _get_train_batch(i)
            user_list.append(user)
            item_list.append(item)
            labels_list.append(label)

        return user_list, item_list, labels_list

    else:       # TODO(wgcn96): BPR related? but never go on here!
        #  TODO(wgcn96): for BPR-MF
        _user_input, _item_input, _item_input_neg = samples
        _batch_size = batch_size
        _index = list(range(len(_user_input)))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size
        user_list = []
        item_list = []
        labels_list = []
        for i in range(num_batch):
            user, item, label = _get_train_batch_BPR(i)
            user_list.append(user)
            item_list.append(item)
            labels_list.append(label)
        return user_list, item_list, labels_list

'''
def _get_dataset_behavior_proba(dataset):

    proba_list = []

    total_items_count = dataset.num_buy + 1/2*dataset.num_cart + 1/3*dataset.num_ipv

    proba_list.append(dataset.num_buy/total_items_count)
    proba_list.append(1/2*dataset.num_cart/total_items_count)
    proba_list.append(1/3*dataset.num_ipv/total_items_count)

    return proba_list
'''


'''
def _get_user_behavior_proba(user_log, behavior_list):
    """
    calculate the MC-BPR weight according to the original paper
    :param user_log: all types of user iterations
    :param behavior_list: the behavior type list, note it should be listed in order e.g.[buy, collect, ipv]

    :return: [] a list for the probability of the same length with behavior_list
    """
    beha_count = 0
    total_items_count = 0
    proba_list = []
    for beha in behavior_list:
        beha_count += 1
        cur_beha_count = (1/beha_count)*len(user_log[beha])
        total_items_count += cur_beha_count
        proba_list.append(cur_beha_count)

    for i in range(beha_count):
        proba_list[i] = proba_list[i]/total_items_count

    return proba_list
'''


def _sample_neg_accord_prob(prob, log, num_negatives, num_items):
    [uo_neg_prob, ipv_neg_prob] = prob
    neg = []
    for _ in range(num_negatives):
        j = random.random()
        if j < uo_neg_prob:
            neg += _sample_unobserved(1, num_items, log)
        elif j < uo_neg_prob + ipv_neg_prob:
            tem = random.sample(log['ipv'], 1)[0]
            while (tem in neg) or (tem in log['buy']) or (tem in log['cart']):
                tem = random.sample(log['ipv'], 1)[0]
            neg.append(tem)
            # neg += random.sample(log['ipv'], 1)
        else:
            tem = random.sample(log['cart'], 1)[0]
            while (tem in neg) or (tem in log['buy']):
                tem = random.sample(log['cart'], 1)[0]
            neg.append(tem)
            # neg += random.sample(log['cart'], 1)

    return neg


def _sample_unobserved(unob_sample_num, num_items, log):
    neg_uo = []
    for _ in range(unob_sample_num):
        j = np.random.randint(num_items)
        while j in log['buy'] or j in log['cart'] or j in log['ipv']:
            j = np.random.randint(num_items)
        neg_uo.append(j)
    return neg_uo


# dataset [dataset_ipv, dataset_cart, dataset_buy]
def sampling_3(args, dataset, num_negatives):
    start_time = time()
    _user_input, _item_input, _item_input_neg, _labels_3 = [], [], [], []

    if args.model == "Multi_BPR":
        num_users, num_items = dataset.num_users, dataset.num_items
        if args.neg_sample_tech == 'fix':
            for u in dataset.trainDict.keys():
                log = dataset.trainDict[u]
                for i in log['buy']:
                    neg_cart_num = min(num_negatives, len(log['cart']))
                    neg_ipv_num = min(int(num_negatives / 2), len(log['ipv']))
                    neg_uo_num = int(num_negatives / 4)
                    _user_input += [u] * (neg_cart_num + neg_ipv_num + neg_uo_num)
                    _item_input += [i] * (neg_cart_num + neg_ipv_num + neg_uo_num)
                    _item_input_neg += random.sample(log['cart'], neg_cart_num)
                    _item_input_neg += random.sample(log['ipv'], neg_ipv_num)
                    _item_input_neg += _sample_unobserved(neg_uo_num, num_items, log)

                for i in log['cart']:
                    neg_ipv_num = min(num_negatives, len(log['ipv']))
                    neg_uo_num = int(num_negatives / 2)
                    _user_input += [u] * (neg_ipv_num + neg_uo_num)
                    _item_input += [i] * (neg_ipv_num + neg_uo_num)
                    _item_input_neg += random.sample(log['ipv'], neg_ipv_num)
                    _item_input_neg += _sample_unobserved(neg_uo_num, num_items, log)

                for i in log['ipv']:
                    neg_uo_num = int(num_negatives)
                    _user_input += [u] * neg_uo_num
                    _item_input += [i] * neg_uo_num
                    _item_input_neg += _sample_unobserved(neg_uo_num, num_items, log)

        elif args.neg_sample_tech == 'prob':
            for u in dataset.trainDict.keys():
                log = dataset.trainDict[u]
                for i in log['buy']:
                    _user_input += [u] * num_negatives
                    _item_input += [i] * num_negatives
                    uo_neg_prob = args.beta
                    cart_weight, ipv_weight = len(log['cart']) / 2.0, len(log['ipv']) / 4.0
                    ipv_neg_prob = (1 - args.beta) * ipv_weight / (cart_weight + ipv_weight)
                    _item_input_neg += _sample_neg_accord_prob([uo_neg_prob, ipv_neg_prob], log, num_negatives,
                                                               num_items)

                for i in log['cart']:
                    _user_input += [u] * num_negatives
                    _item_input += [i] * num_negatives
                    uo_neg_prob = args.beta
                    ipv_neg_prob = 1 - args.beta
                    _item_input_neg += _sample_neg_accord_prob([uo_neg_prob, ipv_neg_prob], log, num_negatives,
                                                               num_items)

                for i in log['ipv']:
                    _user_input += [u] * num_negatives
                    _item_input += [i] * num_negatives
                    _item_input_neg += _sample_neg_accord_prob([1, 0], log, num_negatives, num_items)

        return _user_input, _item_input, _item_input_neg

    elif args.model == "BPR":
        num_users, num_items = dataset.num_users, dataset.num_items
        for u in dataset.trainDict.keys():
            log = dataset.trainDict[u]['buy'] + dataset.trainDict[u]['cart'] + dataset.trainDict[u]['ipv']
            _user_input += [u] * num_negatives * len(log)
            for i in log:
                _item_input += [i] * num_negatives
                for _ in range(num_negatives):
                    j = np.random.randint(num_items)
                    while j in log:
                        j = np.random.randint(num_items)
                    _item_input_neg.append(j)
        # print(len(_user_input), len(_item_input), len(_item_input_neg))
        return _user_input, _item_input, _item_input_neg

    else:
        # load data and create matrix
        _dataset_ipv, _dataset_cart, _dataset_buy = dataset
        num_users, num_items = _dataset_ipv.trainMatrix.shape
        if args.b_num == 3:
            _dataset = _dataset_ipv.trainMatrix + _dataset_cart.trainMatrix + _dataset_buy.trainMatrix  # combining 3 behaviors in one matrix, TODO(wgcn96): generate by iterating three matrices
        else:
            if args.b_2_type == 'vc':
                _dataset = _dataset_ipv.trainMatrix + _dataset_cart.trainMatrix
            elif args.b_2_type == 'cb':
                _dataset = _dataset_cart.trainMatrix + _dataset_buy.trainMatrix
            else:
                _dataset = _dataset_ipv.trainMatrix + _dataset_buy.trainMatrix

        print('num_users: %d \nnum_items: %d \nnum_ipv: %d \nnum_cart: %d \nnum_buy: %d \nnum_all %d'
              % (num_users, num_items, _dataset_ipv.trainMatrix.nnz, _dataset_cart.trainMatrix.nnz,
                 _dataset_buy.trainMatrix.nnz, _dataset.nnz))

        # wrong code !
        # define mapping for 1-d label to 3-d label
        # label_map = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 1.0], 2: [0.0, 1.0, 1.0], 3: [1.0, 1.0, 1.0]}
        # for (u, i) in _dataset.keys():
        #     # positive instance
        #     _user_input.append(u)
        #     _item_input.append(i)
        #     _labels.append(int(_dataset[u, i]))
        #     # negative instances
        #     for t in range(num_negatives):
        #         j = np.random.randint(num_items)
        #         while (u, j) in _dataset.keys():
        #             j = np.random.randint(num_items)
        #         _user_input.append(u)
        #         _item_input.append(j)
        #         _labels.append(0)
        # _labels_sr = pd.Series(_labels)
        # _labels_3 = _labels_sr.map(label_map)
        # _labels_3 = _labels_3.tolist()
        # print('samples eg:%s, %s, %s, %s [%.1f]'
        #       %(_user_input[:10], _item_input[:10], _labels[:10], _labels_3[:10], time()-start_time))

        for (u, i) in _dataset.keys():
            # positive instance
            _user_input.append(u)
            _item_input.append(i)
            cur_label_3 = []
            for beha in range(len(dataset)-1, -1, -1):       # TODO(wgcn96): 他输入数据跟dataset的顺序是反的，哭！！！
                cur_label_3.append(dataset[beha].trainMatrix[u, i])
            _labels_3.append(cur_label_3)

            # negative instances
            for k in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in _dataset.keys():
                    j = np.random.randint(num_items)
                _user_input.append(u)
                _item_input.append(j)
                _labels_3.append([0] * len(dataset))

        return _user_input, _item_input, _labels_3


def shuffle_3(samples, batch_size, args):
    global _user_input
    global _item_input
    global _item_input_neg
    # global _labels
    global _labels_3
    global _batch_size
    global _index

    user_list = []
    item_list = []
    argv_list = []
    if args.model in ['Multi_GMF', 'Multi_MLP', 'Multi_NCF', 'CMF'] or 'Our' in args.model:
        _user_input, _item_input, _labels_3 = samples
        _batch_size = batch_size
        _index = list(range(len(_labels_3)))
        np.random.shuffle(_index)
        num_batch = len(_labels_3) // _batch_size

        for i in range(num_batch):
            user, item, argv = _get_train_batch_3(i)
            user_list.append(user)
            item_list.append(item)
            argv_list.append(argv)

    elif 'BPR' in args.model:
        _user_input, _item_input, _item_input_neg = samples
        _batch_size = batch_size
        _index = list(range(len(_user_input)))
        np.random.shuffle(_index)
        num_batch = len(_user_input) // _batch_size

        for i in range(num_batch):
            user, item, argv = _get_train_batch_BPR(i)
            user_list.append(user)
            item_list.append(item)
            argv_list.append(argv)

    # print('sample size is %d, first sample is [%d, %d, %s]' 
    #       %(batch_size, user_list[0][0], item_list[0][0], argv_list[0][0]))

    return user_list, item_list, argv_list


def _get_train_batch(i):
    user_batch, item_batch, labels_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        labels_batch.append(_labels[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(labels_batch)


def _get_train_batch_FISM(i):
    user_batch, item_batch, item_rate_batch, item_num_batch, labels_batch = [], [], [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        item_rate_batch.append(_item_rate[_index[idx]])
        item_num_batch.append(_item_num[_index[idx]])
        labels_batch.append(_labels[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(item_rate_batch), np.array(item_num_batch), np.array(
        labels_batch)


def _get_train_batch_BPR(i):
    user_batch, item_batch, item_neg_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        item_neg_batch.append(_item_input_neg[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(item_neg_batch)


def _get_train_batch_3(i):
    user_batch, item_batch, labels_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        labels_batch.append(_labels_3[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(labels_batch)

