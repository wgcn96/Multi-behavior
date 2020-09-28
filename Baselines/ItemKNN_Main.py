# -*- coding: utf-8 -*-

"""
main file for ItemKNN

"""

__author__ = 'Wang Chen'
__time__ = '2019/12/3'

import numpy as np
import math

from Dataset import *

np.random.seed(1234)


def _cos_similarity(a, b):
    assert (np.sum(a & b) / (math.sqrt(np.sum(a)) * math.sqrt(np.sum(b)))) == (np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return np.sum(a & b) / (math.sqrt(np.sum(a)) * math.sqrt(np.sum(b)))
    # return np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _generate_matrix(rate_matrix, similiarity_matrix, pos_list, neightbor_n):
    """
    根据k近邻生成结果矩阵。计算公式：weight_j* rate_j
    :param rate_matrix:  打分矩阵
    :param similiarity_matrix:  相似性矩阵
    :param pos_list:  需要计算KNN的user ID list
    :param neightbor_n:  n近邻
    :return:
    """
    argsort_matrix = np.argsort(-similiarity_matrix, axis=1)
    result_matrix = np.zeros(rate_matrix.shape, dtype=np.float32)
    for target_pos in pos_list:
        # 最相近的一定是自己，权重为1，因此取前[1,n+1)的元素。
        most_similiar_movie_list = argsort_matrix[target_pos, 1:neightbor_n + 1]

        weight_list = similiarity_matrix[most_similiar_movie_list, target_pos]
        neighbor_rate_matrix = rate_matrix[most_similiar_movie_list,].T
        weight_matrix = np.multiply(weight_list, neighbor_rate_matrix).T
        result_row = np.sum(weight_matrix, axis=0)
        result_matrix[target_pos] = result_row

    return result_matrix


def _eval_one_rating(arg_item_vector, gtItem):
    """
    测试一个 ground item 的结果，返回当前 hr 和 ndcg
    :param arg_item_vector: 当前打分下的item降序排列
    :param gtItem: ground truth item 的位置
    :return:
    """
    hr = 0
    ndcg = 0

    # 计算打分
    if gtItem in arg_item_vector:
        hr = 1
        pos = np.where(arg_item_vector == gtItem)[0]
        ndcg = math.log(2) / math.log(pos + 2)

    return hr, ndcg


if __name__ == '__main__':
    neighbor_k = 10
    result_k = 100
    print('begin loading matrix...')
    path = '/home/wangchen/multi-behavior/ijaci15/sample_version_one/'
    # 'http://172.18.166.184:8099/lab/tree/beibei/sample_version_one'
    dataset_buy = Dataset(path=path, b_type='buy', load_type='matrix')
    matrix = dataset_buy.trainMatrix.toarray().astype(np.int32).T
    print('load matrix finish! matrix shape: ', matrix.shape)

    num_users, num_items = dataset_buy.num_users, dataset_buy.num_items

    print('begin calculate similarity matrix...')
    similarity_matrix = np.eye(num_items)
    for i in range(num_items):
        print('current item {}'.format(i))
        for j in range(i + 1, num_items, 1):
            similarity_ij = _cos_similarity(matrix[i], matrix[j])
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity_ij
    print('calculate similarity matrix finish! matrix shape: ', similarity_matrix.shape)
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    # np.save('item_similarity_matrix.npy', similarity_matrix)

    # similarity_matrix = np.load('item_similarity_matrix.npy')

    print('begin generate knn result...')
    reuslt_matrix = _generate_matrix(matrix, similarity_matrix, np.arange(num_items), neighbor_k).T
    # result_matrix = result_matrix
    print('generate result finish, matrix shape: ', reuslt_matrix.shape)

    for result_k in [50, 80, 100, 200]:
        print('begin calculate HR NDCG result...')
        hits, ndcgs = [], []
        arg_sort_matrix = np.argsort(-reuslt_matrix, axis=1)[:, :result_k]
        test_ratings_list = dataset_buy.testRatings
        for test_pair in test_ratings_list:
            user_id, gtItem = test_pair[0], test_pair[1]
            arg_sort_vec = arg_sort_matrix[user_id]
            cur_hr, cur_ndcg = _eval_one_rating(arg_sort_vec, gtItem)
            hits.append(cur_hr)
            ndcgs.append(cur_ndcg)
        HR, NDCG = np.array(hits).mean(), np.array(ndcgs).mean()
        print('all finish! reuslt: {}, {}'.format(HR, NDCG))
