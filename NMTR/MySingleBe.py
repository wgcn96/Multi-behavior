# -*- coding: utf-8 -*-

"""
note something here

单线程重构主函数，MLP结构
"""

__author__ = 'Wang Chen'
__time__ = '2019/11/13'

import os
import math
import logging
from time import time, sleep
from time import strftime
from time import localtime
import argparse
import pickle as pkl
import setproctitle

import numpy as np

from Models import *
from BasicMF import *
import BatchGen.BatchGenUser as BatchUser
import Evaluate.EvaluateUser as EvalUser
from Dataset import Dataset

# compatiable version for tensorflow v1
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSGAN.")
    parser.add_argument('--dataset', nargs='?', default='ali',
                        help='Choose a dataset: bb1, bb2, bb3, ali, kaggle')
    parser.add_argument('--model', nargs='?', default='pure_MLP',
                        help='Choose model: GMF, MLP, FISM, Multi_GMF, Multi_BPR, BPR')
    parser.add_argument('--loss_func', nargs='?', default='logloss',
                        help='Choose loss: logloss, BPR, square_loss')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2048,
                        help='batch_size')
    parser.add_argument('--batch_size_ipv', nargs='?', type=int, default=2048,
                        help='batch_size_ipv')
    parser.add_argument('--batch_size_cart', nargs='?', type=int, default=512,
                        help='batch_size_cart')
    parser.add_argument('--batch_size_buy', nargs='?', type=int, default=256,
                        help='batch_size_buy')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed batch_size: generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type=int, nargs='?', default=0,
                        help='layer number')
    parser.add_argument('--regs', nargs='?', default='[0,0]',  # TODO(wgcn96): why not choose reg?
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of FISM')
    parser.add_argument('--train_loss', type=bool, default=True,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--process_name', nargs='?', default='Single_MLP@wangchen',
                        help='Input process name.')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU.')
    parser.add_argument('--evaluate', nargs='?', default='yes',
                        help='Evaluate or not.')
    parser.add_argument('--frozen', nargs='?', default='',
                        help='Frozen user or item')
    parser.add_argument('--optimizer', nargs='?', default='Adagrad',
                        help='Choose an optimizer')
    parser.add_argument('--add_fc', nargs='?', type=bool, default=True,
                        help='Add fully connected layer or not.')
    parser.add_argument('--plot_network', nargs='?', type=bool, default=False,
                        help='If choosing to plot network, the train will be skipped')
    parser.add_argument('--training_type', nargs='?', default='independent',
                        help='Choose type of training: independent or cascade')
    parser.add_argument('--pretrain', nargs='?', default='',
                        help='Load pre-trained vectors')
    parser.add_argument('--validate', nargs='?', type=bool, default=False,
                        help='Enable the calculation of validation losss during training')
    parser.add_argument('--batch_sample', nargs='?', type=bool, default=False,
                        help='generate batch samples for dataset')
    parser.add_argument('--data_gen', nargs='?', type=bool, default=False,
                        help='generate dataset or not')
    parser.add_argument('--topK', nargs='?', type=int, default=100,
                        help='topK for hr/ndcg')
    parser.add_argument('--frozen_type', nargs='?', type=int, default=0,
                        help='0:no_frozen, 1:item_frozen, 2:all_frozen')
    parser.add_argument('--loss_coefficient', nargs='?', default='[1/3,1/3,1/3]',
                        help='loss coefficient for Multi_GMF')
    parser.add_argument('--multiprocess', nargs='?', default='no',
                        help='Evaluate multiprocessingly or not')
    parser.add_argument('--trial_id', nargs='?', default='1',
                        help='Indicate trail id with same condition')
    parser.add_argument('--recover', nargs='?', default='no',
                        help='recover result from the server')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='Multi_BPR beta')
    parser.add_argument('--neg_sample_tech', nargs='?', default='prob',
                        help='Multi_BPR sample technique')
    parser.add_argument('--en_MC', nargs='?', default='no',
                        help='enable Multi-Channel for single behavior model')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='dropout keep_prob')
    parser.add_argument('--b_num', nargs='?', type=int, default=3,
                        help='control the behavior number in multitask learning')
    parser.add_argument('--b_2_type', nargs='?', default='cb',
                        help='when b_num=2, three condition: vc->view/cart, cb->cart/buy, vb->view/buy')
    return parser.parse_args()


def init_logging_and_result(args):
    global filename
    path_log = 'Log'
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    # define factors
    F_model = args.model
    F_dataset = args.dataset
    F_embedding = args.embed_size
    F_topK = args.topK
    F_layer_num = args.layer_num
    F_num_neg = args.num_neg
    F_trail_id = args.trial_id
    F_optimizer = args.optimizer + str(args.lr)
    F_loss_weight = args.loss_coefficient
    F_beta = args.beta
    F_alpha = args.alpha
    F_en_MC = args.en_MC
    F_dropout = args.dropout
    F_reg = args.regs
    F_b_num = args.b_num
    F_b_2_type = args.b_2_type

    if F_model not in ['pure_NCF', 'pure_MLP', 'Multi_NCF', 'Multi_MLP', 'GMF_FC', 'NCF_FC']:
        F_layer_num = 'X'
    if F_model not in ['Multi_MLP', 'Multi_NCF', 'Multi_GMF']:
        F_b_2_type = 'X'
    if F_model != 'Multi_BPR':
        F_dropout = 'X'
    if (F_model != 'Multi_BPR') and (F_en_MC != 'yes'):
        F_beta = 'X'
    if F_num_neg == 4:
        F_num_neg = 'D'
    if F_optimizer == 'Adagrad0.01':
        F_optimizer = 'D'
    if F_loss_weight == '[1/3,1/3,1/3]':
        F_loss_weight = 'D'
    if F_model != 'FISM':
        F_alpha = 'X'

    filename = "log-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-b-%s-a-%s" % (
        F_model, F_dataset, F_embedding, F_topK, F_layer_num, F_num_neg, F_loss_weight,
        F_optimizer, F_trail_id, F_beta, F_dropout, F_reg, F_b_2_type, F_alpha)

    logging.basicConfig(filename=path_log + '/' + filename, level=logging.INFO)
    logging.info('Use Multiprocess to Evaluate: %s' % args.multiprocess)


def eval_from_saved_model(model, args):
    global hr_recover, ndcg_recover

    # import tensorflow as tf
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if args.model == 'Multi_GMF':
        EvalDict = EvalUser.init_evaluate_model(model, dataset[0])
    else:
        EvalDict = EvalUser.init_evaluate_model(model, dataset)

    hits, ndcgs = [], []
    loader = None

    eval_begin = time()
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        model_load_path = 'Model/' + filename
        loader = tf.train.import_meta_graph(model_load_path + '.meta', clear_devices=True)

    with tf.Session(graph=eval_graph, config=config) as sess:
        loader.restore(sess, model_load_path)

        for idx in range(len(EvalDict)):
            if args.model == 'Multi_GMF':
                gtItem = dataset[0].testRatings[idx][1]
                predictions = sess.run('loss/inference/score_buy:0', feed_dict=EvalDict[idx])
            else:
                gtItem = dataset.testRatings[idx][1]
                predictions = sess.run('loss/inference/output:0', feed_dict=EvalDict[idx])
            rank = 0
            rank_score = predictions[gtItem]
            for i in predictions:
                if i > rank_score:
                    rank += 1
            if rank < args.topK:
                hr_tmp = 1
                ndcg_tmp = math.log(2) / math.log(rank + 2)
            else:
                hr_tmp = 0
                ndcg_tmp = 0
            hits.append(hr_tmp)
            ndcgs.append(ndcg_tmp)

        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

        print("Final HR = %.4f, NDCG = %.4f" % (hr, ndcg))

        # save info to the server
        hr_recover, ndcg_recover = hr, ndcg


# used for single task learning
def training_batch(model, sess, batches, args, optimizer, behave_type=None):  # TODO(wgcn96): behave_type是控制多行为loss的参数
    loss_train = 0.0
    loss_train_all = 0.0

    if args.model == 'FISM':
        user_input, item_input, item_rate, item_num, labels = batches
        num_batch = len(batches[1])
        # print('shape of item rate: (%d, %d)' %(item_rate[0].shape[0], item_rate[0].shape[1]))
        # print('first item rate: %s' % item_rate[0])
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.item_rate: item_rate[i],
                         model.item_num: np.reshape(item_num[i], [-1, 1]),
                         model.labels: np.reshape(labels[i], [-1, 1])}
            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train

    elif args.model == 'MF' and args.loss_func == 'BPR':
        user_input, item_input, neg_item_input = batches
        num_batch = len(batches[0])
        for i in range(num_batch):
            feed_dict = {model.user_input: user_input[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.item_input_neg: neg_item_input[i][:, None]}
            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train

    else:
        loss_no_reg_all = 0.0
        loss_reg_all = 0.0
        loss_combined_all = 0.0

        user_input, item_input, labels = batches
        # print(user_input, item_input, labels)
        num_batch = len(batches[1])
        for i in range(len(labels)):
            # print(user_input[i].shape)
            feed_dict = {model.user_input: user_input[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.labels: labels[i][:, None]}
            if behave_type == 'ipv':
                _, loss_train = sess.run([optimizer, model.loss1], feed_dict)
            elif behave_type == 'cart':
                _, loss_train = sess.run([optimizer, model.loss2], feed_dict)
            elif behave_type == 'buy':
                _, loss_train = sess.run([optimizer, model.loss3], feed_dict)
            else:
                _, loss_train = sess.run([optimizer, model.loss], feed_dict)

            if args.model == 'pure_GMF':
                loss_no_reg, loss_reg, loss_combined = sess.run([model.loss_no_reg, model.loss_reg, model.loss],
                                                                feed_dict)
                loss_reg_all += loss_reg
                loss_no_reg_all += loss_no_reg
                loss_combined_all += loss_combined

            loss_train_all += loss_train

        if args.model == 'pure_GMF':
            print('loss_tarin:{}'.format(loss_train))
            print('num of batch:{}'.format(num_batch))
            print('loss_combined:{}'.format(loss_combined_all / num_batch))
            print('loss_no_reg:%.6f, loss_reg:%.6f' % (loss_no_reg_all / num_batch, loss_reg_all / num_batch))

    return loss_train_all / num_batch


def training(model, args, behave_type=None, base_epoch=0, save=True):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    global pool, dataset, eval_queue, job_num, loss_list, hr_list, ndcg_list

    # initialize for Evaluate
    EvalDict = EvalUser.init_evaluate_model(model, dataset, args)

    with model.g.as_default():

        with tf.name_scope('optimizer'):
            if args.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(model.loss)

            elif args.optimizer == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=args.lr, initial_accumulator_value=1e-8).minimize(
                    model.loss)

            else:  # TODO(wgcn96): SGD
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr).minimize(loss=model.loss)

    with tf.Session(graph=model.g, config=config) as sess:
        # initial training
        sess.run(tf.global_variables_initializer())

        logging.info("--- Start training ---")
        print("--- Start training ---")

        # plot network then exit
        print('plot_network:', args.plot_network)
        if args.plot_network:
            print("writing network to TensorBoard/graphs/network")
            writer = tf.summary.FileWriter('TensorBoard/graphs/network')
            writer.add_graph(sess.graph)
            # return 0

        # dict for printing behavior type
        b_dict = {'vb': 'view and buy', 'cb': 'cart and buy', 'vc': 'view and cart'}

        samples = BatchUser.sampling(args, dataset, args.num_neg)  # TODO(wgcn96): 生成数据集不需要放在for循环里面
        print('all training number: %d' % len(samples[0]))
        # print('first label: %s' % samples[2][:20])
        # print('first user: %s' % samples[0][:20])
        # print('first item: %s' % samples[1][:20])

        if behave_type == 'ipv':
            bs = args.batch_size_ipv
        elif behave_type == 'cart':
            bs = args.batch_size_cart
        else:
            bs = args.batch_size_buy

        best_hr, best_ndcg, best_epoch, best_loss = 0, 0, 0, 0  # TODO(wgcn96): 记录所有循环中的最佳结果

        # train by epoch
        for epoch_count in range(args.epochs):

            # TODO(wgcn96): 单行为优化
            batches = BatchUser.shuffle(samples, bs, args)
            print('Already generate batch, batch size is %d' % bs)

            train_begin = time()
            train_loss = training_batch(model, sess, batches, args, optimizer)
            train_time = time() - train_begin
            # print('train time: %d' % train_time)

            if epoch_count % args.verbose == 0 and args.evaluate == 'yes':

                eval_begin = time()
                hits, ndcgs = EvalUser.eval(model, sess, dataset, EvalDict, args)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                eval_time = time() - eval_begin
                logging.info(
                    "Epoch %d [train %.1fs]: train_loss = %.4f  [test %.1fs] HR = %.4f, NDCG = %.4f"
                    % (epoch_count + 1, train_time, train_loss, eval_time, hr, ndcg))
                print("Epoch %d [train %.1fs]: train_loss = %.4f  [test %.1fs] HR = %.4f, NDCG = %.4f" % (
                    epoch_count + 1, train_time, train_loss, eval_time, hr, ndcg))
                # print('eval time : {}'.format(eval_time))

                if hr >= best_hr and ndcg >= best_ndcg:
                    best_hr = hr
                    best_ndcg = ndcg
                    best_epoch = epoch_count + 1
                    best_loss = train_loss

                # save results, save model
                (hr_list[base_epoch + epoch_count], ndcg_list[base_epoch + epoch_count],
                 loss_list[base_epoch + epoch_count]) = (hr, ndcg, train_loss)

        if epoch_count == (args.epochs - 1):
            print("All finish, best result hr: {}, ndcg: {}, epoch: {}, train loss: {}".format(best_hr, best_ndcg,
                                                                                               best_epoch,
                                                                                               best_loss))
            hr_list.append(best_hr)
            ndcg_list.append(best_ndcg)
            loss_list.append(best_epoch)


def save_results(args):
    if args.recover == 'yes':
        path_result = 'Recover'
    else:
        path_result = 'Result'

    import datetime
    prefix = datetime.datetime.now().strftime("%m%d")

    path_result += '/'
    path_result += prefix

    if not os.path.exists(path_result):
        os.makedirs(path_result)

    if args.recover == 'yes':
        with open(path_result + '/' + filename, 'w') as output:
            output.write('HR:%.4f,NDCG:%.4f' % (hr_recover, ndcg_recover))
    else:
        with open(path_result + '/' + filename, 'w') as output:
            for i in range(len(loss_list)):
                output.write('%.4f,%.4f,%.4f\n' % (loss_list[i], hr_list[i], ndcg_list[i]))


if __name__ == '__main__':
    args = parse_args()

    dataset = None
    filename = None
    hr_recover = None
    ndcg_recover = None

    if 'FC' in args.model:
        loss_list = list(range(3 * args.epochs))
        hr_list = list(range(3 * args.epochs))
        ndcg_list = list(range(3 * args.epochs))
    else:
        loss_list = list(range(args.epochs))
        hr_list = list(range(args.epochs))
        ndcg_list = list(range(args.epochs))

    print('------ %s ------' % (args.process_name))
    setproctitle.setproctitle(args.process_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_logging_and_result(args)

    print('--- data generation start ---')
    data_gen_begin = time()

    if args.dataset == 'ali':  # TODO(wgcn96): 切换路径
        path = '/home/wangchen/multi-behavior/ijaci15/sample_version_one/'
        args.b_num = 3
    else:  # TODO(wgcn96): 'beibei'
        path = '/home/wangchen/multi-behavior/beibei/sample_version_one/'
        args.b_num = 2
        # enable b_2_type ...
        args.b_2_type = 'vb'

    if ('BPR' in args.model) or (args.en_MC == 'yes') or (args.model == 'FISM'):
        dataset_all = Dataset(path=path, load_type='dict')
    else:
        dataset_ipv = Dataset(path=path, b_type='ipv')
        dataset_cart = Dataset(path=path, b_type='cart')
        dataset_buy = Dataset(path=path, b_type='buy')
        dataset_all = (dataset_ipv, dataset_cart, dataset_buy)

    print('data generation [%.1f s]' % (time() - data_gen_begin))

    if args.model == 'pure_GMF':
        if args.en_MC == 'yes':
            dataset = dataset_all
        else:
            dataset = dataset_buy
        model = pure_GMF(dataset.num_users, dataset.num_items, args)
        print('num_users:%d   num_items:%d' % (dataset.num_users, dataset.num_items))
        model.build_graph()

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                pass
            else:
                print('start single process')
                training(model, args, behave_type='buy')
                # training(model, args, behave_type='buy')

    elif args.model == 'pure_MLP':
        if args.en_MC == 'yes':
            dataset = dataset_all
        else:
            dataset = dataset_buy

        model = pure_MLP(dataset.num_users, dataset.num_items, args)
        print('num_users:%d   num_items:%d' % (dataset.num_users, dataset.num_items))
        model.build_graph()

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                pass
            else:
                print('start single process')
                training(model, args, behave_type='buy')

    elif args.model == 'pure_NCF':
        if args.en_MC == 'yes':
            dataset = dataset_all
        else:
            dataset = dataset_buy

        model = pure_NCF(dataset.num_users, dataset.num_items, args)
        print('num_users:%d   num_items:%d' % (dataset.num_users, dataset.num_items))
        model.build_graph()

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            if args.multiprocess == 'yes':
                pass
            else:
                print('start single process')
                training(model, args, behave_type='buy')

    elif args.model == 'FISM':
        model = FISM(dataset_all.num_items, dataset_all.num_users, dataset_all.max_rate, args)
        print('num_users:%d   num_items:%d  max_rate:%d' % (
            dataset_all.num_users, dataset_all.num_items, dataset_all.max_rate))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            print('start single process')
            training(model, args)

    elif 'BPR' in args.model:
        model = BPR(dataset_all.num_users, dataset_all.num_items, args)
        print('num_users:%d   num_items:%d' % (dataset_all.num_users, dataset_all.num_items))
        model.build_graph()
        dataset = dataset_all

        # recover result or not
        if args.recover == 'yes':
            eval_from_saved_model(model, args)

        else:
            print('start single process')
            training(model, args)

    elif 'MF' == args.model:
        dataset = dataset_buy
        model = MF(dataset.num_users, dataset.num_items, args)
        print('num_users:{}, num_items:{}'.format(dataset.num_users, dataset.num_items))
        model.build_graph()
        training(model, args, behave_type='buy')

    save_results(args)
