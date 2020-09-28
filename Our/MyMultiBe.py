# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/11/20'

import os
import math
import logging
from time import time, sleep
from time import strftime
from time import localtime
import datetime
import argparse
import pickle as pkl
import setproctitle

import numpy as np

from Our_algorithm import *
from Our_algorithm_revised import *
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
                        help='Choose a dataset: beibei, ali')
    parser.add_argument('--model', nargs='?', default='Our',
                        help='Choose model: Our, Our_GMF, Our_MLP, Our_NCF')
    parser.add_argument('--loss_func', nargs='?', default='logloss',
                        help='Choose loss: logloss, BPR')
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
    parser.add_argument('--layer_num', type=int, nargs='?', default=1,
                        help='layer number')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-6]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of FISM')
    parser.add_argument('--train_loss', type=bool, default=True,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--process_name', nargs='?', default='MBR@wangchen',
                        help='Input process name.')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU.')
    parser.add_argument('--evaluate', nargs='?', default='yes',
                        help='Evaluate or not.')
    parser.add_argument('--frozen', nargs='?', default='',
                        help='Frozen user or item')
    parser.add_argument('--optimizer', nargs='?', default='Adam',
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
    parser.add_argument('--b_2_type', nargs='?', default='vb',
                        help='when b_num=2, three condition: vc->view/cart, cb->cart/buy, vb->view/buy')
    parser.add_argument('--add_datetime', nargs='?', default='no')
    parser.add_argument('--save_file', nargs='?', default='result.txt')
    parser.add_argument('--recover_path', nargs='?', default='no',
                        help='indicate the recover model path, only useful when recover is yes')
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # EvalDict = EvalUser.init_evaluate_model(model, dataset[0], args)  # TODO(wgcn96): why this?
    EvalDict = EvalUser.gen_feed_dict(dataset[0])

    hits, ndcgs = [], []
    loader = None

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        # model_load_path = 'Model/model_' + filename
        loader = tf.train.import_meta_graph(args.recover_path + '.meta', clear_devices=True)

    with tf.Session(graph=eval_graph, config=config) as sess:
        loader.restore(sess, args.recover_path)

        for idx in range(len(EvalDict)):
            gtItem = dataset[0].testRatings[idx][1]
            predictions = sess.run('loss/errors/inference/score_buy:0', feed_dict=EvalDict[idx])

            sort_indexes = np.argsort(-predictions.reshape(predictions.shape[0]))
            top_K = sort_indexes[:args.topK]

            hr_tmp = 0
            ndcg_tmp = 0

            # 计算打分
            if gtItem in top_K:
                hr_tmp = 1
                pos = np.where(top_K == gtItem)[0]
                ndcg_tmp = math.log(2) / math.log(pos + 2)

            hits.append(hr_tmp)
            ndcgs.append(ndcg_tmp)

            """
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
            """

        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

        print("Final HR = %.4f, NDCG = %.4f" % (hr, ndcg))

        # save info to the server
        hr_recover, ndcg_recover = hr, ndcg


# used for multitask-learning
def training_batch_3(model, sess, batches, args, optimizer):
    loss_train = 0.0
    loss_train_all = 0.0

    if 'BPR' in args.model:
        user_input, item_input, item_input_neg = batches
        num_batch = len(batches[1])
        for i in range(len(user_input)):
            feed_dict = {model.user_input: np.reshape(user_input[i], [-1, 1]),
                         model.item_input: np.reshape(item_input[i], [-1, 1]),
                         model.item_input_neg: np.reshape(item_input_neg[i], [-1, 1])}
            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train

    else:
        user_input, item_input, labels = batches
        num_batch = len(batches[1])
        for i in range(len(labels)):
            if args.b_num == 2:
                if args.b_2_type == 'vc':
                    feed_dict = {model.user_input: user_input[i][:, None],
                                 model.item_input: item_input[i][:, None],
                                 model.labels_ipv: labels[i][:, 2][:, None],
                                 model.labels_cart: labels[i][:, 1][:, None]}

                elif args.b_2_type == 'vb':
                    feed_dict = {model.user_input: user_input[i][:, None],
                                 model.item_input: item_input[i][:, None],
                                 model.labels_ipv: labels[i][:, 2][:, None],
                                 model.labels_buy: labels[i][:, 0][:, None]}

                else:
                    feed_dict = {model.user_input: user_input[i][:, None],
                                 model.item_input: item_input[i][:, None],
                                 model.labels_cart: labels[i][:, 1][:, None],
                                 model.labels_buy: labels[i][:, 0][:, None]}

            else:
                feed_dict = {model.user_input: user_input[i][:, None],
                             model.item_input: item_input[i][:, None],
                             model.labels_ipv: labels[i][:, 2][:, None],
                             model.labels_cart: labels[i][:, 1][:, None],
                             model.labels_buy: labels[i][:, 0][:, None]}

            _, loss_train = sess.run([optimizer, model.loss], feed_dict)
            loss_train_all += loss_train

    return loss_train_all / num_batch


def training(model, args, behave_type=None, base_epoch=0, save=True):

    # import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1

    global pool, dataset, eval_queue, job_num, loss_list, hr_list, ndcg_list

    # initialize for Evaluate
    EvalDict = EvalUser.init_evaluate_model(model, dataset[0], args)

    with model.g.as_default():

        saver = tf.train.Saver()

        with tf.name_scope('optimizer'):
            if args.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(model.loss)

            elif args.optimizer == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=args.lr, initial_accumulator_value=1e-8).minimize(model.loss)

            else:   # TODO(wgcn96): SGD
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
            return 0

        # dict for printing behavior type
        b_dict = {'vb': 'view and buy', 'cb': 'cart and buy', 'vc': 'view and cart'}

        samples = BatchUser.sampling_3(args, dataset, args.num_neg)  # TODO(wgcn96): 生成数据集不需要放在for循环里面
        print('all training number: %d' % len(samples[0]))

        if args.b_num == 3:
            bs = args.batch_size
        elif args.b_2_type == 'cb':
            bs = args.batch_size_cart
        else:
            bs = args.batch_size_ipv

        best_hr, best_ndcg, best_epoch, best_loss = 0, 0, 0, 0  # TODO(wgcn96): 记录所有循环中的最佳结果

        # train by epoch
        for epoch_count in range(args.epochs):
            batches = BatchUser.shuffle_3(samples, bs, args)

            # print('Already generate batch, behavior is %d(%s), \n\
            #        batch size is %d, all training entries: %d' % (
            #     args.b_num, b_dict[args.b_2_type], bs, len(samples[0])))

            train_begin = time()
            train_loss = training_batch_3(model, sess, batches, args, optimizer)
            train_time = time() - train_begin

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

                    model_save_path = 'Model/' + args.process_name + '_best'     # save best model
                    saver.save(sess, model_save_path)

                # save best result
                (hr_list[base_epoch + epoch_count], ndcg_list[base_epoch + epoch_count],
                 loss_list[base_epoch + epoch_count]) = (hr, ndcg, train_loss)

        if epoch_count == (args.epochs - 1):
            print(
                "All finish, best result hr: {}, ndcg: {}, epoch: {}, train loss: {}".format(best_hr, best_ndcg,
                                                                                             best_epoch,
                                                                                             best_loss))
            hr_list.append(best_hr)
            ndcg_list.append(best_ndcg)
            loss_list.append(best_epoch)
            prefix = ''

            if args.add_datetime == 'yes':
                prefix += datetime.datetime.now().strftime("%m%d")
                prefix += '/'
                if not os.path.exists(prefix):
                    os.makedirs(prefix)

            with open(prefix + args.save_file, 'a+') as fo:
                fo.write("---------------\n")
                # fo.write("loss_func: %s\n" % args.loss_func)
                fo.write("batch_size: %s\n" % args.batch_size)
                fo.write("layer_num: %s\n" % args.layer_num)
                fo.write("lr: %s\n" % args.lr)
                fo.write("epoch: %s\n" % args.epochs)
                fo.write("optimizer: %s\n" % args.optimizer)
                fo.write("reg: %s\n" % args.regs)
                fo.write(
                    "All finish, best result hr: {}, ndcg: {}, epoch: {}, train loss: {}\n".format(best_hr, best_ndcg,
                                                                                                   best_epoch,
                                                                                                   best_loss))


def save_results(args, cascade=False):
    if args.recover == 'yes':
        path_result = 'Recover'
    else:
        path_result = 'Result'

    if args.add_datetime == 'yes':
        prefix = datetime.datetime.now().strftime("%m%d")
        path_result += '/'
        path_result += prefix

    if not os.path.exists(path_result):
        os.makedirs(path_result)

    if args.recover == 'yes':
        with open(path_result + '/' + filename, 'a+') as output:
            output.write('HR:%.4f,NDCG:%.4f' % (hr_recover, ndcg_recover))
    else:
        if cascade:
            pass
        else:
            with open(path_result + '/' + filename, 'a+') as output:
                for i in range(len(loss_list)):
                    output.write('%.4f,%.4f,%.4f\n' % (loss_list[i], hr_list[i], ndcg_list[i]))


"""
if __name__ == '__main__':
    import gc
    args = parse_args()

    print('--- data generation start ---')
    data_gen_begin = time()

    if args.dataset == 'ali':  # TODO(wgcn96): 切换路径
        path = '/home/wangchen/multi-behavior/ijaci15/sample_version_one/'
        args.b_num = 3
    else:   # TODO(wgcn96): 'beibei'
        path = '/home/wangchen/multi-behavior/beibei/sample_version_one/'
        args.b_num = 2
        # enable b_2_type ...
        args.b_2_type = 'vb'
        
    dataset_ipv = Dataset(path=path, b_type='ipv')
    dataset_cart = Dataset(path=path, b_type='cart')
    dataset_buy = Dataset(path=path, b_type='buy')
    dataset_all = (dataset_ipv, dataset_cart, dataset_buy)

    print('data generation [%.1f s]' % (time() - data_gen_begin))

    # 在此处增加arg的设置，并把结果写到文件当中
    # lossfunc = ['logloss', 'square_loss']   # log loss 由于square loss
    # batchsize = [512, 1024, 2048]   # 512 最优 不过没有特别大变化
    layernum = [1, 2, 3, 4]
    # learningrate = [0.001, 0.002, 0.01, 0.02, 0.05,]
    learningrate = [0.0005, 0.001, 0.002]
    # optimizers = ['Adam', 'Adagrad']
    optimizers = ['Adam']
    regs = ['[1e-5, 1e-6]', '[5e-6, 5e-6]', '[1e-6, 1e-6]']

    process_count = 0
    for optimizer in optimizers:
        for reg in regs:
            # for batch_size in batchsize:
                for lr in learningrate:
                    for layer_num in layernum:
                        process_count += 1
                        print("optimizer: %s" % optimizer)
                        # print("batch_size: %s" % batch_size)
                        print("learning_rate: %s" % lr)
                        print("layer_num: %s" % layer_num)
                        print("regs %s" % reg)
                        print("------------")

                        args.optimizer = optimizer
                        args.regs = reg
                        # args.batch_size = batch_size
                        args.lr = lr
                        args.layer_num = layer_num
                        args.process_name += str(process_count)

                        # TODO(wgcn96): global variable
                        dataset = None
                        filename = None
                        hr_recover = None
                        ndcg_recover = None

                        loss_list = list(range(args.epochs))
                        hr_list = list(range(args.epochs))
                        ndcg_list = list(range(args.epochs))

                        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

                        print('------ %s ------' % (args.process_name))
                        setproctitle.setproctitle(args.process_name)

                        init_logging_and_result(args)

                        if args.model == 'Our':
                            model = Our_Algorithm(dataset_all[0].num_users, dataset_all[0].num_items, args)
                        elif args.model == 'Our_GMF':
                            model = Our_Algorithm_GMF(dataset_all[0].num_users, dataset_all[0].num_items, args)
                        elif args.model == 'Our_MLP':
                            model = Our_Algorithm_MLP(dataset_all[0].num_users, dataset_all[0].num_items, args)
                        elif args.model == 'Our_NCF':
                            model = Our_Algorithm_NCF(dataset_all[0].num_users, dataset_all[0].num_items, args)

                        print('num_users:%d   num_items:%d' % (dataset_ipv.num_users, dataset_ipv.num_items))
                        model.build_graph()
                        dataset = dataset_all
                        print('start single process')

                        train_begin = time()
                        training(model, args)
                        print("current train finish {:.2f}".format(time()-train_begin))

                        save_results(args)
                        del model
                        gc.collect()
                        """


if __name__ == '__main__':

    args = parse_args()

    print('--- data generation start ---')
    data_gen_begin = time()

    if args.dataset == 'ali':  # TODO(wgcn96): 切换路径
        path = '/home/wangchen/multi-behavior/ijaci15/sample_version_one/'
        args.b_num = 3
    else:   # TODO(wgcn96): 'beibei'
        path = '/home/wangchen/multi-behavior/beibei/sample_version_one/'
        args.b_num = 2
        # enable b_2_type ...
        args.b_2_type = 'vb'

    dataset_ipv = Dataset(path=path, b_type='ipv')
    dataset_cart = Dataset(path=path, b_type='cart')
    dataset_buy = Dataset(path=path, b_type='buy')
    dataset_all = (dataset_ipv, dataset_cart, dataset_buy)

    print('data generation [%.1f s]' % (time() - data_gen_begin))

    # TODO(wgcn96): global variable
    dataset = None
    filename = None
    hr_recover = None
    ndcg_recover = None

    loss_list = list(range(args.epochs))
    hr_list = list(range(args.epochs))
    ndcg_list = list(range(args.epochs))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print('------ %s ------' % args.process_name)
    setproctitle.setproctitle(args.process_name)

    init_logging_and_result(args)

    if args.model == 'Our':
        model = Our_Algorithm(dataset_all[0].num_users, dataset_all[0].num_items, args)
    elif args.model == 'Our_GMF':
        model = Our_Algorithm_GMF(dataset_all[0].num_users, dataset_all[0].num_items, args)
    elif args.model == 'Our_MLP':
        model = Our_Algorithm_MLP(dataset_all[0].num_users, dataset_all[0].num_items, args)
    elif args.model == 'Our_NCF':
        model = Our_Algorithm_NCF(dataset_all[0].num_users, dataset_all[0].num_items, args)

    print('num_users:%d   num_items:%d' % (dataset_ipv.num_users, dataset_ipv.num_items))
    model.build_graph()
    dataset = dataset_all
    print('start single process')

    if args.recover == 'yes':
        eval_begin = time()
        eval_from_saved_model(model, args)
        print("current train finish {:.2f}".format(time() - eval_begin))
    else:
        train_begin = time()
        training(model, args)
        print("current train finish {:.2f}".format(time()-train_begin))

    save_results(args)
