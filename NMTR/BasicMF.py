# -*- coding: utf-8 -*-

"""
My implement for basic MF and BPR-based MF

"""

__author__ = 'Wang Chen'
__time__ = '2019/12/4'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(1234)


class MF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.loss_func = args.loss_func
        self.dropout = args.dropout  # TODO(wgcn96): drop out only for BPR variables

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size],
                                    mean=0.0, stddev=0.01), name='embedding_P',
                dtype=tf.float32)  # (users, embedding_size)

            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size],
                                    mean=0.0, stddev=0.01), name='embedding_Q',
                dtype=tf.float32)  # (items, embedding_size)

            self.h = tf.Variable(
                tf.random_uniform([self.embedding_size, 1],
                                  minval=-tf.sqrt(3 / self.embedding_size), maxval=tf.sqrt(3 / self.embedding_size),
                                  name='h', dtype=tf.float32))

        with tf.name_scope("bias"):
            self.Bias = tf.Variable(tf.zeros([self.num_items, 1], name='Bias', dtype=tf.float32))

    def _create_inference(self, item_input, name):
        with tf.name_scope("inference"):
            self.b = tf.reduce_sum(tf.nn.embedding_lookup(self.Bias, item_input), 1)  # (batch, 1)
            # print(self.b.shape.as_list())

            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input),
                                             1)  # (batch, embedding_size)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (batch, embedding_size)
            return tf.sigmoid(tf.reduce_sum(self.embedding_p * self.embedding_q, axis=1, keepdims=True),
                              name=name)   # (b, embedding_size) * (embedding_size, 1)
            # return tf.sigmoid(tf.matmul(self.embedding_p * self.embedding_q, self.h), name=name)

    def _create_loss(self):
        with tf.name_scope("loss"):

            if self.loss_func == 'square_loss':  # TODO(平方损失 done)
                self.output = self._create_inference(self.item_input, 'output')
                # print(self.output.shape.as_list())
                self.error = tf.reduce_sum(tf.square(self.labels - self.output))
                self.loss = self.error + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_p)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_q))

            elif self.loss_func == 'log_loss':
                self.output = self._create_inference(self.item_input, 'output')
                # print(self.output.shape.as_list())
                self.error = tf.losses.log_loss(self.labels, self.output)
                self.loss = self.error + self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_p)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_q))

            else:  # BPR
                self.output = self._create_inference(self.item_input, 'output')
                self.output_neg = self._create_inference(self.item_input_neg, 'output_neg')
                # print(self.output.shape.as_list())
                self.result = self.output - self.output_neg
                # self.error = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))    # TODO(wgcn96): how to implement BPR loss ?
                self.error = tf.reduce_sum(-tf.log(tf.sigmoid(self.result)))
                self.loss = self.error + self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_p)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_q))

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()


if __name__ == '__main__':
    from ARGS import *

    args = ARGS(embed_size=64,
                lr=0.01,
                regs='[0, 0]',
                loss_func='square_loss',
                dropout=None)
    model = MF(num_users=10000, num_items=10000, args=args)
    model.build_graph()
    print(model)
    print('all done!')
