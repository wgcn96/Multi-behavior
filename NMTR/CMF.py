# -*- coding: utf-8 -*-

"""
My implementation for CMF
"""

__author__ = 'Wang Chen'
__time__ = '2019/12/15'

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

tf.set_random_seed(1234)


class CMF:
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        loss_coefficient = eval(args.loss_coefficient)
        self.cart_loss_coefficient = loss_coefficient[0]
        self.buy_loss_coefficient = loss_coefficient[1]
        self.ipv_loss_coefficient = 1 - self.cart_loss_coefficient - self.buy_loss_coefficient
        self.b_num = args.b_num

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name='user_input')
            self.item_input = tf.placeholder(tf.int32, shape=[None, None], name='item_input')
            self.labels_ipv = tf.placeholder(tf.float32, shape=[None, 1], name='labels_ipv')
            self.labels_cart = tf.placeholder(tf.float32, shape=[None, 1], name='labels_cart')
            self.labels_buy = tf.placeholder(tf.float32, shape=[None, 1], name='labels_buy')

    def _create_variables(self):

        with tf.name_scope('linear_embedding'):
            self.embedding_P_1 = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size],
                                    mean=0.0, stddev=0.01), name='embedding_P_1', dtype=tf.float32)
            self.embedding_P_2 = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size],
                                    mean=0.0, stddev=0.01), name='embedding_P_2', dtype=tf.float32)
            if self.b_num == 3:  # b_num == 2 or 3
                self.embedding_P_3 = tf.Variable(
                    tf.truncated_normal(shape=[self.num_users, self.embedding_size],
                                        mean=0.0, stddev=0.01), name='embedding_P_3', dtype=tf.float32)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size],
                                    mean=0.0, stddev=0.01), name='embedding_Q', dtype=tf.float32)

    def _create_inference(self):

        with tf.name_scope('inference'):
            if self.b_num == 3:
                # linear embeddings [B, E]
                embedding_p_1 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_1, self.user_input), 1)
                embedding_p_2 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_2, self.user_input), 1)
                embedding_p_3 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_3, self.user_input), 1)

                embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)

                # predict ipv
                output_ipv = tf.reduce_sum(embedding_p_1 * embedding_q, axis=1, keepdims=True)

                # predict cart
                output_cart = tf.reduce_sum(embedding_p_2 * embedding_q, axis=1, keepdims=True)

                # predict buy
                output_buy = tf.reduce_sum(embedding_p_3 * embedding_q, axis=1, keepdims=True)

                if self.loss_func == 'square_loss':
                    return output_ipv, output_cart, output_buy
                else:
                    return (tf.sigmoid(output_ipv, name='score_ipv'),
                            tf.sigmoid(output_cart, name='score_cart'),
                            tf.sigmoid(output_buy, name='score_buy'))
            else:
                embedding_p_1 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_1, self.user_input), 1)
                embedding_p_2 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_2, self.user_input), 1)

                embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)

                # predict ipv
                output_ipv = tf.reduce_sum(embedding_p_1 * embedding_q, axis=1, keepdims=True)

                # predict buy
                output_buy = tf.reduce_sum(embedding_p_2 * embedding_q, axis=1, keepdims=True)

                if self.loss_func == 'square_loss':
                    return output_ipv, output_buy
                else:
                    return (tf.sigmoid(output_ipv, name='score_ipv'),
                            tf.sigmoid(output_buy, name='score_buy'))

    def _create_loss(self):
        with tf.name_scope('loss'):

            with tf.name_scope('regs'):
                loss_em = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P_1)) + \
                          self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P_2)) + \
                          self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))

                if self.b_num == 3:
                    loss_em += self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P_3))

                self.loss_reg = loss_em

            with tf.name_scope('errors'):
                if self.loss_func == 'logloss':
                    if self.b_num == 3:
                        self.score_ipv, self.score_cart, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_cart = tf.losses.log_loss(self.labels_cart, self.score_cart)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.error = self.ipv_loss_coefficient * self.loss_ipv + \
                                     self.cart_loss_coefficient * self.loss_cart + \
                                     self.buy_loss_coefficient * self.loss_buy
                        self.loss = self.error + self.loss_reg
                    else:  # behavior typr == 2 TODO(wgcn96): leave for future
                        self.score_ipv, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.losses.log_loss(self.labels_ipv, self.score_ipv)
                        self.loss_buy = tf.losses.log_loss(self.labels_buy, self.score_buy)
                        self.error = self.ipv_loss_coefficient * self.loss_ipv + \
                                     self.buy_loss_coefficient * self.loss_buy
                        self.loss = self.error + self.loss_reg

                elif self.loss_func == 'square_loss':
                    if self.b_num == 3:
                        self.score_ipv, self.score_cart, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.reduce_sum(tf.square(self.labels_ipv - self.score_ipv))
                        self.loss_cart = tf.reduce_sum(tf.square(self.labels_cart - self.score_cart))
                        self.loss_buy = tf.reduce_sum(tf.square(self.labels_buy - self.score_buy))
                        self.error = self.ipv_loss_coefficient * self.loss_ipv + \
                                     self.cart_loss_coefficient * self.loss_cart + \
                                     self.buy_loss_coefficient * self.loss_buy
                        self.loss = self.error + self.loss_reg
                    else:
                        self.score_ipv, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.reduce_sum(tf.square(self.labels_ipv - self.score_ipv))
                        self.loss_buy = tf.reduce_sum(tf.square(self.labels_buy - self.score_buy))
                        self.error = self.ipv_loss_coefficient * self.loss_ipv + \
                                     self.buy_loss_coefficient * self.loss_buy
                        self.loss = self.error + self.loss_reg
                else:  # TODO(wgcn96):unknown loss, leave for BPR?
                    pass

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
                dropout=None,
                loss_coefficient='[1/3,1/3,1/3]')
    model = CMF(num_users=10000, num_items=10000, args=args)
    model.build_graph()
    print(model)
    print('all done!')
