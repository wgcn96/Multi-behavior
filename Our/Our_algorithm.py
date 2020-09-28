# -*- coding: utf-8 -*-

"""
The first version of our algorithm, implementing it without the attention layer

Second Version in Nov. 27th, changed in attention layer;
buy still used single attetion instead of multi-head attention.
"""

__author__ = 'Wang Chen'
__time__ = '2019/11/25'

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# util 本意是想 copy atrank 的函数过来，但发现tensorflow已经集成了attention层，所以没有用它的
# from util import *

tf.set_random_seed(1234)


class Our_Algorithm():
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
        self.opt = args.optimizer
        self.layer_num = args.layer_num
        self.b_num = args.b_num
        self.b_2_type = args.b_2_type

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
            if self.b_num == 3:     # b_num == 2 or 3
                self.embedding_P_3 = tf.Variable(
                    tf.truncated_normal(shape=[self.num_users, self.embedding_size],
                                        mean=0.0, stddev=0.01), name='embedding_P_3', dtype=tf.float32)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size],
                                    mean=0.0, stddev=0.01), name='embedding_Q', dtype=tf.float32)

        with tf.name_scope('attention_layer'):
            pass

        # with tf.name_scope('shared_bias'):
        #     self.bias = tf.Variable(tf.zeros([self.num_items, 1]), name='bias', dtype=tf.float32)

        with tf.name_scope('NCF'):

            # the h-vector in original paper
            v_size = self.embedding_size + int(self.embedding_size / (2 ** (self.layer_num - 1)))
            self.v_1 = tf.Variable(tf.random_uniform([v_size, 1], minval=-tf.sqrt(3 / v_size),
                                                     maxval=tf.sqrt(3 / v_size)), name='v_1')
            self.v_2 = tf.Variable(tf.random_uniform([v_size, 1], minval=-tf.sqrt(3 / v_size),
                                                     maxval=tf.sqrt(3 / v_size)), name='v_2')
            if self.b_num == 3:
                self.v_3 = tf.Variable(tf.random_uniform([v_size, 1], minval=-tf.sqrt(3 / v_size),
                                                         maxval=tf.sqrt(3 / v_size)), name='v_3')

            if self.layer_num == 0:
                pass  # no variable

            elif self.layer_num == 1:
                # view specific
                self.W1 = tf.Variable(
                    tf.random_uniform(shape=[2 * self.embedding_size, self.embedding_size],
                                      minval=-tf.sqrt(1 / self.embedding_size),
                                      maxval=tf.sqrt(1 / self.embedding_size)), name='W1')
                self.b1 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b1')

                # add cart specific
                self.W2 = tf.Variable(
                    tf.random_uniform(shape=[2 * self.embedding_size, self.embedding_size],
                                      minval=-tf.sqrt(3 / (2 * self.embedding_size)),
                                      maxval=tf.sqrt(3 / (2 * self.embedding_size))), name='W2')
                self.b2 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b2')

                # buy specific
                if self.b_num == 3:
                    self.W3 = tf.Variable(
                        tf.random_uniform(shape=[2 * self.embedding_size, self.embedding_size],
                                          minval=-tf.sqrt(3 / (2 * self.embedding_size)),
                                          maxval=tf.sqrt(3 / (2 * self.embedding_size))), name='W3')
                    self.b3 = tf.Variable(tf.zeros([1, self.embedding_size]), dtype=tf.float32, name='b3')

            else:
                self.W1, self.b1 = [], []
                self.W2, self.b2 = [], []
                if self.b_num == 3:
                    self.W3, self.b3 = [], []

                for i in range(self.layer_num):
                    input_size = int(2 * self.embedding_size / (2 ** i))
                    output_size = int(2 * self.embedding_size / (2 ** (i + 1)))
                    self.W1.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                                 minval=-tf.sqrt(3 / input_size),
                                                                 maxval=tf.sqrt(3 / input_size)), name='W1_%d' % i))
                    self.b1.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b1_%d' % i))
                    self.W2.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                                 minval=-tf.sqrt(3 / input_size),
                                                                 maxval=tf.sqrt(3 / input_size)), name='W2_%d' % i))
                    self.b2.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b2_%d' % i))
                    if self.b_num == 3:
                        self.W3.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                                     minval=-tf.sqrt(3 / input_size),
                                                                     maxval=tf.sqrt(3 / input_size)), name='W3_%d' % i))
                        self.b3.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b3_%d' % i))

        with tf.name_scope('multi_task_learning'):
            self.H = tf.Variable(tf.ones(shape=[self.num_users, self.b_num]),
                                 name='co_relation_h', dtype=tf.float32)
            # tmp = np.zeros([self.num_users, self.b_num], dtype=np.int32)
            # tmp[:, self.b_num-1] = 1
            # self.H = tf.constant(tmp, name='co_relation_h', dtype=tf.float32)

    def _create_inference(self):

        with tf.name_scope('inference'):
            # [B, 1] item-popularity
            # b = tf.reduce_sum(tf.nn.embedding_lookup(self.bias, self.item_input), 1)

            # user-specific multi-behavior co-relation variable
            h = tf.reduce_sum(tf.nn.embedding_lookup(self.H, self.user_input), 1)
            # print(h.shape.as_list())

            # linear embeddings [B, E]
            embedding_p_1 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_1, self.user_input), 1)
            embedding_p_2 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_2, self.user_input), 1)
            if self.b_num == 3:
                embedding_p_3 = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P_3, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)

            # attention layer [B, E]
            if self.b_num == 3:
                query = tf.stack([embedding_p_1, embedding_p_2, embedding_p_3],
                             axis=1)  # TODO(wgcn96): expected shape [B, 3, E]

                # batch_size = self.user_input.shape.to_list()[0]
                # query, weight_vec = multihead_attention(queries=query,
                #                                                 queries_length=tf.constant(3, shape=[batch_size,]),
                #                                                 keys=query,
                #                                                 keys_length=tf.constant(3, shape=[batch_size,]),
                #                                                 scope="self_attention"
                #                                                 )

                value = query
                query = tf.keras.layers.Attention()([query, value])

                embedding_p_1, embedding_p_2, embedding_p_3 = tf.split(query, [1, 1, 1],
                                                                       axis=1)  # expected [B, E], but [B, 1, E]
                embedding_p_1 = tf.squeeze(embedding_p_1)  # [B, E]
                embedding_p_2 = tf.squeeze(embedding_p_2)  # [B, E]
                embedding_p_3 = tf.squeeze(embedding_p_3)  # [B, E]

                # [B, 2E]
                concat_vec_1 = tf.concat([embedding_p_1, embedding_q], 1, name='concat_vec_1')
                concat_vec_2 = tf.concat([embedding_p_2, embedding_q], 1, name='concat_vec_2')
                concat_vec_3 = tf.concat([embedding_p_3, embedding_q], 1, name='concat_vec_3')
            else:
                query = tf.stack([embedding_p_1, embedding_p_2],
                                 axis=1)  # TODO(wgcn96): expected shape [B, 2, E]
                # batch_size = self.user_input.shape.to_list()[0]
                # query, weight_vec = multihead_attention(queries=query,
                #                                                 queries_length=tf.constant(3, shape=[batch_size,]),
                #                                                 keys=query,
                #                                                 keys_length=tf.constant(3, shape=[batch_size,]),
                #                                                 scope="self_attention"
                #                                                 )

                value = query
                query = tf.keras.layers.Attention()([query, value])

                embedding_p_1, embedding_p_2 = tf.split(query, [1, 1], axis=1)  # expected [B, E], but [B, 1, E]
                embedding_p_1 = tf.squeeze(embedding_p_1)  # [B, E]
                embedding_p_2 = tf.squeeze(embedding_p_2)  # [B, E]

                # [B, 2E]
                concat_vec_1 = tf.concat([embedding_p_1, embedding_q], 1, name='concat_vec_1')
                concat_vec_2 = tf.concat([embedding_p_2, embedding_q], 1, name='concat_vec_2')

            # NCF unit
            if self.layer_num == 0:
                if self.b_num == 3:
                    # predict ipv
                    latent_factor_1 = tf.concat([concat_vec_1, embedding_p_1 * embedding_q], 1, name='latent_factor_1')
                    output_ipv = tf.matmul(latent_factor_1, self.v_1)

                    # predict cart
                    latent_factor_2 = tf.concat([concat_vec_2, embedding_p_2 * embedding_q], 1, name='latent_factor_2')
                    output_cart = tf.matmul(latent_factor_2, self.v_2)

                    # predict buy
                    latent_factor_3 = tf.concat([concat_vec_3, embedding_p_3 * embedding_q], 1, name='latent_factor_3')
                    output_buy = tf.matmul(latent_factor_3, self.v_3)

                    output_vec = tf.concat([output_ipv, output_cart, output_buy], axis=1, name='output_contact')
                    output_buy = tf.reduce_sum(tf.multiply(h, output_vec), axis=1, keepdims=True, name='final_buy')

                    if self.loss_func == 'square_loss':
                        return output_ipv, output_cart, output_buy
                    else:
                        return (tf.sigmoid(output_ipv, name='score_ipv'),
                            tf.sigmoid(output_cart, name='score_cart'),
                            tf.sigmoid(output_buy, name='score_buy'))
                else:
                    # predict ipv
                    latent_factor_1 = tf.concat([concat_vec_1, embedding_p_1 * embedding_q], 1, name='latent_factor_1')
                    output_ipv = tf.matmul(latent_factor_1, self.v_1)

                    # predict buy
                    latent_factor_2 = tf.concat([concat_vec_2, embedding_p_2 * embedding_q], 1, name='latent_factor_2')
                    output_buy = tf.matmul(latent_factor_2, self.v_2)

                    output_vec = tf.concat([output_ipv, output_buy], axis=1, name='output_contact')
                    output_buy = tf.reduce_sum(tf.multiply(h, output_vec), axis=1, keepdims=True, name='final_buy')

                    if self.loss_func == 'square_loss':
                        return output_ipv, output_buy
                    else:
                        return (tf.sigmoid(output_ipv, name='score_ipv'),
                                tf.sigmoid(output_buy, name='score_buy'))

            elif self.layer_num == 1:
                if self.b_num == 3:
                    # predict ipv
                    latent_factor_1 = tf.nn.relu(tf.matmul(concat_vec_1, self.W1) + self.b1)
                    latent_factor_1 = tf.concat([latent_factor_1, embedding_p_1 * embedding_q], 1,
                                                name='latent_factor_1')
                    output_ipv = tf.matmul(latent_factor_1, self.v_1)

                    # predict cart
                    latent_factor_2 = tf.nn.relu(tf.matmul(concat_vec_2, self.W2) + self.b2)
                    latent_factor_2 = tf.concat([latent_factor_2, embedding_p_2 * embedding_q], 1,
                                                name='latent_factor_2')
                    output_cart = tf.matmul(latent_factor_2, self.v_2)

                    # predict buy
                    latent_factor_3 = tf.nn.relu(tf.matmul(concat_vec_3, self.W3) + self.b3)
                    latent_factor_3 = tf.concat([latent_factor_3, embedding_p_3 * embedding_q], 1,
                                                name='latent_factor_3')
                    output_buy = tf.matmul(latent_factor_3, self.v_3)

                    output_vec = tf.concat([output_ipv, output_cart, output_buy], axis=1, name='output_contact')
                    output_buy = tf.reduce_sum(tf.multiply(h, output_vec), axis=1, keepdims=True, name='final_buy')

                    if self.loss_func == 'square_loss':
                        return output_ipv, output_cart, output_buy
                    else:
                        return (tf.sigmoid(output_ipv, name='score_ipv'),
                                tf.sigmoid(output_cart, name='score_cart'),
                                tf.sigmoid(output_buy, name='score_buy'))

                else:  # behavior typr == 2 TODO(wgcn96): leave for future
                    latent_factor_1 = tf.nn.relu(tf.matmul(concat_vec_1, self.W1) + self.b1)
                    latent_factor_1 = tf.concat([latent_factor_1, embedding_p_1 * embedding_q], 1,
                                                name='latent_factor_1')
                    output_ipv = tf.matmul(latent_factor_1, self.v_1)

                    # predict buy
                    latent_factor_2 = tf.nn.relu(tf.matmul(concat_vec_2, self.W2) + self.b2)
                    latent_factor_2 = tf.concat([latent_factor_2, embedding_p_2 * embedding_q], 1,
                                                name='latent_factor_2')
                    output_buy = tf.matmul(latent_factor_2, self.v_2)

                    output_vec = tf.concat([output_ipv, output_buy], axis=1, name='output_contact')
                    output_buy = tf.reduce_sum(tf.multiply(h, output_vec), axis=1, keepdims=True, name='final_buy')

                    if self.loss_func == 'square_loss':
                        return output_ipv, output_buy
                    else:
                        return (tf.sigmoid(output_ipv, name='score_ipv'),
                                tf.sigmoid(output_buy, name='score_buy'))

            else:
                if self.b_num == 3:
                    fc_1, fc_2, fc_3 = [], [], []
                    for i in range(self.layer_num):
                        if i == 0:
                            fc_1.append(tf.nn.relu(tf.matmul(concat_vec_1, self.W1[i]) + self.b1[i]))
                            fc_2.append(tf.nn.relu(tf.matmul(concat_vec_2, self.W2[i]) + self.b2[i]))
                            fc_3.append(tf.nn.relu(tf.matmul(concat_vec_3, self.W3[i]) + self.b3[i]))

                        else:
                            fc_1.append(tf.nn.relu(tf.matmul(fc_1[i - 1], self.W1[i]) + self.b1[i]))
                            fc_2.append(tf.nn.relu(tf.matmul(fc_2[i - 1], self.W2[i]) + self.b2[i]))
                            fc_3.append(tf.nn.relu(tf.matmul(fc_3[i - 1], self.W3[i]) + self.b3[i]))

                    # predict ipv
                    latent_factor_1 = tf.concat([fc_1[i], embedding_p_1 * embedding_q], 1,
                                                name='latent_factor_1')
                    output_ipv = tf.matmul(latent_factor_1, self.v_1)

                    # predict cart
                    latent_factor_2 = tf.concat([fc_2[i], embedding_p_2 * embedding_q], 1,
                                                name='latent_factor_2')
                    output_cart = tf.matmul(latent_factor_2, self.v_2)

                    # predict buy
                    latent_factor_3 = tf.concat([fc_3[i], embedding_p_3 * embedding_q], 1,
                                                name='latent_factor_3')
                    output_buy = tf.matmul(latent_factor_3, self.v_3)

                    output_vec = tf.concat([output_ipv, output_cart, output_buy], axis=1, name='output_contact')
                    output_buy = tf.reduce_sum(tf.multiply(h, output_vec), axis=1, keepdims=True, name='final_buy')

                    if self.loss_func == 'square_loss':
                        return output_ipv, output_cart, output_buy
                    else:
                        return (tf.sigmoid(output_ipv, name='score_ipv'),
                                tf.sigmoid(output_cart, name='score_cart'),
                                tf.sigmoid(output_buy, name='score_buy'))

                else:  # behavior type == 2 TODO(wgcn96): leave for future
                    fc_1, fc_2 = [], []
                    for i in range(self.layer_num):
                        if i == 0:
                            fc_1.append(tf.nn.relu(tf.matmul(concat_vec_1, self.W1[i]) + self.b1[i]))
                            fc_2.append(tf.nn.relu(tf.matmul(concat_vec_2, self.W2[i]) + self.b2[i]))

                        else:
                            fc_1.append(tf.nn.relu(tf.matmul(fc_1[i - 1], self.W1[i]) + self.b1[i]))
                            fc_2.append(tf.nn.relu(tf.matmul(fc_2[i - 1], self.W2[i]) + self.b2[i]))
                    # predict ipv
                    latent_factor_1 = tf.concat([fc_1[i], embedding_p_1 * embedding_q], 1,
                                                name='latent_factor_1')
                    output_ipv = tf.matmul(latent_factor_1, self.v_1)

                    # predict buy
                    latent_factor_2 = tf.concat([fc_2[i], embedding_p_2 * embedding_q], 1,
                                                name='latent_factor_2')
                    output_buy = tf.matmul(latent_factor_2, self.v_2)

                    output_vec = tf.concat([output_ipv, output_buy], axis=1, name='output_contact')
                    output_buy = tf.reduce_sum(tf.multiply(h, output_vec), axis=1, keepdims=True,
                                               name='final_buy')

                    if self.loss_func == 'square_loss':
                        return output_ipv, output_buy
                    else:
                        return (tf.sigmoid(output_ipv, name='score_ipv'),
                                tf.sigmoid(output_buy, name='score_buy'))

    def _create_loss(self):
        with tf.name_scope('loss'):

            with tf.name_scope('regs'):
                loss_W1, loss_W2, loss_W3 = 0, 0, 0

                if self.layer_num == 0:
                    pass
                elif self.layer_num == 1:
                    loss_W1 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1))
                    loss_W2 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2))
                    if self.b_num == 3:
                        loss_W3 = self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3))
                else:
                    for i in range(len(self.W1)):
                        loss_W1 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W1[i]))
                        loss_W2 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W2[i]))
                        if self.b_num == 3:
                            loss_W3 += self.lambda_bilinear * tf.reduce_sum(tf.square(self.W3[i]))

                loss_em = self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P_1)) + \
                          self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P_2)) + \
                          self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))

                if self.b_num == 3:
                    loss_em += self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P_3))

                self.loss_reg = loss_em + loss_W1 + loss_W2 + loss_W3

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
                    else:   # behavior type == 2 TODO(wgcn96): leave for future
                        self.score_ipv,  self.score_buy = self._create_inference()
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
                    else:   # behavior type == 2 TODO(wgcn96): leave for future
                        self.score_ipv, self.score_buy = self._create_inference()
                        self.loss_ipv = tf.reduce_sum(tf.square(self.labels_ipv - self.score_ipv))
                        self.loss_buy = tf.reduce_sum(tf.square(self.labels_buy - self.score_buy))
                        self.error = self.ipv_loss_coefficient * self.loss_ipv + \
                                     self.buy_loss_coefficient * self.loss_buy
                        self.loss = self.error + self.loss_reg
                else:   # TODO(wgcn96):unknown loss, leave for BPR?
                    pass

    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss()
