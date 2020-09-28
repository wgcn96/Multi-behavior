# -*- coding: utf-8 -*-

"""
attention part

this function is copied from Google original paper
"Attention is all you need"
"""

__author__ = 'Wang Chen'
__time__ = '2019/11/26'


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def multihead_attention(queries,
                        # queries_length,
                        keys,
                        # keys_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    """Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q]. # 每个维度具体对应的是什么？N个query,T种行为，C个特征的维度？
    # queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k]. # 为什么keys的各个维度和query一样，是经过特殊处理的吗？
    # keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size. # 如何理解？（应该是query的个数）
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads. # 在代码中如何体现映射到多个空间？
    scope: Optional scope for `variable_scope`. # 如何理解？
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C) # 此处value和key是相同的？

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h) # 为什么要进行split?
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        # query-key score matrix
        # each big score matrix is then split into h score matrix with same size
        # w.r.t. different part of the feature
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        # key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (N, T_k)
        # key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        #
        # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

        # Causality = Future blinding: No use, removed

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k) # 求出来的是类似于相关性矩阵的，T_q是query的维度，T_k是key的维度

        # Query Masking
        # query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
        # query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        # outputs *= query_masks  # broadcasting. (h*N, T_q, T_k) # 而且对应的h个head

        # Attention vector
        att_vec = outputs

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs, att_vec


def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                reuse=None):
    """Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    """Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
    `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs
