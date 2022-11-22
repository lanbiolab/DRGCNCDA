import tensorflow as tf
import numpy as np


def dot_or_lookup(features, weights, onehot_input=False):
    if onehot_input:
        return tf.nn.embedding_lookup(weights, features)
    else:
        return tf.matmul(features, weights)


def glorot_variance(shape):
    return 3 / np.sqrt(shape[0] + shape[1])# shape[0]=1713,实体数量;shape[1]=500,嵌入维度


def make_tf_variable(mean, variance, shape, init="normal"):
    if init == "normal":
        initializer = np.random.normal(mean, variance, size=shape).astype(np.float32)# 对于numpy.random.normal函数，有三个参数（loc, scale, size），分别l代表生成的高斯分布的随机数的均值、方差以及输出的size
    elif init == "uniform":
        initializer = np.random.uniform(mean, variance, size=shape).astype(np.float32)

    return tf.Variable(initializer)


def make_tf_bias(shape, init=0):
    if init == 0:
        return tf.Variable(np.zeros(shape).astype(np.float32))
    elif init == 1:
        return tf.Variable(np.ones(shape).astype(np.float32))
