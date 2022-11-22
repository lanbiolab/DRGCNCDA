import numpy as np
import tensorflow as tf
from model import Model

from common.shared_functions import glorot_variance, make_tf_variable, make_tf_bias

class AffineTransform(Model):
    embedding_width = None

    W = None
    b = None
    use_nonlinearity = False
    onehot_input = False
    use_bias = True
    shape = None

    def __init__(self, shape, settings, next_component=None, use_nonlinearity=False, onehot_input=False, use_bias=True):
        Model.__init__(self, next_component, settings)
        self.shape = shape#
        self.use_nonlinearity = use_nonlinearity
        self.use_bias = use_bias
        self.onehot_input = onehot_input

    def local_initialize_train(self):
        variance = glorot_variance(self.shape)# shape[1713,500]
        # print("##############")
        # print(self.shape, self.use_bias, self.use_nonlinearity, self.onehot_input)
        # print("##############")

        self.W = make_tf_variable(0, variance, self.shape)# 返回从高斯分布中抽取的样本
        self.b = make_tf_bias(self.shape[1])

    def local_get_weights(self):
        # return self.W
        return [self.W, self.b]

    def get_all_subject_codes(self, mode='train'):
        #print("3")
        if self.onehot_input:#不执行
            hidden = self.W
        else:#执行
            code = self.next_component.get_all_subject_codes(mode=mode)
            hidden = tf.matmul(code, self.W)# tf.matmul()

        if self.use_bias:#执行
            hidden += self.b

        if self.use_nonlinearity:#不执行
            hidden = tf.nn.relu(hidden)

        return hidden

    def get_all_object_codes(self, mode='train'):
        #print("2")
        if self.onehot_input:
            hidden = self.W
        else:
            code = self.next_component.get_all_object_codes(mode=mode)
            hidden = tf.matmul(code, self.W)

        if self.use_bias:
            hidden += self.b

        if self.use_nonlinearity:
            hidden = tf.nn.relu(hidden)

        return hidden

    def get_all_codes(self, mode='train'):
        #print("1")
        print(self.onehot_input, self.use_bias, self.use_nonlinearity)
        if self.onehot_input:# 执行
            hidden_subject = self.W
            hidden_object = self.W
            hidden_relation = None
        else:
            codes = self.next_component.get_all_codes(mode=mode)
            # print(type(codes))
            hidden_subject = tf.matmul(codes[0], self.W)# 矩阵相乘
            hidden_object = tf.matmul(codes[2], self.W)
            hidden_relation = codes[1]

        if self.use_bias:# 不执行
            hidden_subject += self.b
            hidden_object += self.b

        if self.use_nonlinearity:# 不执行
            hidden_subject = tf.nn.relu(hidden_subject)
            hidden_object = tf.nn.relu(hidden_object)


        return hidden_subject, hidden_relation, hidden_object # 返回所有