# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RelationPrediction-master 
@File    ：disen_gcn.py
@IDE     ：PyCharm 
@Author  ：Firo
@Date    ：2021/11/6 17:22 
'''
import tensorflow as tf
import numpy as np
from common.shared_functions import dot_or_lookup, glorot_variance, make_tf_variable, make_tf_bias
# import sys
# sys.path.append("C:/Users\zhy\Desktop\第三学期学习\第一篇论文参考论文\RelationPrediction-master\code_R-GCN\encoders\layers.py")
from model import Model
# from layers import *
from encoders.layers import Disen_Conv

class Disen_GCN(Model):
    onehot_input = True
    use_nonlinearity = True
    vertex_embedding_function = {'train': None, 'test': None}

    def __init__(self, settings, next_component=None):  # 输入维数， 通道数目， 每个通道的输出维数， 迭代次数， 平衡因子
        Model.__init__(self, next_component, settings)
        self.dropout_rate = settings['dropout']
        self.channels = settings['channels']
        self.c_dim = settings['c_dim']
        self.iterations = settings['iterations']
        self.beta = settings['beta']
        self.layer_num = settings['layer_num']
        self.in_dim = settings['in_dim']
        self.out_dim = settings['out_dim']
        self.adj = settings['adj_matrix']

        # self.weight_list1 = []
        # self.weight_list2 = []
        # self.bias_list1 = []
        # self.bias_list2 = []
        # self.disconv1 = Disen_Conv(self.in_dim[0], self.channels[0], self.C_dim[0], self.iterations, self.beta)
        # self.disconv2 = Disen_Conv(self.in_dim[1], self.channels[1], self.C_dim[1], self.iterations, self.beta)

        # self.init_parameters()
        # self.triplets = triples

    # def init_parameters(self):
    #     tf.Variable(self.W_o)
    #     tf.Variable(self.bias)
    def local_initialize_train(self):
        self.W_o = tf.Variable(np.random.randn(self.channels[-1] * self.c_dim[-1], self.out_dim).astype(np.float32))# 16*8
        self.bias = tf.Variable(np.random.randn(1, self.out_dim).astype(np.float32))# 1*8
        self.w1 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))# 1713*8
        self.b1 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))# 1*8
        self.w2 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b2 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.w3 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b3 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.w4 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b4 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.w5 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b5 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.w6 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b6 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.w7 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b7 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.w8 = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))  # 1713*8
        self.b8 = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))  # 1*8
        self.weight_list = [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8]
        self.b_list = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8]
        # self.w2 = tf.Variable(np.random.randn(self.in_dim[1], self.c_dim[1]).astype(np.float32))# 16*4
        # self.b2 = tf.Variable(np.random.randn(1 ,self.c_dim[1]).astype(np.float32))# 1*4
        # for i in range(self.channels[0]):
        #     w = tf.Variable(np.random.randn(self.in_dim[0], self.c_dim[0]).astype(np.float32))
        #     self.weight_list1.append(w)
        # for i in range(self.channels[0]):
        #     b = tf.Variable(np.random.randn(1, self.c_dim[0]).astype(np.float32))
        #     self.bias_list1.append(b)
        #
        # for i in range(self.channels[1]):
        #     w = tf.Variable(np.random.randn(self.in_dim[1], self.c_dim[1]).astype(np.float32))
        #     self.weight_list2.append(w)
        # for i in range(self.channels[1]):
        #     b = tf.Variable(np.random.randn(1, self.c_dim[1]).astype(np.float32))
        #     self.bias_list2.append(b)

    def local_get_weights(self):
        return[self.W_o, self.bias, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5,
               self.w6, self.b6,self.w7, self.b7, self.w8, self.b8]

    def forward(self, features):
        h = features
        h = self.disen_conv(self.in_dim[0], self.channels[0], self.c_dim[0], self.iterations, self.beta, self.adj, h, self.weight_list, self.b_list)
        # h = tf.nn.dropout(h.forward(), self.dropout_rate)
        h = tf.nn.dropout(h, self.dropout_rate)
        # h = self.disen_conv(self.in_dim[1], self.channels[1], self.c_dim[1], self.iterations, self.beta, self.adj, h, self.w2, self.b2)
        # # h = tf.nn.dropout(h.forward(), self.dropout_rate)
        # h = tf.nn.dropout(h, self.dropout_rate)
        output = tf.matmul(h, self.W_o) + self.bias
        return output
    def disen_conv(self, in_dim, channels, c_dim, iterations, beta, adj, features, w, b):
        c_features = []
        # weight_list = []
        # bias_list = []
        # self.w = tf.Variable(np.random.randn(in_dim, c_dim).astype(np.float32))
        # # weight_list.append(w)
        #
        # self.b = tf.Variable(np.random.randn(1, c_dim).astype(np.float32))
        # bias_list.append(b)
        for i in range(channels):
            z = tf.matmul(features, w[i]) + b[i]  # 矩阵相乘加偏置
            z = tf.nn.l2_normalize(z, axis=1)  # 归一化
            c_features.append(z)
        out_features = c_features  # 执行论文的第一个两重循环，得到初始化的通道向量
        for l in range(iterations):
            c_attentions = []
            for i in range(channels):
                channel_f = out_features[i]  # channel_f =
                c_attentions.append(self.parse_attention(adj, channel_f, beta))  # adj=1713*1713
            all_attentions = tf.concat([c_attentions[i] for i in range(len(c_attentions))], axis=2)
            all_attentions = tf.nn.softmax(all_attentions, axis=2)
            neg_all_attention = tf.zeros_like(all_attentions)
            adj_all = tf.tile(tf.expand_dims(adj, axis=2), [1, 1, channels])
            all_attentions = tf.where(adj_all > 0, all_attentions, neg_all_attention)
            for k in range(channels):
                feat = out_features[k]
                atte = tf.squeeze(all_attentions[:, :, k])
                out_features[k] = (tf.nn.l2_normalize(feat + tf.matmul(atte, feat), axis=1))
        output = tf.concat([out_features[i] for i in range(len(out_features))], axis=1)
        return output

    def parse_attention(self, adj, features, beta):
        attention_matrix = tf.matmul(features, tf.transpose(features))
        neg_attention = tf.zeros_like(attention_matrix)
        attention_matrix = tf.where(adj > 0, attention_matrix, neg_attention)
        attention_matrix = attention_matrix * 1.0 / (beta)
        attention_matrix = tf.expand_dims(attention_matrix, axis=2)

        return attention_matrix

    def get_all_codes(self, mode = 'train'):
        features = self.next_component.get_all_codes(mode=mode)
        # message = self.forward(tf.Variable(np.random.randn(1713, 1713).astype(np.float32)))
        message = self.forward(features[0])
        # message = self.next_component.get_all_codes(mode=mode)
        return message, None, message
    # def get_features(self):
    #     # 对特征使用one_hot编码
    #     triplets = tf.transpose(self.triplets)
    #     sender_features = triplets[0]
    #     receiver_features = triplets[2]



