# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RelationPrediction-master 
@File    ：layers.py
@IDE     ：PyCharm 
@Author  ：Firo
@Date    ：2021/11/7 16:16 
'''
import tensorflow as tf
import numpy as np
# from model import Model


class Disen_Conv():
    def __init__(self, in_dim, channels, C_dim, iterations, beta, adj, features):  # 输入维数， 通道数目， 每个通道的输出维数， 迭代次数， 平衡因子
        # Model.__init__(self, next_component, settings)
        self.channels = channels# 通道个数
        self.in_dim = in_dim# 输入维度
        self.c_dim = C_dim
        self.iterations = iterations#路由的循环次数
        self.beta = beta
        self.weight_list = []
        self.bias_list = []
        self.adj = adj
        self.features = features
        for i in range(self.channels):
            w = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
            self.weight_list.append(w)
        for i in range(self.channels):
            b = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
            self.bias_list.append(b)
        # init_op = tf.local_variables_initializer()
        #
        # with tf.Session() as sess:
        #     sess.run(init_op)
        # self.relu = tf.nn.relu()
        # w1 = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
        # w2 = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
        # w3 = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
        # w4 = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
        # w5 = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
        # self.weight_list.append(w1)
        # self.weight_list.append(w2)
        # self.weight_list.append(w3)
        # self.weight_list.append(w4)
        # self.weight_list.append(w5)
        #
        # b1 = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
        # b2 = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
        # b3 = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
        # b4 = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
        # b5 = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
        # self.bias_list.append(b1)
        # self.bias_list.append(b2)
        # self.bias_list.append(b3)
        # self.bias_list.append(b4)
        # self.bias_list.append(b5)

        # self.weight_list = tf.constant(size=(self.in_dim, self.c_dim), dtype=tf.float32)
        # 在列表中保存权重参数

        # self.bias_list = nn.ParameterList(
        #     nn.Parameter(torch.empty(size=(1, self.c_dim), dtype=torch.float), requires_grad=True) for i in
        #     range(self.channels))# 在列表中保存偏置参数
        # self.init_parameters()
        # self.weight_list.ini
        # with tf.Session() as sess:
        #     sess.run(self.weight_list)
        #     sess.run(self.bias_list)


    # def init_parameters(self):
    #     for i, item in enumerate(self.parameters()):# 初始化所有参数
    #         torch.nn.init.normal_(item, mean=0, std=1)
    def local_initialize_train(self):
        for i in range(self.channels):
            w = tf.Variable(np.random.randn(self.in_dim, self.c_dim).astype(np.float32))
            self.weight_list.append(w)
        for i in range(self.channels):
            b = tf.Variable(np.random.randn(1, self.c_dim).astype(np.float32))
            self.bias_list.append(b)
    def local_get_weights(self):
        return [self.weight_list, self.bias_list]
    # def local_get_bias(self):
    #     return self.bias_list
    def forward(self):
        c_features = []

        for i in range(self.channels):
            z = tf.matmul(self.features, self.weight_list[i]) + self.bias_list[i]# 矩阵相乘加偏置
            z = tf.nn.l2_normalize(z, axis=1)# 归一化
            c_features.append(z)
        out_features = c_features# 执行论文的第一个两重循环，得到初始化的通道向量
        for l in range(self.iterations):
            c_attentions = []
            for i in range(self.channels):
                channel_f = out_features[i]# channel_f =
                c_attentions.append(self.parse_attention(self.adj, channel_f))# adj=1713*1713
            all_attentions = tf.concat([c_attentions[i] for i in range(len(c_attentions))], axis=2)
            all_attentions = tf.nn.softmax(all_attentions, axis=2)
            neg_all_attention = tf.zeros_like(all_attentions)
            adj_all = tf.tile(tf.expand_dims(self.adj, axis=2), [1, 1, self.channels])
            all_attentions = tf.where(adj_all > 0, all_attentions, neg_all_attention)
            for k in range(self.channels):
                feat = out_features[k]
                atte = tf.squeeze(all_attentions[:, :, k])
                out_features[k] = (tf.nn.l2_normalize(feat + tf.matmul(atte, feat), axis=1))
        output = tf.concat([out_features[i] for i in range(len(out_features))], axis=1)
        return output

    def parse_attention(self, adj, features):
        attention_matrix = tf.matmul(features, tf.transpose(features))
        neg_attention = tf.zeros_like(attention_matrix)
        attention_matrix = tf.where(adj > 0, attention_matrix, neg_attention)
        attention_matrix = attention_matrix * 1.0 / (self.beta)
        attention_matrix = tf.expand_dims(attention_matrix, axis=2)

        return attention_matrix
