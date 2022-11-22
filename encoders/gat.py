# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RelationPrediction-master 
@File    ：gat.py.py
@IDE     ：PyCharm 
@Author  ：Firo
@Date    ：2021/11/1 20:35 
'''
import tensorflow as tf
import numpy as np
# from encoder.message_gcn_undirect import MessageGat

class gat():
    def __init__(self, triples, settings):
        self.triples = np.array(triples)
        self.entity_count = settings['EntityCount']
        self.relation_count = settings['RelationCount']
        self.edge_count = self.triples.shape[0]
    def adj_matrix(self,triplets):
        adj_triplets = triplets
        adj_matrix = np.zeros((self.entity_count, self.entity_count))
        for triplet in adj_triplets:
            adj_matrix[triplet[0]][triplet[2]] = 1
            adj_matrix[triplet[2]][triplet[0]] = 1
        return adj_matrix

    def adj_to_bias(self, sizes, nhood=1):# 对应bias_matrix将adj转化为对应的只有邻接注意力系数的矩阵
        adj = self.adj_matrix(self.triples)
        nb_graphs = adj.shape[0]
        mt = np.empty(adj.shape)
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])
            for _ in range(nhood):
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
        return -1e9 * (1.0 - mt)

    def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation

    def inference(self):
        n_heads = [8, 1]
        bias_mat = self.adj_to_bias()
        hid_units = [8]
        activation = tf.nn.elu
        ffd_drop = 0.0
        attn_drop = 0.0
        attns = []
        for _ in range(n_heads[0]):
            attns.append(self.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(self.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits
# class GAT():
#     onehot_input = True
#     use_nonlinearity = True
#     # vertex_embedding_function = {'train': None, 'test': None}
#
#     def __init__(self, shape, settings, next_component=None, onehot_input=False, use_nonlinearity=True):
#         self.onehot_input = onehot_input
#         self.use_nonlinearity = use_nonlinearity
#         self.shape = shape
#         Model.__init__(self, next_component, settings)
#
#     def needs_graph(self):
#         return True
#     # GAT代码中注意点1.返回值必须是三个矩阵code[0], None, code[1]；2.
#     def get_all_codes(self, code = "train"):
#         return
#
#     def attn_head(self, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
#         # conv1d = tf.layers.conv1d
#         with tf.name_scope('my_attn'):
#             if in_drop != 0.0:
#                 seq = tf.nn.dropout(seq, 1.0 - in_drop)# 防止过拟合，扔掉一部分神经元
#
#             seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)# seq表示输入的特征, out_sz表示输出空间的维度, kernel_size表示一维卷积窗口的长度
#
#             # simplest self-attention possible
#             f_1 = tf.layers.conv1d(seq_fts, 1, 1)
#             f_2 = tf.layers.conv1d(seq_fts, 1, 1)
#             logits = f_1 + tf.transpose(f_2, [0, 2, 1])# [0, 2, 1]进行矩阵的转置, 不管第一个维度，将第二个和第三个维度转置
#             coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)# 将得到的注意力系数进行归一化
#
#             if coef_drop != 0.0:
#                 coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
#             if in_drop != 0.0:
#                 seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
#
#             vals = tf.matmul(coefs, seq_fts)# 用新的注意力系数改变权重
#             ret = tf.contrib.layers.bias_add(vals)# 正则化函数
#
#             # residual connection
#             # if residual:
#             #     if seq.shape[-1] != ret.shape[-1]:
#             #         ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
#             #     else:
#             #         ret = ret + seq
#
#             return activation(ret)  # activation
#
#     def inference(self, inputs, nb_classes, attn_drop, ffd_drop,
#                   bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
#         attns = []
#         for _ in range(n_heads[0]):# 多头注意力
#             attns.append(self.attn_head(inputs, bias_mat=bias_mat,
#                                           out_sz=hid_units[0], activation=activation,
#                                           in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
#         h_1 = tf.concat(attns, axis=-1)# 拼接操作
#         for i in range(1, len(hid_units)):
#             # h_old = h_1
#             attns = []
#             for _ in range(n_heads[i]):
#                 attns.append(self.attn_head(h_1, bias_mat=bias_mat,
#                                               out_sz=hid_units[i], activation=activation,
#                                               in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
#             h_1 = tf.concat(attns, axis=-1)
#         out = []
#         for i in range(n_heads[-1]):
#             out.append(self.attn_head(h_1, bias_mat=bias_mat,
#                                         out_sz=nb_classes, activation=lambda x: x,
#                                         in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
#         logits = tf.add_n(out) / n_heads[-1]
#
#         return logits
#
# class graph_representation(Model):
#     normalization = "global"
#     graph = None
#     X = None
#
#     def __init__(self, triples, settings):
#         self.triples = np.array(triples)
#         self.entity_count = settings['EntityCount']
#         self.relation_count = settings['RelationCount']
#         self.edge_count = self.triples.shape[0]
#
#         # self.process(self.triples)
#
#         # self.graph = None#MessageGraph(triples, self.entity_count, self.relation_count)
#
#     def adj_to_bias(self, adj, sizes, nhood=1):
#         nb_graphs = adj.shape[0]
#         mt = np.empty(adj.shape)
#         for g in range(nb_graphs):
#             mt[g] = np.eye(adj.shape[1])
#             for _ in range(nhood):
#                 mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
#             for i in range(sizes[g]):
#                 for j in range(sizes[g]):
#                     if mt[g][i][j] > 0.0:
#                         mt[g][i][j] = 1.0
#         return -1e9 * (1.0 - mt)
#
#     def get_adj(self):
#         # adj_list = [[] for _ in settings['entities']]
#         # for i, triplet in enumerate(train_triplets):
#         #     adj_list[train_triplets[0]].append([i, train_triplets[2]])
#         adj_matrix = np.zeros((self.entity_count, self.entity_count))
#
#         for triplet in self.triples:
#             adj_matrix[triplet[0]][triplet[2]] = 1
#             adj_matrix[triplet[2]][triplet[1]] = 1
#
#         return adj_matrix
#
#     def get_graph(self):
#         adj_matrix = self.get_adj()
#         adj_bias = self.adj_to_bias(adj_matrix, 1, 1)
#         if self.graph is None:
#             self.graph = GAT(self.X, self.entity_count, self.relation_count)
#
#         return self.graph
#
#     def local_initialize_train(self):
#         self.X = tf.placeholder(tf.int32, shape=[None, 3], name='graph_edges')
#
#     def local_get_train_input_variables(self):
#         return [self.X]
#
#     def local_get_test_input_variables(self):
#         return [self.X]