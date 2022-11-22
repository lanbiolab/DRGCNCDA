import tensorflow as tf
from model import Model


class BilinearDiag(Model):
    X = None
    Y = None

    encoder_cache = {'train': None, 'test': None}

    def parse_settings(self):
        self.regularization_parameter = float(self.settings['RegularizationParameter'])

    def compute_codes(self, mode='train'):
        if self.encoder_cache[mode] is not None:
            return self.encoder_cache[mode]

        subject_codes, relation_codes, object_codes = self.next_component.get_all_codes(mode=mode)# 对应affine_transform.py
        print(self.X[:, 0], self.X[:, 1], self.X[:, 2])
        e1s = tf.nn.embedding_lookup(subject_codes, self.X[:, 0])# 对应论文中的Rd，在训练集中查找序号为0的张量
        rs = tf.nn.embedding_lookup(relation_codes, self.X[:, 1])# 对应论文中的r
        e2s = tf.nn.embedding_lookup(object_codes, self.X[:, 2])

        self.encoder_cache[mode] = (e1s, rs, e2s)
        return self.encoder_cache[mode]# 返回一个保存三元组的字典


    def get_loss(self, mode='train'):
        e1s, rs, e2s = self.compute_codes(mode=mode)

        energies = tf.reduce_sum(e1s * rs * e2s, 1)

        weight = int(self.settings['NegativeSampleRate'])
        weight = 1
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.Y, energies, weight))

    def local_initialize_train(self):
        self.Y = tf.placeholder(tf.float32, shape=[None])
        self.X = tf.placeholder(tf.int32, shape=[None, 3])

    def local_get_train_input_variables(self):
        return [self.X, self.Y]

    def local_get_test_input_variables(self):
        return [self.X]

    def predict(self):
        e1s, rs, e2s = self.compute_codes(mode='test')# 对应model.py中的得分函数
        print(e1s,e2s,rs)
        energies = tf.reduce_sum(e1s * rs * e2s, 1)# 计算张量中所有元素的和,按第二个维度进行求和
        return tf.nn.sigmoid(energies)

    def predict_all_subject_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_subject_codes = self.next_component.get_all_subject_codes(mode='test')
        all_energies = tf.transpose(tf.matmul(all_subject_codes, tf.transpose(rs * e2s)))
        return tf.nn.sigmoid(all_energies)

    def predict_all_object_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        print(type(e1s), e1s)
        all_object_codes = self.next_component.get_all_object_codes(mode='test')
        all_energies = tf.matmul(e1s * rs, tf.transpose(all_object_codes))
        return tf.nn.sigmoid(all_energies)

    def local_get_regularization(self):
        e1s, rs, e2s = self.compute_codes(mode='train')
        regularization = tf.reduce_mean(tf.square(e1s))
        regularization += tf.reduce_mean(tf.square(rs))
        regularization += tf.reduce_mean(tf.square(e2s))

        return self.regularization_parameter * regularization
