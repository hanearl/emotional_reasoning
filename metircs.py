import tensorflow as tf
from tensorflow import keras


class MultiLabelAccuracy(keras.metrics.Metric):
    def __init__(self, name='multi_label_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.union = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        union = tf.reduce_sum(tf.cast((y_pred >= 0.5) | (y_true == 1), dtype=tf.float32))
        self.inter.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return self.inter/(self.union + 1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.union.assign(0.)


class MultiLabelPrecision(keras.metrics.Metric):
    def __init__(self, name='multi_label_precision', **kwargs):
        super(MultiLabelPrecision, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.pred_sum = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        pred_sum = tf.reduce_sum(tf.cast(y_pred >= 0.5, dtype=tf.float32))
        self.inter.assign_add(inter)
        self.pred_sum.assign_add(pred_sum)

    def result(self):
        return self.inter / (self.pred_sum + 1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.pred_sum.assign(0.)


class MultiLabelRecall(keras.metrics.Metric):
    def __init__(self, name='multi_label_recall', **kwargs):
        super(MultiLabelRecall, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.true_sum = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        true_sum = tf.reduce_sum(tf.cast((y_true == 1), dtype=tf.float32))
        self.inter.assign_add(inter)
        self.true_sum.assign_add(true_sum)

    def result(self):
        return self.inter/(self.true_sum+1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.true_sum.assign(0.)


class MultiLabelF1(keras.metrics.Metric):
    def __init__(self, name='multi_label_f1_score', **kwargs):
        super(MultiLabelF1, self).__init__(name=name, **kwargs)
        self.inter = self.add_weight(name='tp', initializer='zeros')
        self.true_sum = self.add_weight(name='tp', initializer='zeros')
        self.pred_sum = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter = tf.reduce_sum(tf.cast((y_pred >= 0.5) & (y_true == 1), dtype=tf.float32))
        true_sum = tf.reduce_sum(tf.cast(y_true == 1, dtype=tf.float32))
        pred_sum = tf.reduce_sum(tf.cast(y_pred >= 0.5, dtype=tf.float32))
        self.inter.assign_add(inter)
        self.true_sum.assign_add(true_sum)
        self.pred_sum.assign_add(pred_sum)

    def result(self):
        return (2 * self.inter) / (self.pred_sum + self.true_sum + 1e-8)

    def reset_states(self):
        self.inter.assign(0.)
        self.true_sum.assign(0.)
        self.pred_sum.assign(0.)
