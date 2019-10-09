# ! -*- coding: utf-8 -*-
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import *
import tensorflow as tf
import tensorflow.keras.backend as K

class QAoutputBlock(Layer):
    def __init__(self, ans_limit=30, **kwargs):
        self.ans_limit = ans_limit
        super(QAoutputBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        super(QAoutputBlock, self).build(input_shape)

    def call(self, x, mask=None):
        x1 ,x2 = x
        outer = tf.matmul(tf.expand_dims(x1, axis=2), tf.expand_dims(x2, axis=1))
        outer = tf.linalg.band_part(outer, 0, self.ans_limit)
        output1 = tf.reshape(tf.cast(tf.argmax(tf.reduce_max(outer, axis=2), axis=1), tf.float32),(-1,1))
        output2 = tf.reshape(tf.cast(tf.argmax(tf.reduce_max(outer, axis=1), axis=1), tf.float32),(-1,1))

        return [output1, output2]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0],1), (input_shape[0][0],1)]