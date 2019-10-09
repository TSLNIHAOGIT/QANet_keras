# ! -*- coding: utf-8 -*-
from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K

class LayerDropout(Layer):
    def __init__(self, dropout = 0.1, **kwargs):
        self.dropout = dropout
        super(LayerDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LayerDropout, self).build(input_shape)
    #0.2 should be self.dropout，但这里总是出错，就先改成0.2
    def call(self, x, mask=None, training=None):
        x, residual = x
        pred = tf.random.uniform([]) < self.dropout
        #print('self.dropout',self.dropout)
        x_train = tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(x, 1.0 -0.2 ) + residual)
        x_test = x + residual
        return K.in_train_phase(x_train, x_test, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape