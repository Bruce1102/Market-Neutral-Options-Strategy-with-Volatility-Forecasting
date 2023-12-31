import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class Dain_Gateless(layers.Layer):
    def __init__(self, name="dain_gateless", **kwargs):
        # input tensor: (batch_size, rows, n_features)
        super(Dain_Gateless, self).__init__(name=name, **kwargs)
        self.eps = 1e-8

        self.mean_layer = layers.Dense(
            1, use_bias=False, kernel_initializer="identity", name="dain-mean"
        )
        self.scaling_layer = layers.Dense(
            1, use_bias=False, kernel_initializer="identity", name="dain-scale"
        )
        # self.gating_layer = layers.Dense(
        #     n_features, activation="sigmoid", name="dain-gate"
        # )

        # self.transpose = layers.Permute((2, 1))
        # self.reshape_2d = layers.Reshape((dim, n_features))

    def call(self, inputs):
        # step 1: adapative average
        # from (batch, rows, n_features) to (batch, n_features, rows)
        # inputs = self.transpose(inputs)
        avg = K.mean(inputs, axis=(1, 2), keepdims=True)
        adaptive_avg = self.mean_layer(avg)
        # adaptive_avg = K.reshape(adaptive_avg, (-1, 1, 1))
        inputs -= adaptive_avg

        # # step 2: adapative scaling
        std = K.mean(inputs ** 2, axis=(1, 2), keepdims=True)
        std = K.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        # fn = lambda elem: K.switch(K.less_equal(elem, 1.0), K.ones_like(elem), elem)
        # adaptive_std = K.map_fn(fn, adaptive_std)
        adaptive_std = tf.where(tf.math.less_equal(
            adaptive_std, self.eps), tf.ones_like(adaptive_std), adaptive_std)
        # adaptive_std = K.reshape(adaptive_std, (-1, self.n_features, 1))
        inputs /= adaptive_std

        # # step 3: gating
        # avg = K.mean(inputs, axis=2)
        # gate = self.gating_layer(avg)
        # gate = K.reshape(gate, (-1, self.n_features, 1))
        # inputs *= gate
        # from (batch, n_features, rows) => (batch, rows, n_features)
        # inputs = self.transpose(inputs)

        return inputs

    def get_config(self):

        config = super().get_config().copy()
        config.update({'name': self.name})
        return config
