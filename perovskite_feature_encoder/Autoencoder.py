import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import initializers

initializer = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)


class Autoencoder(Model):
    def __init__(self, latent_dim,out_dims):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(16, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2), kernel_initializer=initializer),
            layers.Dense(8, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2), kernel_initializer=initializer),
            layers.Dense(4, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2), kernel_initializer=initializer),
            layers.Dense(latent_dim, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2),
                kernel_initializer=initializer)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(4, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2), kernel_initializer=initializer),
            layers.Dense(8, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2), kernel_initializer=initializer),
            layers.Dense(16, activation=tf.keras.layers.LeakyReLU(
                alpha=0.2),
                kernel_initializer=initializer),
            layers.Dense(out_dims, activation='sigmoid',
                         kernel_initializer=initializer)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
