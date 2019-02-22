import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.

(tr_img, tr_lbl), (tst_img, tst_lbl) = fashion_mnist.load_data()

def make_board_model(x, y):
    return tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(x, y)),
    tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(x*(y-1) + y*(x-1), activation=tf.nn.softmax)])

model = make_board_model(8, 8)


