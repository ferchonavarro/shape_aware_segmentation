import tensorflow as tf
import numpy as np
"""
Library fror auxiliary functions in Deep Learning
Autor: Fernando Navarro
Contact: fernando.navarro@tum.de
Created: 22.03.2018
"""

def crop_and_concat(x, name='concat'):
    """

    :param x: list of tensors to be concatenated, the order in the list is the order of concatenation
    :param name: name of the operation
    :return: concatened tensors in the third dimension
    """
    h_list=[]
    w_list = []
    for i in range(len(x)):
        h_list.append(x[i].get_shape().as_list()[1])
        w_list.append(x[i].get_shape().as_list()[2])

    # if h_list[0]  == h_list[1] and w_list[0] == w_list[1]:
    #     return tf.concat([x[0], x[1]], 3, name=name)
    # else:
    tensor_to_concat=[]
    h_shape = min(h_list)
    w_shape = min(w_list)

    for i in range(len(x)):
        aux_shape = x[i].get_shape().as_list()  # tf.shape(x2)
        if aux_shape[1] == h_shape and aux_shape[2] == w_shape:
            tensor_to_concat.append(x[i])
        else:
            offsets = [0, abs(h_shape - aux_shape[1]) // 2, abs((w_shape - aux_shape[2])) // 2, 0]
            size = [aux_shape[0], h_shape, w_shape, aux_shape[3]]
            tensor_to_concat.append(tf.slice(x[i], offsets, size))
    return tf.concat(tensor_to_concat, 3, name=name)

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
