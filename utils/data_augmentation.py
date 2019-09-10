import tensorflow as tf
import math

"""
Augmentation Multitask
"""
def random_crop_multitask(slice, seg, edges, dist, height, width, channels):
    ## change the 20 to the porcentage u want to be resized
    height_new = height + tf.cast(
        tf.ceil( tf.cast(height, tf.float32) * tf.constant(15.0) / tf.constant(100.0)), tf.int32)
    width_new = width + tf.cast(
        tf.ceil( tf.cast(width, tf.float32) * tf.constant(15.0) / tf.constant(100.0)), tf.int32)

    slice = tf.image.resize_images(slice, [height_new, width_new])
    seg = tf.image.resize_images(seg, [height_new, width_new], method=1)
    edges = tf.image.resize_images(edges, [height_new, width_new], method=1)
    dist = tf.image.resize_images(dist, [height_new, width_new])

    combined = tf.concat([slice, seg, edges, dist], axis=2)
    ## channels +1
    combined = tf.random_crop(combined, size=[height, width, int(channels+3)])
    if channels== 1: ## grayscale image
        slice = tf.expand_dims(tf.gather(combined, 0, axis=2), axis=-1)
        seg = tf.expand_dims(tf.gather(combined, 1, axis=2), axis=-1)
        edges = tf.expand_dims(tf.gather(combined, 2, axis=2), axis=-1)
        dist = tf.expand_dims(tf.gather(combined, 3, axis=2), axis=-1)
    else: #color
        slice = tf.gather(combined, [0, 1, 2], axis=2)
        seg = tf.expand_dims(tf.gather(combined, 4, axis=2), axis=-1)
        edges = tf.expand_dims(tf.gather(combined, 5, axis=2), axis=-1)
        dist = tf.expand_dims(tf.gather(combined, 6, axis=2), axis=-1)
    return slice, seg, edges, dist


def random_rotation_multitask(slice, seg, edges, dist, channels=1):
    pi = tf.constant(math.pi)
    factor = tf.constant(180.0)
    # rotate counterclockwise randomly between zero and 10 degrees change the 10 for max val
    angle = tf.random_uniform((), 0., 10.)
    # combined = tf.concat([slice, seg, edges, dist], axis=2)
    slice = tf.contrib.image.rotate(slice, angle * pi / factor, interpolation='BILINEAR')
    seg = tf.contrib.image.rotate(seg, angle * pi / factor, interpolation='NEAREST')
    edges = tf.contrib.image.rotate(edges, angle * pi / factor, interpolation='NEAREST')
    dist = tf.contrib.image.rotate(dist, angle * pi / factor, interpolation='BILINEAR')

    return slice, seg, edges, dist


def random_zoom_multitask(slice, seg, edges, dist, height, width, channels=1):
    # dim1 = tf.constant(height)
    factor = tf.random_uniform((), 0.6, 1.1)
    new_height = tf.ceil(factor * tf.cast(height, tf.float32))
    new_height = tf.cast(new_height, tf.int32)

    # dim2 = tf.constant(width)
    new_width = tf.ceil(factor * tf.cast(width, tf.float32))
    new_width = tf.cast(new_width, tf.int32)

    combined = tf.concat([slice, seg, edges, dist], axis=2)
    combined = tf.image.resize_image_with_crop_or_pad(combined, new_height, new_width)

    if channels== 1: ## grayscale image
        slice= tf.expand_dims(tf.gather(combined,0,axis=2),axis=-1)
        seg = tf.expand_dims(tf.gather(combined,1, axis=2), axis=-1)
        edges = tf.expand_dims(tf.gather(combined, 2, axis=2), axis=-1)
        dist = tf.expand_dims(tf.gather(combined, 3, axis=2), axis=-1)
    else: #color
        slice= tf.gather(combined,[0,1,2],axis=2)
        seg = tf.expand_dims(tf.gather(combined, 4, axis=2), axis=-1)
        edges = tf.expand_dims(tf.gather(combined, 5, axis=2), axis=-1)
        dist = tf.expand_dims(tf.gather(combined, 6, axis=2), axis=-1)

    slice = tf.image.resize_images(slice, size= [height, width])
    slice = tf.saturate_cast(slice, dtype=tf.float32)

    seg = tf.image.resize_images(seg, size= [height, width], method=1)
    edges = tf.image.resize_images(edges, size= [height, width], method=1)
    dist = tf.image.resize_images(dist, size=[height, width])
    # seg = tf.saturate_cast(seg, dtype=tf.int64)

    return slice, seg, edges, dist


def original_multitask(slice, seg, edges, dist):
    return slice, seg, edges, dist


def augment_slice_multitask(slice, seg, edges, dist, height, width, channels):
    # Random crop
    # height= int(height)
    # width= int(width)
    # channels=int(channels)
    # channels= tf.constant(channels, dtype=tf.int64)
    coin = tf.less(tf.random_uniform((), 0., 1.), 0.5)
    slice, seg, edges, dist = tf.cond(coin, lambda: random_crop_multitask(slice, seg, edges, dist, height, width, channels), lambda: original_multitask(slice, seg, edges, dist))

    # Random Rotation
    coin = tf.less(tf.random_uniform((), 0., 1.), 0.5)

    slice, seg, edges, dist = tf.cond(coin, lambda: random_rotation_multitask(slice,seg, edges, dist, channels), lambda: original_multitask(slice, seg, edges, dist))

    coin = tf.less(tf.random_uniform((), 0., 1.), 0.5)

    # Random Zoom
    slice, seg, edges, dist = tf.cond(coin, lambda: random_zoom_multitask(slice, seg, edges, dist, height, width, channels), lambda: original_multitask(slice, seg, edges, dist))

    return slice, seg, edges, dist


