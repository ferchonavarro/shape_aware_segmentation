import tensorflow as tf

"""
Evaluation Metrics for networks in tensorflow

"""

def accuracy(logits, y_true):

    pred = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(pred, 3), tf.argmax(y_true, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def dice_score(gt, logits,smooth=1e-7):
    """
    Computes the dice score between predictions (logits) and ground truth (gt)
    :param gt:  the one hot encoded version of the ground truth [batch_size,h,w,n_classes]
    :param logits: the output of the network without softmax [batch_size, h,w,n_classes]
    :param smooth: the value to avoid divide by zero optional
    :return:
    """
    # # intersect= tf.reduce_sum(tf.cast(tf.equal(gt, pred), tf.float32))
    # intersect = tf.reduce_sum(gt * pred)
    # union = smooth+ tf.reduce_sum(gt + pred)
    # dice= 2*intersect/ (union)
    prediction = tf.nn.softmax(logits)
    prediction = prediction[:,:,:,1:]
    gt = gt[:,:,:,1:]
    intersection = tf.reduce_sum(gt * prediction)
    union = tf.reduce_sum(prediction) + tf.reduce_sum(gt)
    dice = (2.0 * intersection) / (union +smooth)

    return tf.reduce_mean(dice)
