import tensorflow as tf
EPSILON= 1e-08


def dice_loss(gt, logits,smooth=1e-7):
    """
    Computes the dice score between predictions (logits) and ground truth (gt)
    :param gt:  the one hot encoded version of the ground truth [batch_size,h,w,n_classes]
    :param logits: the output of the network without softmax [batch_size, h,w,n_classes]
    :param smooth: the value to avoid divide by zero optional
    :param weight_map: the weight map for class imbalance [batch_size, h,w,n_classes],
    it is a constant weighted map that is used to balance the classes
    :return:
    """
    # # intersect= tf.reduce_sum(tf.cast(tf.equal(gt, pred), tf.float32))
    # intersect = tf.reduce_sum(gt * pred)
    # union = smooth+ tf.reduce_sum(gt + pred)
    # dice= 2*intersect/ (union)
    # prediction = tf.nn.softmax(logits)
    flat_logits = tf.reshape(logits, [-1, 2])
    flat_labels = tf.reshape(gt, [-1, 2])
    intersection = tf.reduce_sum(flat_logits * flat_labels, axis=0)
    union = tf.reduce_sum(flat_logits, axis=0) + tf.reduce_sum(flat_labels, axis=0)
    dice = (2.0 * intersection) / (union +smooth)

    return 1-tf.reduce_mean(dice)


def weighted_crossentropy(gt,logits, class_weights=None, n_class=2):
    """
    Function to compute weighted crossentropy for segmentation
    Uses pure tensorflow functions without tf.nn.softmax_cross_entropy_with_logits
    :param gt: ground trueh tensor [batch_size,h,w,n_classes]
    :param logits: logits tensor [batch_size,h,w,n_classes]
    :param weight_map: weight map tensor [n_classes,1] as one hot encoded
    :return: crossentropy loss
    """
    #If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant.
    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(gt, [-1, n_class])

    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                          labels=flat_labels)

    if class_weights is not None:
        # flat_weights = tf.reshape(class_weights, [-1, 2])
        # weight_map = tf.multiply(flat_labels, flat_weights)
        weight_map= tf.matmul(flat_labels, class_weights, a_is_sparse=True)
        weight_map= tf.reshape(weight_map, [-1])
        # weight_map = tf.reduce_sum(weight_map, axis=1)
        weighted_loss = tf.multiply(loss_map, weight_map)
        loss = tf.reduce_mean(weighted_loss)
    else:
        loss = tf.reduce_mean(loss_map)

    return loss
    




