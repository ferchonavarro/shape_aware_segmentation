import sys
sys.path.append('./')
import tensorflow as tf
from utils.netutils import crop_and_concat, get_image_summary
from utils.losses import dice_loss, weighted_crossentropy
from utils.evaluation_metrics import dice_score
from collections import OrderedDict
import numpy as np
import logging

"""
Building function for the contracting part of the UNet
"""


def build_unet(x, is_training, summaries=True, root_filters=8, num_classes=2, branches=['SegBranch']):
    pools = OrderedDict() # visualization of filters after maxpooling
    deconv = OrderedDict() # visualization of filters after unpooling and concatenation
    dw_h_convs = OrderedDict() # visualization of filters  in encoders after last convolution and before maxpooling
    up_h_convs = OrderedDict() # visualization of filters  in decoders after last convolution and before unpooling
    convs = [] # pair of convolutions
    num_filters = root_filters
    #64
    with tf.variable_scope("Encoder1"):
        conv1= tf.layers.conv2d(x, num_filters, (3, 3), padding='same', name='conv1')
        bn1= tf.layers.batch_normalization(conv1, training=is_training, name='bn1')
        # drop1 = tf.layers.dropout(bn1, keep_prob)
        relu1= tf.nn.relu(bn1, name='relu1')
        conv2 = tf.layers.conv2d(relu1, num_filters, (3, 3), padding='same', name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, training=is_training, name='bn2')
        relu2 = tf.nn.relu(bn2, name='relu2')
        ## push the tensor for visualization
        end_point ='Encoder1-conv2'
        dw_h_convs[end_point]= relu2
        # drop1 = tf.layers.dropout(relu1, keep_prob)
        pool1= tf.layers.max_pooling2d(relu2, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool1')
        ## push pool layer for visualization
        end_point = 'Encoder1-pool1'
        pools[end_point] = pool1
        convs.append((conv1, conv2))

    num_filters = 2*num_filters
    #128
    with tf.variable_scope("Encoder2"):
        conv3 = tf.layers.conv2d(pool1, num_filters, (3, 3), padding='same', name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, training=is_training, name='bn3')
        relu3 = tf.nn.relu(bn3, name='relu3')
        conv4 = tf.layers.conv2d(relu3, num_filters, (3, 3), padding='same', name='conv4')
        bn4 = tf.layers.batch_normalization(conv4, training=is_training, name='bn4')
        relu4 = tf.nn.relu(bn4, name='relu4')
        ## push the tensor for visualization
        end_point = 'Encoder2-conv4'
        dw_h_convs[end_point] = relu4
        pool2 = tf.layers.max_pooling2d(relu4, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool2')
        ## push pool layer for visualization
        end_point = 'Encoder2-pool2'
        pools[end_point] = pool2
        convs.append((conv3, conv4))

    num_filters = 2 * num_filters
    #256
    with tf.variable_scope("Encoder3"):
        conv5 = tf.layers.conv2d(pool2, num_filters, (3, 3), padding='same', name='conv5')
        bn5 = tf.layers.batch_normalization(conv5, training=is_training, name='bn5')
        relu5 = tf.nn.relu(bn5, name='relu5')
        conv6 = tf.layers.conv2d(relu5, num_filters, (3, 3), padding='same', name='conv6')
        bn6 = tf.layers.batch_normalization(conv6, training=is_training, name='bn6')
        relu6 = tf.nn.relu(bn6, name='relu6')
        ## push the tensor for visualization
        end_point = 'Encoder3-conv6'
        dw_h_convs[end_point] = relu6
        pool3 = tf.layers.max_pooling2d(relu6, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool3')
        ## push pool layer for visualization
        end_point = 'Encoder3-pool3'
        pools[end_point] = pool3
        convs.append((conv5, conv6))

    num_filters = 2 * num_filters
    # 512
    with tf.variable_scope("Encoder4"):
        conv7 = tf.layers.conv2d(pool3, num_filters, (3, 3), padding='same', name='conv7')
        bn7 = tf.layers.batch_normalization(conv7, training=is_training, name='bn7')
        relu7 = tf.nn.relu(bn7, name='relu7')
        conv8 = tf.layers.conv2d(relu7, num_filters, (3, 3), padding='same', name='conv8')
        bn8 = tf.layers.batch_normalization(conv8, training=is_training, name='bn8')
        relu8 = tf.nn.relu(bn8, name='relu8')
        end_point = 'Encoder4-conv8'
        dw_h_convs[end_point] = relu8
        pool4 = tf.layers.max_pooling2d(relu8, pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool4')
        ## push pool layer for visualization
        end_point = 'Encoder4-pool4'
        pools[end_point] = pool4
        convs.append((conv7, conv8))

    num_filters = 2 * num_filters
    # 1024
    with tf.variable_scope("BottleNeck"):
        conv9= tf.layers.conv2d(pool4, num_filters, (3, 3), padding='same', name='conv9')
        bn9 = tf.layers.batch_normalization(conv9, training=is_training, name='bn9')
        relu9 = tf.nn.relu(bn9, name='relu9')
        conv10 = tf.layers.conv2d(relu9, num_filters, (3, 3), padding='same', name='conv10')
        bn10 = tf.layers.batch_normalization(conv10, training=is_training, name='bn10')
        relu10 = tf.nn.relu(bn10, name='relu10')
        # endpoint for visualization
        end_point = 'BottleNeck-conv10'
        dw_h_convs[end_point] = relu10
        convs.append((conv9, conv10))

    num_filters = int(num_filters/2)
    # 512
    with tf.variable_scope("Decoder1"):

        upconv1= tf.layers.conv2d_transpose(relu10, filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name='upconv1')
        concat1= crop_and_concat([relu8, upconv1], name= 'concat1')
        # print(concat1.get_shape())
        #for visualization
        end_point = 'Decoder1-upconv1'
        deconv[end_point] = concat1
        # concat1= tf.concat([upconv1, relu8], axis=-1, name="concat1")

        conv11 = tf.layers.conv2d(concat1, num_filters, (3, 3), padding='same', name='conv11')
        bn11 = tf.layers.batch_normalization(conv11, training=is_training, name='bn11')
        relu11 = tf.nn.relu(bn11, name='relu11')
        conv12 = tf.layers.conv2d(relu11, num_filters, (3, 3), padding='same', name='conv12')
        bn12 = tf.layers.batch_normalization(conv12, training=is_training, name='bn12')
        relu12 = tf.nn.relu(bn12, name='relu12')
        #visualization
        end_point = 'Decoder1-conv12'
        up_h_convs[end_point] = relu12
        convs.append((conv11, conv12))

    num_filters = int(num_filters / 2)
    # 256
    with tf.variable_scope("Decoder2"):

        upconv2= tf.layers.conv2d_transpose(relu12, filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name='upconv2')
        concat2 = crop_and_concat([relu6, upconv2], name='concat2')
        # for visualization
        end_point = 'Decoder2-upconv2'
        deconv[end_point] = concat2
        # concat2= tf.concat([upconv2, relu6], axis=-1, name="concat2")

        conv13 = tf.layers.conv2d(concat2, num_filters, (3, 3), padding='same', name='conv13')
        bn13 = tf.layers.batch_normalization(conv13, training=is_training, name='bn13')
        relu13 = tf.nn.relu(bn13, name='relu13')
        conv14 = tf.layers.conv2d(relu13, num_filters, (3, 3), padding='same', name='conv14')
        bn14 = tf.layers.batch_normalization(conv14, training=is_training, name='bn14')
        relu14 = tf.nn.relu(bn14, name='relu14')
        # visualization
        end_point = 'Decoder1-conv14'
        up_h_convs[end_point] = relu14
        convs.append((conv13, conv14))

    num_filters = int(num_filters / 2)
    # 128
    with tf.variable_scope("Decoder4"):

        upconv3= tf.layers.conv2d_transpose(relu14, filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name='upconv3')
        concat3 = crop_and_concat([relu4, upconv3], name='concat3')
        # for visualization
        end_point = 'Decoder3-upconv3'
        deconv[end_point] = concat3
        # concat3= tf.concat([upconv3, relu4], axis=-1, name="concat3")

        conv15 = tf.layers.conv2d(concat3, num_filters, (3, 3), padding='same', name='conv15')
        bn15 = tf.layers.batch_normalization(conv15, training=is_training, name='bn15')
        relu15 = tf.nn.relu(bn15, name='relu15')
        conv16 = tf.layers.conv2d(relu15, num_filters, (3, 3), padding='same', name='conv16')
        bn16 = tf.layers.batch_normalization(conv16, training=is_training, name='bn16')
        relu16 = tf.nn.relu(bn16, name='relu16')
        # visualization
        end_point = 'Decoder5-conv16'
        up_h_convs[end_point] = relu16
        convs.append((conv15, conv16))

    num_filters = int(num_filters / 2)
    # 64
    with tf.variable_scope("SegBranch"):
        upconv4= tf.layers.conv2d_transpose(relu16, filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name='upconv4')
        concat4 = crop_and_concat([relu2, upconv4], name='concat4')
        # for visualization
        end_point = 'Decoder5-upconv4'
        deconv[end_point] = concat4

        ## branch 1
        conv17 = tf.layers.conv2d(concat4, num_filters, (3, 3), padding='same', name='conv17')
        bn17 = tf.layers.batch_normalization(conv17, training=is_training, name='bn17')
        relu17 = tf.nn.relu(bn17, name='relu17')
        conv18 = tf.layers.conv2d(relu17, num_filters, (3, 3), padding='same', name='conv18')
        bn18 = tf.layers.batch_normalization(conv18, training=is_training, name='bn18')
        relu18 = tf.nn.relu(bn18, name='relu18')
        # visualization
        end_point = 'Decoder5-conv18'
        up_h_convs[end_point] = relu18
        convs.append((conv17, conv18))

        ### output map
        seg_map = tf.layers.conv2d(relu18, num_classes, (1, 1), padding='same', name='out_map')
        # out_map = tf.nn.relu(out_map)
        # for visualization
        end_point = 'seg_map'
        up_h_convs[end_point] = seg_map


    if "EdgesBranch" in branches:
        with tf.variable_scope("EdgesBranch"):
            ## branch 2
            conv17_2 = tf.layers.conv2d(concat4, num_filters, (3, 3), padding='same', name='conv17_2')
            bn17_2 = tf.layers.batch_normalization(conv17_2, training=is_training, name='bn17_2')
            relu17_2 = tf.nn.relu(bn17_2, name='relu17_2')
            conv18_2 = tf.layers.conv2d(relu17_2, num_filters, (3, 3), padding='same', name='conv18_2')
            bn18_2 = tf.layers.batch_normalization(conv18_2, training=is_training, name='conv18_2')
            relu18_2 = tf.nn.relu(bn18_2, name='relu18_2')
            # visualization
            end_point = 'Decoder5-conv18_2'
            up_h_convs[end_point] = relu18_2
            convs.append((conv17_2, conv18_2))

            ### output map
            edges_map = tf.layers.conv2d(relu18_2, 2, (1, 1), padding='same', name='edges_map')
            # out_map = tf.nn.relu(out_map)
            # for visualization
            end_point = 'edges_map'
            up_h_convs[end_point] = edges_map

    if "DistanceTransformBranch" in branches:
        with tf.variable_scope("DistanceTransformBranch"):
            ## branch 2
            conv17_3 = tf.layers.conv2d(concat4, num_filters, (3, 3), padding='same', name='conv17_3')
            bn17_3 = tf.layers.batch_normalization(conv17_3, training=is_training, name='bn17_3')
            relu17_3 = tf.nn.relu(bn17_3, name='relu17_3')
            conv18_3 = tf.layers.conv2d(relu17_3, num_filters, (3, 3), padding='same', name='conv18_3')
            bn18_3 = tf.layers.batch_normalization(conv18_3, training=is_training, name='conv18_3')
            relu18_3 = tf.nn.relu(bn18_3, name='relu18_3')
            # visualization
            end_point = 'Decoder5-conv18_3'
            up_h_convs[end_point] = relu18_3
            convs.append((conv17_3, conv18_3))

            ### output map
            dist_map = tf.layers.conv2d(relu18_3, 1, (1, 1), padding='same', name='dist_map')
            # out_map = tf.nn.relu(out_map)
            # for visualization
            end_point = 'dist_map'
            up_h_convs[end_point] = dist_map

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_{}'.format(k), get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_{}'.format(k), get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_{}/activations".format(k), dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_{}/activations".format(k), up_h_convs[k])
    
    if "DistanceTransformBranch" in branches and "EdgesBranch" in branches :
        return seg_map, edges_map, dist_map
    elif "DistanceTransformBranch" in branches and not("EdgesBranch" in branches) :
        return seg_map, dist_map, None
    elif "EdgesBranch" in branches and not("DistanceTransformBranch" in branches):
        return seg_map, edges_map, None
    else:
        return seg_map



class Unet(object):

    def __init__(self,
                 channels=1,
                 batch_size=2,
                 IMAGE_H=512,
                 IMAGE_W=512,
                 n_class=2,
                 cost="dice_coefficient",
                 root_filters=8,
                 branches=['SegBranch'],
                 cost_kwargs={}, **kwargs):
        # tf.reset_default_graph(

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        self.IMAGE_H= IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.batch_size = batch_size
        self.channels = channels
        self.branches = branches
        self.x = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_H, IMAGE_W, channels], name="x")
        self.y1 = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_H, IMAGE_W, n_class], name="y1")
        self.y2 = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_H, IMAGE_W, 2], name="y2")
        self.y3 = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_H, IMAGE_W, 1], name="y3")
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.logits1, self.logits2, self.logits3 = build_unet(x=self.x,
                                 is_training=self.is_training,
                                 summaries=self.summaries,
                                 root_filters=root_filters,
                                 num_classes=self.n_class, branches=self.branches)

        self.cost, self.losses = self._get_cost(cost,cost_kwargs)

        with tf.name_scope("dice"):
            self.dice = dice_score(self.y1,  self.logits1)

        self.predicter1 = tf.nn.softmax(self.logits1)
        if "EdgesBranch" in self.branches:
            self.predicter2 = tf.nn.softmax(self.logits2)
            self.correct_pred2 = tf.equal(tf.argmax(self.predicter2, 3), tf.argmax(self.y2, 3))
            self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_pred2, tf.float32))


        if "DistanceTransformBranch" in self.branches:
            self.predicter3 = tf.nn.softmax(self.logits3)
            self.error = tf.metrics.mean_squared_error(self.predicter3, self.y3)


        self.correct_pred1 = tf.equal(tf.argmax(self.predicter1, 3), tf.argmax(self.y1, 3))
        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_pred1, tf.float32))
        

    # def train_samples(self):
    #     return self.images_train, self.segs_train
    #
    # def eval_samples(self):
    #     return self.images_val, self.segs_val

    def _get_cost(self, cost_name, cost_kwargs):
        loss_aux = OrderedDict()
        with tf.name_scope("cost"):
            class_weights = cost_kwargs.pop("class_weights", None)
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
                loss1 = weighted_crossentropy(self.y1, self.logits1, n_class=self.n_class, class_weights=class_weights)
            else:
                loss1 = weighted_crossentropy(self.y1, self.logits1,  n_class=self.n_class, class_weights=None)

        loss_aux['segloss']= loss1

        ## edges loss
        if "EdgesBranch" in self.branches:
            loss2 = weighted_crossentropy(self.y2, self.logits2, n_class=2, class_weights=None)
            loss_aux['edgesloss'] = loss2
        ## distance loss
        if "DistanceTransformBranch" in self.branches:
            loss3 = tf.losses.mean_squared_error(self.y3, self.logits3)
            loss_aux['distloss'] = loss3

        ## total loss
        if "DistanceTransformBranch" in self.branches and "EdgesBranch" in self.branches :
            loss = loss1 +loss2 +loss3
            loss_aux['totalloss'] = loss
            return loss, loss_aux
        elif "DistanceTransformBranch" in self.branches and not("EdgesBranch" in self.branches) :
            loss = loss1+loss3
            loss_aux['totalloss'] = loss
            return loss, loss_aux
        elif "EdgesBranch" in self.branches and not("DistanceTransformBranch" in self.branches):
            loss = loss1+loss2
            loss_aux['totalloss'] = loss
            return loss, loss_aux

    def predict(self, sess, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """
        y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
        prediction, logits = sess.run([self.predicter1, self.logits1], feed_dict={self.x: x_test, self.y1: y_dummy, self.is_training:False})

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)




