import os
import tensorflow as tf
import sys
sys.path.append('./')
from skimage.exposure import rescale_intensity
from utils.visualization import labels2colors
from utils.visualization import pretty_plot_confusion_matrix
import numpy as np
import shutil
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
import math
from pandas import DataFrame

"""
Trainer class to train any segmentation network

"""


class Trainer(object):
    """
    Trains a segmentation instance:
    used to train and validate

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param verification_batch_size: size of verification batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """

    def __init__(self, net, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.current_dice= 0
        self.epsilon= 1e-7
        self.train_epoch= -1

    def _get_optimizer(self, decay_step, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            if self.optimizer == "momentum":
                learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
                decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
                momentum = self.opt_kwargs.pop("momentum", 0.2)

                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step,
                                                                     decay_steps=decay_step,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)

                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                       **self.opt_kwargs).minimize(self.net.cost,
                                                                                   global_step=global_step)
            elif self.optimizer == "adam":
                learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
                ## using exponential decay in Adam
                decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step,
                                                                     decay_steps=decay_step,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)


                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=global_step)
            else:
                raise ValueError("Unknown optimizar name: " % self.optimizer)

        return optimizer

    def _initialize(self, decay_step, output_path, restore):
        global_step = tf.Variable(0, name="global_step")

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('dice', self.net.dice)
        tf.summary.scalar('accuracy', self.net.accuracy1)
        tf.summary.image('images', self.net.x, max_outputs=3)

        ##todo: add summaries for images and segmentations

        self.optimizer = self._get_optimizer(decay_step, global_step)


        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()

        ## summaries computed as the average
        self.avg_loss = tf.placeholder(tf.float32)
        tf.summary.scalar('avg_loss', self.avg_loss, collections=['average_eval'])

        self.avg_dice = tf.placeholder(tf.float32)
        tf.summary.scalar('avg_dice', self.avg_dice, collections=['average_eval'])

        self.avg_acc = tf.placeholder(tf.float32)
        tf.summary.scalar('avg_acc', self.avg_acc, collections=['average_eval'])

        self.summary_op_avg = tf.summary.merge_all('average_eval')

        init = tf.global_variables_initializer()

        self.prediction_path = os.path.join(output_path, 'predictions')
        self.output_path = output_path
        self.cm_path = os.path.join(output_path, 'cm')


        if not restore:
            logging.info("Removing '{:}'".format(self.prediction_path))
            shutil.rmtree(self.prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(self.output_path):
            logging.info("Allocating '{:}'".format(self.output_path))
            os.makedirs(output_path)

        if not os.path.exists(self.prediction_path):
            logging.info("Allocating '{:}'".format(self.prediction_path))
            os.makedirs(self.prediction_path)

        if not os.path.exists(self.cm_path):
            logging.info("Allocating '{:}'".format(self.cm_path))
            os.makedirs(self.cm_path)

        return init

    def plot_predictions(self, batch_x, batch_seg, batch_edges, batch_dist, pred_segs, pred_edges, pred_dists, name, index):

        x = np.squeeze(batch_x)
        yseg = np.argmax(batch_seg, axis=3)
        edges = np.argmax(batch_edges, axis=3)
        dist = np.squeeze(batch_dist)

        pred_seg = np.argmax(pred_segs, axis=3)
        pred_edges = np.argmax(pred_edges, axis=3)
        pred_dist = np.squeeze(pred_dists)
        batch_size = np.shape(x)[0]
        for i in range(batch_size):
            fig, ax = plt.subplots(2, 4)
            ## convert to colors
            aux_slice = rescale_intensity(x[i, :, :], out_range=(0.0, 1.0))
            aux_seg = labels2colors(yseg[i, :, :])
            aux_edges = labels2colors(edges[i, :, :])
            aux_dist = dist[i, :, :]

            aux_segpred = labels2colors(pred_seg[i, :, :])
            aux_edgespred = labels2colors(pred_edges[i, :, :])
            aux_distpred = pred_dist[i, :, :]



            ax[0,0].imshow(aux_slice, cmap='gray')
            ax[0,1].imshow(aux_seg)
            ax[0,2].imshow(aux_edges)
            ax[0,3].imshow(aux_dist, cmap='jet')

            ax[1,1].imshow(aux_segpred)
            ax[1,2].imshow(aux_edgespred)
            ax[1,3].imshow(aux_distpred, cmap='jet')


            ax[0,0].set_title("Input")
            ax[0,1].set_title("GT Seg")
            ax[0,2].set_title("GT Egdes")
            ax[0,3].set_title("GT Dist")

            ax[1,1].set_title("Pred Seg")
            ax[1,2].set_title("Pred Egdes")
            ax[1,3].set_title("Pred Dist")


            ax[0,0].set_xticks([])
            ax[0,1].set_xticks([])
            ax[0,2].set_xticks([])
            ax[0,3].set_xticks([])
            ax[1,0].set_xticks([])
            ax[1,1].set_xticks([])
            ax[1,2].set_xticks([])
            ax[1,3].set_xticks([])


            ax[0,0].set_yticks([])
            ax[0,1].set_yticks([])
            ax[0,2].set_yticks([])
            ax[0,3].set_yticks([])
            ax[1,0].set_yticks([])
            ax[1,1].set_yticks([])
            ax[1,2].set_yticks([])
            ax[1,3].set_yticks([])
            _name = '{}_batch_{}_image_{}.png'.format(name, index, i)
            file_name = os.path.join(self.prediction_path, _name)
            plt.savefig(file_name)
            
    def confusion_matrix(self, batch_y, y_pred, name):
        y_pred = np.argmax(y_pred, axis=3)
        y_pred = y_pred.flatten()
        y_true = np.argmax(batch_y, axis=3)
        y_true = y_true.flatten()

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.labels)

        ### eliminate nans
        cm = np.nan_to_num(cm)

        df_cm = DataFrame(cm, index=self.target_names, columns=self.target_names)
        # colormap: see this and choose your more dear
        cmap = 'PuRd'
        fz = 4;
        figsize = [12, 12];
        show_null_values = 2

        # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        _name = '{}_cm.png'.format(name)
        file_name = os.path.join(self.cm_path, _name)

        pretty_plot_confusion_matrix(df_cm, cmap=cmap, name=file_name, fz=fz, figsize=figsize,
                                     show_null_values=show_null_values)


        ## send data to excel file
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df = DataFrame(cm_norm, index=self.target_names, columns=self.target_names)
        _name = '{}_cm.xlsx'.format(name)
        file_name = os.path.join(self.cm_path, _name)
        df.to_excel(file_name)

        return True

    def evaluate_model(self, iter,sess,epsilon = 0.000001):
        sum_loss =[]
        sum_dice =[]
        sum_acc =[]
        print('Evaluating model ...')
        for ival in range(self.iterations2epoch_val):

            batch_x, batch_y, batch_edges, batch_dist = self.val_provider.next_batch()
            if (ival+1)* self.val_provider.batch_size <= self.n_plots:
            # if ival == 0:
                if "DistanceTransformBranch" in self.branches and "EdgesBranch" in self.branches :
                    summary, pred_seg, pred_edges, pred_dist, loss, dice, acc = sess.run((self.summary_op,
                                                                  self.net.predicter1, self.net.predicter2,
                                                                  self.net.predicter3,
                                                                  self.net.cost,
                                                                  self.net.dice,
                                                                  self.net.accuracy1),
                                                                 feed_dict={self.net.x: batch_x,
                                                                            self.net.y1: batch_y,
                                                                            self.net.y2: batch_edges,
                                                                            self.net.y3: batch_dist,
                                                                            self.net.is_training: False})

                

                elif "DistanceTransformBranch" in self.branches and not("EdgesBranch" in self.branches):
                    summary, pred_seg, pred_dist, loss, dice, acc = sess.run((self.summary_op,
                                                                  self.net.predicter1,
                                                                  self.net.predicter3,
                                                                  self.net.cost,
                                                                  self.net.dice,
                                                                  self.net.accuracy1),
                                                                 feed_dict={self.net.x: batch_x,
                                                                            self.net.y1: batch_y,
                                                                            self.net.y2: batch_edges,
                                                                            self.net.y3: batch_dist,
                                                                            self.net.is_training: False})

                    pred_edges=pred_seg
        
                elif "EdgesBranch" in self.branches and not("DistanceTransformBranch" in self.branches):
                    summary, pred_seg, pred_edges, loss, dice, acc = sess.run((self.summary_op,
                                                                  self.net.predicter1,
                                                                  self.net.predicter2,
                                                                  self.net.cost,
                                                                  self.net.dice,
                                                                  self.net.accuracy1),
                                                                 feed_dict={self.net.x: batch_x,
                                                                            self.net.y1: batch_y,
                                                                            self.net.y2: batch_edges,
                                                                            self.net.y3: batch_dist,
                                                                            self.net.is_training: False})
                    pred_dist=pred_seg
                else:
                    summary, pred_seg, loss, dice, acc = sess.run((self.summary_op,
                                                                  self.net.predicter1,
                                                                  self.net.cost,
                                                                  self.net.dice,
                                                                  self.net.accuracy1),
                                                                 feed_dict={self.net.x: batch_x,
                                                                            self.net.y1: batch_y,
                                                                            self.net.y2: batch_edges,
                                                                            self.net.y3: batch_dist,
                                                                            self.net.is_training: False})
                    pred_edges=pred_seg
                    pred_dist=pred_seg
        

                


                self.val_writer.add_summary(summary, iter)
                self.val_writer.flush()

                sum_loss.append(loss)
                sum_dice.append(dice)
                sum_acc.append(acc)

                name= "epoch_%s" % self.train_epoch
                self.plot_predictions(batch_x, batch_y, batch_edges, batch_dist,pred_seg, pred_edges, pred_dist, name, ival)
                self.confusion_matrix(batch_y, pred_seg, name)
                print("Minibach Validation Epoch {:}, Iter {:}, Minibatch Loss= {:.4f}, Minibatch Dice= {:.4f}, Minibatch accuracy= {:.4f}".format(self.train_epoch, iter, loss, dice, acc))
            else:
                loss, dice, acc = sess.run((self.net.cost,
                                                 self.net.dice,
                                                 self.net.accuracy1),
                                                feed_dict={self.net.x: batch_x,
                                                           self.net.y1: batch_y,
                                                           self.net.y2: batch_edges,
                                                           self.net.y3: batch_dist,
                                                           self.net.is_training: False})
                sum_loss.append(loss)
                sum_dice.append(dice)
                sum_acc.append(acc)

        nval = len(sum_loss)
        avg_loss= sum(sum_loss) / (nval + epsilon)
        avg_dice = sum(sum_dice) / (nval + epsilon)
        avg_acc = sum(sum_acc) / (nval + epsilon)

        summary = sess.run(self.summary_op_avg,
                           feed_dict={self.avg_loss: avg_loss,
                                      self.avg_dice: avg_dice,
                                      self.avg_acc: avg_acc})

        print("Validation Stats Epoch {:}, Iter {:}, Loss= {:.4f},  Dice= {:.4f},  accuracy= {:.4f}".format( self.train_epoch, iter, avg_loss, avg_dice, avg_acc))

        self.val_writer.add_summary(summary, iter)
        self.val_writer.flush()
        self.train_epoch += 1

        if avg_dice >= self.current_dice:
            name = "epoch_%s" % self.train_epoch
            ckpt_path = os.path.join(self.output_path, name + '_model_iter_' + str(iter) + '.ckpt')
            model_path = self.net.save(sess, ckpt_path)
            print("Model saved in : ", model_path)
            self.current_dice = avg_dice

    def train_val(self,
                  train_provider,
                  val_provider,
                  sess,
                  output_path,
                  labels= [0,2],
                  decay_step=10,
                  epochs=20,
                  display_step=20,
                  evaluate_model=1,
                  restore=False,
                  write_graph=False,
                  target_names= None,
                  n_plots=4, branches=['SegBranch']):
        """
        Lauches the training process

        :param train_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        self.output_path= output_path
        self.train_provider = train_provider
        self.val_provider = val_provider
        self.target_names = target_names
        self.labels = labels
        self.n_plots = n_plots
        self.branches= branches

        start_time = datetime.now()

        self.iterations2epoch_train = int(math.ceil(self.train_provider.images2epoch / self.net.batch_size)) ## number of iterations needed for one epoch in training set
        

        self.iterations2epoch_val = int(math.ceil(self.val_provider.images2epoch / self.net.batch_size)) # number of iteration neede for one epoch in validation dataset
        
        self.train_iters= self.iterations2epoch_train * epochs # total number of iterations to be trained
        self.evaluate= self.iterations2epoch_train * evaluate_model # evaluate the model every n iterations
        self.display_step = display_step
        decay_every_iter= self.iterations2epoch_train * decay_step # decay the learning rate every n iteration converts epochs to iterations

        init = self._initialize(decay_every_iter, output_path, restore)

        if write_graph:
            tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if restore:
            print("Restoring from last checkpoint")
            ckpt = tf.train.get_checkpoint_state(output_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.net.restore(sess, ckpt.model_checkpoint_path)

        self.train_writer = tf.summary.FileWriter(output_path + '/train', graph=sess.graph)
        self.val_writer = tf.summary.FileWriter(output_path + '/val')

        print("Starting optimization ...")
        sum_loss_train = []
        sum_dice_train = []
        sum_acc_train = []


        for iter in range(self.train_iters):

            if iter % self.evaluate == 0:  # when x number of epoch are completed: do validation and save model if better

                self.evaluate_model(iter, sess)

                ## display statistic on loss, dice and accuracy of the current epoch
                nval = len(sum_loss_train)
                avg_loss_train = sum(sum_loss_train) / (nval + self.epsilon)
                avg_dice_train = sum(sum_dice_train) / (nval + self.epsilon)
                avg_acc_train = sum(sum_acc_train) / (nval + self.epsilon)

                summary = sess.run(self.summary_op_avg,
                                   feed_dict={self.avg_loss: avg_loss_train,
                                              self.avg_dice: avg_dice_train,
                                              self.avg_acc: avg_acc_train})

                self.train_writer.add_summary(summary, iter)
                self.train_writer.flush()

                print("Train Stats Epoch {:}, Iter {:}, Loss= {:.4f},  Dice= {:.4f},  accuracy= {:.4f}".format(
                    self.train_epoch, iter, avg_loss_train, avg_dice_train, avg_acc_train))

                sum_loss_train = []
                sum_dice_train = []
                sum_acc_train = []

            if iter % self.display_step == 0: ## display minibatch statistics
            
                batch_x, batch_y, batch_edges, batch_dist = self.train_provider.next_batch()
                
                _, summary, loss, dice, acc = sess.run((self.optimizer,
                                                             self.summary_op,
                                                             self.net.cost,
                                                             self.net.dice,
                                                             self.net.accuracy1),
                                                            feed_dict={self.net.x: batch_x,
                                                                       self.net.y1: batch_y,
                                                                       self.net.y2: batch_edges,
                                                                       self.net.y3: batch_dist,
                                                                       self.net.is_training: True})
                sum_loss_train.append(loss)
                sum_dice_train.append(dice)
                sum_acc_train.append(acc)

                self.train_writer.add_summary(summary, iter)
                self.train_writer.flush()
                print("Training Epoch {:}, Iter {:}, Minibatch Loss= {:.4f}, Minibatch Dice= {:.4f}, Minibatch Accuracy= {:.4f}"
                      .format(self.train_epoch,iter,loss,dice, acc))
            else:
                batch_x, batch_y, batch_edges, batch_dist = self.train_provider.next_batch()
                
                _, loss, dice, acc = sess.run((self.optimizer,
                                               self.net.cost,
                                               self.net.dice,
                                               self.net.accuracy1),
                                              feed_dict={self.net.x: batch_x,
                                                         self.net.y1: batch_y,
                                                         self.net.y2: batch_edges,
                                                         self.net.y3: batch_dist,
                                                         self.net.is_training: True})

                sum_loss_train.append(loss)
                sum_dice_train.append(dice)
                sum_acc_train.append(acc)

        print("Optimization Finished!")
        end_time = datetime.now()
        print('Network Trained for : {}'.format(end_time - start_time))
        # stop the coordinator
        coord.request_stop()
        coord.join(threads)
