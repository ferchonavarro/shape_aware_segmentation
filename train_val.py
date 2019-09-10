import sys
import tensorflow as tf
#sys.path.append("add path to the project main folder")
from data_provider import DataProviderMultitask
from utils.visualization import target_names
import numpy as np
import unet
import seg_trainer

"""
Run the script data2tfrecords.py before this script
Train a U-Net with complementary task learning
"""
nx = 512 ## input size width
ny = 512 ## input size high
tf_file_train = "path to your train.tfrecords"
tf_file_val = "path to your val.tfrecords"
output_path = "path to folder for tensorboard logging"
aug_images_train = 1  # number of images to generate from every single image in dataset
aug_images_val = 1
batch_size = 2
n_plots = 10 # how many images with predictions to show after every epoch
num_images_train = 47333 # number of images in train set
num_images_val = 838 # number of images in val set
branches=['DistanceTransformBranch','EdgesBranch', 'SegBranch']

## weights for imbalance segmentation
weights_array = [1.00000000e+00, 5.74720964e+01, 4.67654400e+02, 1.58137518e+03,
                 7.14735486e+03, 6.28030394e+02, 4.66110436e+02, 3.68521725e+03,
                 4.96670375e+01, 5.63883689e+01, 1.83480061e+03, 5.38223611e+03,
                 2.16616129e+03, 6.64989274e+02, 6.01810566e+02, 3.01955143e+04,
                 2.15210323e+04, 5.88140475e+02, 6.42542047e+02, 6.44860816e+02,
                 6.14258756e+02]
weights_array = np.asanyarray(weights_array)

num_classes = 21 # number of classes including backrground
weights_array = np.reshape(weights_array, (num_classes, 1))
print('Weights shape: {}'.format(weights_array.shape))

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # config = tfigProto()
    # config.gpu_options.visible_device_list = "0"
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #
    # sess = tf.Session()
    # sess = tf.Session(graph=tf.get_default_graph(), config=config)
    train_generator = DataProviderMultitask(tf_records_file=tf_file_train,
                                   sess=sess, images2epoch=num_images_train * aug_images_train, IMAGE_H=nx,
                                   IMAGE_W=ny, batch_size=batch_size, channels=1, shuffle_flag=True, augment=True)

    val_generator = DataProviderMultitask(tf_records_file=tf_file_val,
                                 sess=sess, images2epoch=num_images_val * aug_images_val, IMAGE_H=nx,
                                 IMAGE_W=ny, batch_size=batch_size, channels=1, shuffle_flag=True, augment=False)

    net = unet.Unet(batch_size=batch_size,
                    IMAGE_H=nx,
                    IMAGE_W=ny,
                    channels=1,
                    n_class=num_classes,
                    cost="cross_entropy",
                    cost_kwargs=dict(class_weights=None),
                    root_filters=32,branches=branches)

    trainer_instance = seg_trainer.Trainer(net, optimizer="adam",
                                           opt_kwargs=dict(learning_rate=0.001))


    path = trainer_instance.train_val(train_provider=train_generator,
                                      val_provider=val_generator,
                                      output_path=output_path,
                                      sess=sess,
                                      decay_step=10,  ## decay every 10 epochs
                                      epochs=100, # number of epochs to train
                                      display_step=20, # display network info every n iterations
                                      evaluate_model=1, # evaluate model every n epochs
                                      restore=False,
                                      target_names=target_names,
                                      labels=labels,
                                      n_plots=n_plots, branches= branches)

    sess.close()
