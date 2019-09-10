import sys
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from skimage.transform import resize
from utils.data_augmentation import augment_slice_multitask
import numpy as np
import math
from dipy.align.reslice import reslice
import nibabel as nib


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def normalize_volume(vol_data):
    """
    Normalize volume, if filter is False then take into acount all volume voxels
    otherwise take only voxels with labels
    :param vol_data:
    :param seeg_data:
    :param filter:
    :param axis:
    :return:
    """
    h, w, d = np.shape(vol_data)
    # print("Min ", np.min(vol_data))
    # print("Max ", np.max(vol_data))
    mean = np.sum(vol_data)/(h*w*d)
    std = np.std(vol_data)
    return (vol_data - mean) / std


def compute_median_frequency(file_paths, seg_paths,  num_classes=21, filter=True, axis=0):
    histo = np.zeros(num_classes)
    # n_voxels=0
    for i in range(len(file_paths)):
        if not i % 2:
            print('Computing weigthts in vols: {}/{}'.format(i, len(file_paths)))
            sys.stdout.flush()

        ## read data
        data = nib.load(file_paths[i])
        vol_data = data.get_data()
        seg_data = nib.load(seg_paths[i])
        seg_data = seg_data.get_data()

        ## resample data to isotropic resolution
        affine = data.affine
        zooms = data.header.get_zooms()[:3]
        new_zooms = (zooms[0], zooms[0], zooms[0])
        vol_data, affine = reslice(vol_data, affine, zooms, new_zooms)
        seg_data, affine = reslice(seg_data, affine, zooms, new_zooms, order=0)

        if filter:
            data, seg_data = filter_volume(vol_data, seg_data, axis=axis)

        unique, count = np.unique(seg_data, return_counts=True)
        for n in range(num_classes):
            if n in unique:
                index = np.where(unique == n)
                histo[n] += count[index]

        # print('')

    # compute median frequency balance
    freq = histo / np.sum(histo)
    med_freq = np.median(freq)
    weights = med_freq / freq
    return np.asarray(weights)


def filter_volume(vol_data, seg_data, axis=0):
    """
    Return only voxels containing labels according to axis
    :param vol_data: CT or MRI data
    :param seg_data: merged segmentation file
    :param axis:
    :return:
    """
    h, w, d = np.where(seg_data != 0)

    if axis == 0:  # sagital
        start = min(h)
        end = max(h) + 1
        vol_filter = vol_data[start:end, :, :]
        seg_filter = seg_data[start:end, :, :]

    elif axis == 1:  # coronal
        start = min(w)
        end = max(w) + 1
        vol_filter = vol_data[:, start:end, :]
        seg_filter = seg_data[:, start:end, :]

    else:  # 2 axial
        start = min(d)
        end = max(d) + 1
        vol_filter = vol_data[:, :, start:end]
        seg_filter = seg_data[:, :, start:end]

    return vol_filter, seg_filter



"""
Multitask data to tfrecords
"""


def write_multitask_to_tfrecords(file_paths, seg_paths,
                           out_file_name,
                           IMAGE_dim1,
                           IMAGE_dim2,
                           axis=2,
                           weights=None,
                           filter=True,
                                num_classes=21):
    """
    Write  all matlab files from Maries software to tfrecords file
    :param dir_name: directory where the files are stored
    :param out_file_name: path to the tfrecord file
    :param IMAGE_HEIGHT: dimension of the image H
    :param IMAGE_WIDTH: dimension of the image W
    :param axis:  the axist to get slices 0:sagital 1:coronal 2:axial
    :return:
    """

    import scipy.ndimage as ndimage

    def seg_contour_dist_trans(img):

        # one-hot encoding
        img_one_hot = np.eye(21)[np.uint8(img)] > 0.0

        contour = np.uint8(np.zeros(img_one_hot.shape))
        edt = np.zeros(img_one_hot.shape)

        for i in range(1, 21):
            if np.sum(img_one_hot[:, :, i]) != 0:
                # fill holes
                img = ndimage.morphology.binary_fill_holes(img_one_hot[:, :, i])

                # extract contour
                contour[:, :, i] = ndimage.morphology.binary_dilation(img == 0.0) & img

                # distance transform
                tmp = ndimage.morphology.distance_transform_edt(img)
                edt[:, :, i] = (tmp - np.amin(tmp)) / (np.amax(tmp) - np.amin(tmp))

        return np.sum(contour, axis=-1) > 0.0, np.sum(edt, axis=-1)

    # file_paths, seg_paths = read_paths_from_excel(excel_file)
    total_images=0

    if weights is None:
        print("Computing the class weights using median frequency ...")
        weights = compute_median_frequency(file_paths,seg_paths, num_classes=num_classes, filter=True, axis=axis)

        print(" Median Frequency weights = ", weights)

    filename = out_file_name + '.tfrecords'  # address to save the TFRecords file
    print('Writing {} files in {}'.format(len(file_paths), filename))

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(file_paths)):
        # print how many volumenes have being written
        if not i % 2:
            print('Data: {}/{}'.format(i, len(file_paths)))
            sys.stdout.flush()

        ## read data
        data = nib.load(file_paths[i])
        vol_data = data.get_data()
        seg_data = nib.load(seg_paths[i])
        seg_data = seg_data.get_data()

        # ## resample data to isotropic resolution
        # affine = data.affine
        # zooms = data.header.get_zooms()[:3]
        # new_zooms = (zooms[0], zooms[0], zooms[0])
        # vol_data, affine = reslice(vol_data, affine, zooms, new_zooms)
        # seg_data, affine = reslice(seg_data, affine, zooms, new_zooms, order=0)

        vol_data = normalize_volume(vol_data)

        if filter:
            vol_data, seg_data = filter_volume(vol_data, seg_data, axis=axis)

        n_slices = np.shape(vol_data)[axis]

        for j in range(n_slices):
            if axis == 0:  # sagital
                vol_slice = vol_data[j, :, :]
                seg_slice = seg_data[j, :, :]

            elif axis == 1:  # coronal
                vol_slice = vol_data[:, j, :]
                seg_slice = seg_data[:, j, :]

            elif axis == 2:  # axial
                vol_slice = vol_data[:, :, j]
                seg_slice = seg_data[:, :, j]
            else:
                raise ValueError("Unknown axis: it must be between 0 and 3 got--> " % axis)

            edges, dist = seg_contour_dist_trans(seg_slice)
            edges = edges.astype(np.float32)
            dist = dist.astype(np.float32)

            ### resize image if is greater than 800 in the y direction
            h, w = np.shape(seg_slice)

            if h !=IMAGE_dim1 or w != IMAGE_dim2:
                vol_slice = resize(vol_slice, output_shape=(IMAGE_dim1, IMAGE_dim2), order=1, mode='constant', anti_aliasing=False,preserve_range=True)
                seg_slice = resize(seg_slice, output_shape=(IMAGE_dim1, IMAGE_dim2), order=0, mode='constant', anti_aliasing=False, preserve_range=True)
                edges = resize(edges, output_shape=(IMAGE_dim1, IMAGE_dim2), order=0, mode='constant', anti_aliasing=False, preserve_range=True)
                dist = resize(dist, output_shape=(IMAGE_dim1, IMAGE_dim2), order=1, mode='constant', anti_aliasing=False, preserve_range=True)


            # convert images to float32
            if np.any(seg_slice) !=0:
                vol_slice = np.expand_dims(vol_slice, axis=-1)
                seg_slice = np.expand_dims(seg_slice, axis=-1)
                edges = np.expand_dims(edges, axis=-1)
                dist = np.expand_dims(dist, axis=-1)

                vol_slice = vol_slice.astype(np.float32)
                seg_slice = seg_slice.astype(np.float32)
                edges = edges.astype(np.float32)
                dist = dist.astype(np.float32)

                vol_slice = vol_slice.tostring()
                seg_slice = seg_slice.tostring()
                edges = edges.tostring()
                dist = dist.tostring()

                # Create a feature
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature={
                    'data/slice': _bytes_feature(vol_slice),
                    'data/seg': _bytes_feature(seg_slice),
                    'data/edges': _bytes_feature(edges),
                    'data/dist': _bytes_feature(dist)}))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
                total_images +=1
    print("The total number of samples in dataset is:{}".format(total_images))
    writer.close()
    sys.stdout.flush()
    return weights


def read_and_decode_multitask(records_file, IMAGE_HEIGHT,IMAGE_WIDTH, batch_size=10,channels=1, num_classes=21, shuffle_flag=True, augment=False):

    filename_queue = tf.train.string_input_producer([records_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data/slice': tf.FixedLenFeature([], tf.string),
            'data/seg': tf.FixedLenFeature([], tf.string),
            'data/edges': tf.FixedLenFeature([], tf.string),
            'data/dist': tf.FixedLenFeature([], tf.string)})

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['data/slice'], tf.float32)
    seg = tf.decode_raw(features['data/seg'], tf.float32)
    edge = tf.decode_raw(features['data/edges'], tf.float32)
    dist = tf.decode_raw(features['data/dist'], tf.float32)


    # Reshape image data into the original shape

    image = tf.reshape(image,shape=[IMAGE_HEIGHT,IMAGE_WIDTH,channels])
    seg = tf.reshape(seg, shape=[IMAGE_HEIGHT,IMAGE_WIDTH,channels])
    edge = tf.reshape(edge, shape=[IMAGE_HEIGHT, IMAGE_WIDTH, channels])
    dist = tf.reshape(dist, shape=[IMAGE_HEIGHT, IMAGE_WIDTH, channels])


    if augment:
        ## TODO: new dimentions are hard coded
        image,seg, edge, dist = augment_slice_multitask(image,seg,edge, dist, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, channels=channels)

    # image = tf.image.per_image_standardization(image)
    ## ground truth to one hot
    seg = tf.cast(seg, tf.int32)
    seg= tf.squeeze(seg, squeeze_dims=[2])
    seg = tf.one_hot(seg, depth=num_classes)

    edge = tf.cast(edge, tf.int32)
    edge = tf.squeeze(edge, squeeze_dims=[2])
    edge = tf.one_hot(edge, depth=2)

    if shuffle_flag:
        min_after_dequeue_train = 10 * batch_size
        num_threads_train = 6
        capacity_train = 20 * batch_size
        images, segs, edges, dists = tf.train.shuffle_batch([image, seg, edge, dist],
                                                        batch_size=batch_size,
                                                        capacity=capacity_train,
                                                        min_after_dequeue=min_after_dequeue_train,
                                                        num_threads=num_threads_train)
        return images, segs, edges, dists

    # collect batches of images
    else:
        images, segs, edges, dists= tf.train.batch(
            [image, seg, edge, dist],
            batch_size=batch_size, num_threads=1)
        return images, segs, edges, dists


class DataProviderMultitask():
    def __init__(self, tf_records_file, sess, images2epoch, IMAGE_H=512,IMAGE_W=512,batch_size=10,channels=1, num_classes=21, shuffle_flag=False, augment=False):
        self.epoch_num=0
        self.tf_records_file = tf_records_file
        self.sess= sess
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.batch_size = batch_size
        self.channels = channels
        self.shuffle_flag = shuffle_flag
        self.augment = augment
        self.images2epoch= images2epoch
        self.iter=0
        self.iter2epoch= int(math.ceil(self.images2epoch / self.batch_size))
        self.num_classes= num_classes

        self.images, self.segs, self.edges, self.dists = read_and_decode_multitask(records_file=self.tf_records_file,
                                                    IMAGE_HEIGHT=self.IMAGE_H,
                                                    IMAGE_WIDTH=self.IMAGE_W,
                                                    batch_size=self.batch_size, channels=self.channels,
                                                    shuffle_flag=self.shuffle_flag,
                                                    augment=self.augment,
                                                    num_classes=self.num_classes)

    def next_batch(self):
        img, seg, edge, dist = self.sess.run([self.images, self.segs, self.edges, self.dists])
        if self.iter % self.iter2epoch == 0 and self.iter != 0:
            self.epoch_num += 1

        self.iter += 1
        return img, seg, edge, dist

    def getEpoch(self):
        return self.epoch_num

    def getIteration(self):
        return self.iter
