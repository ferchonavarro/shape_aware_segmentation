import sys
import tensorflow as tf
sys.path.append("/home/fercho/projects/shape_aware_segmentation")
#sys.path.append("add path to the project main folder")
import numpy as np
import unet
import os
import nibabel as nib
from skimage.transform import resize
import xlrd
from pandas import DataFrame
from collections import OrderedDict
from utils.visualization import target_names, ORGAN_MAP
"""
Run the script data2tf_coronal.py before this script
Train a U-Net using coronal view
Change the output path to your desired folder, it will overwrite the data inside the folder
"""
nx = 512 ## input size width
ny = 512 ## input size high
axis= 2 # choose the axis 0: sagital 1:coronal  2: axial

excel_file ='data/gc_volumes.xlsx'
data_folder='path to the GC volumnes from visceral data'
out_dir='path to folder where resulting segmentation will be save'
batch_size= 1
num_classes=21
model_path ='path to your trained model checkpoint'

print('Model', model_path)
print('Save path', out_dir)


def list_files(excel_file, data_folder):
    """
    Open excel file and select all volumes with the organ number
    :return: list of files
    """

    wb = xlrd.open_workbook(excel_file)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # Extracting number of rows
    nsampes = sheet.nrows
    vol_paths, seg_paths =[],[]

    for i in range(1, nsampes):
        row = sheet.row_values(i)
        row = row[0]

        filename=  row.split('.')[0]
        folder = row.split('C')[0]
        folder = folder[:-1]
        vol_paths.append(os.path.join(data_folder,folder,filename +'.nii.gz'))
        segname= filename+'_seg.nii.gz'
        seg_paths.append(os.path.join(data_folder, folder, segname))

    return vol_paths, seg_paths


def dice_score_per_class(y_true,y_pred, n_class):
    # n_class = np.unique(y_true)
    dices =[]
    for k in range(n_class):
        dice = np.sum(y_pred[y_true == k] == k) * 2.0 / (np.sum(y_pred[y_pred == k] == k) + np.sum(y_true[y_true == k] == k) + np.finfo(float).eps)
        dices.append(dice)
    return dices


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
    mean = np.sum(vol_data)/(h*w*d)
    std = np.std(vol_data)
    return (vol_data - mean) / std


with tf.Graph().as_default():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net= unet.Unet(batch_size=batch_size,
                   IMAGE_H=nx,
                   IMAGE_W=ny,
                   channels=1,
                   n_class=num_classes,
                   cost="cross_entropy",
                   cost_kwargs=dict(class_weights=None),
                   root_filters=32)

    init = tf.global_variables_initializer()
    sess.run(init)

    from tensorflow.contrib.framework.python.framework import checkpoint_utils
    # Restore model weights from previously saved model
    net.restore(sess, model_path)

    file_paths, seg_paths = list_files(excel_file, data_folder)
    dscores =[]
    all_dice = []
    bases =[]

    for i in range(len(file_paths)):
        base = os.path.basename(file_paths[i]).split('.')[0]
        base = base.split('C')[0][:-1]
        bases.append(base)

        print("Patient: {}".format(file_paths[i]))

        dst_folder = os.path.join(out_dir, base)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        ## read data
        vol_data = nib.load(file_paths[i])
        affine = vol_data.affine
        vol_data = vol_data.get_data()
        seg_data = nib.load(seg_paths[i])
        seg_data = seg_data.get_data()

        vol_data = normalize_volume(vol_data)

        n_slices = np.shape(vol_data)[axis]
        h, w, d = np.shape(vol_data)
        seg_result = np.zeros(shape=(h, w, d))

        for j in range(n_slices):
            if axis == 0:  # sagital
                vol_slice = vol_data[j, :, :]
                seg_slice = seg_data[j, :, :]
                hs, ws = np.shape(vol_slice)

            elif axis == 1:  # coronal
                vol_slice = vol_data[:, j, :]
                seg_slice = seg_data[:, j, :]
                hs, ws = np.shape(vol_slice)

            elif axis == 2:  # axial
                vol_slice = vol_data[:, :, j]
                seg_slice = seg_data[:, :, j]
                hs, ws = np.shape(vol_slice)
            else:
                raise ValueError("Unknown axis: it must be between 0 and 3 got--> " % axis)

            ### resize image
            if hs != nx or ws != ny:
                vol_slice = resize(vol_slice, output_shape=(nx, ny), order=1, mode='constant',
                                    preserve_range=True, anti_aliasing=False)
                seg_slice = resize(seg_slice, output_shape=(nx, ny), order=0, mode='constant',
                                    preserve_range=True, anti_aliasing=False)

            x_test = np.expand_dims(np.expand_dims(vol_slice, axis=0), axis=3)
            # y_test = np.expand_dims(seg_slice, axis=0)
            prediction = net.predict(sess, x_test)

            if hs != nx or ws != ny:
                prediction = resize(prediction, output_shape=(hs, ws), order=0, mode='constant',
                                    preserve_range=True, anti_aliasing=False)


            seg = np.argmax(np.squeeze(prediction), axis=-1)

            if axis == 0:  # sagital
                seg_result[j, :, :] = seg
            elif axis == 1:  # coronal
                seg_result[:, j, :] = seg
            elif axis == 2:  #  axial
                seg_result[:, :, j] = seg
            else:
                raise ValueError("Unknown axis: it must be between 0 and 3 got--> " % axis)

        print('Dice Network Predictions')
        dice = dice_score_per_class(y_pred=seg_result, y_true=seg_data, n_class=21)
        print("Average Dice Score: {}".format(np.mean(dice)))
        print("Average Dice no background: {}".format(np.mean(dice[1:])))
        all_dice.append(dice)

        _name = '{}_seg.nii.gz'.format(base)
        file_name = os.path.join(out_dir,base, _name)

        vol = nib.Nifti1Image(seg_result, affine)
        nib.save(vol, file_name)
        print('')

    # print(all_dice)


results_file = os.path.join(out_dir, 'dice_scores.xlsx')
## compute the overall dice per class and together
all_dice = np.stack([np.array(i) for i in all_dice])
dfdict = OrderedDict()
dfdict["Volume"] = bases
headers=[]
headers.append('Volume')
for i in range(len(target_names)):
    headers.append(target_names[i])
    dfdict[target_names[i]] = list(all_dice[:,i])

df = DataFrame(dfdict)
df.to_excel(results_file, index=False,columns=headers)

