import sys
sys.path.append("/home/fercho/projects/shape_aware_segmentation")
from data_provider import write_multitask_to_tfrecords
import os
import xlrd

"""
Script to convert excel file in Visceral multitask  data list to tfrecords
"""


"""
Writing training data
"""

excel_file='data/sc_volumes.xlsx or data/gc_val.xlsx '
tf_file='path to the generated tfrecord file example: /home/Desktop/train or /home/Desktop/val'
data_folder='path to the the folder data'


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

vol_paths, seg_paths = list_files(excel_file,data_folder)


weights_array = [1.00000000e+00, 5.74720964e+01, 4.67654400e+02, 1.58137518e+03,
                 7.14735486e+03, 6.28030394e+02, 4.66110436e+02, 3.68521725e+03,
                 4.96670375e+01, 5.63883689e+01, 1.83480061e+03, 5.38223611e+03,
                 2.16616129e+03, 6.64989274e+02, 6.01810566e+02, 3.01955143e+04,
                 2.15210323e+04, 5.88140475e+02, 6.42542047e+02, 6.44860816e+02,
                 6.14258756e+02]

weigthts = write_multitask_to_tfrecords(vol_paths,seg_paths,
                            out_file_name=tf_file,
                           IMAGE_dim1=512,
                           IMAGE_dim2=512,
                            axis=2,
                            weights=weights_array,
                            filter=True)

print('Data in {} written to file {}.tfrecords'.format(excel_file,tf_file))


# The total number of samples in dataset is:838
