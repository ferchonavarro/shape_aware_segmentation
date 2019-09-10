# Shape-Aware Complementary-Task Learning for Multi-Organ Segmentation

Source code for our paper "Shape-Aware Complementary-Task Learning for Multi-Organ Segmentation" as described in [Early access](https://arxiv.org/abs/1908.05099) on **10th International Workshop on Machine Learning in Medical Imaging (MLMI 2019), MICCAI 2019**).



## Getting Started
### Pre-requisites

You need to have following in order for this library to work as expected

1. python >= 3.6.5
2. pip >= 18.1
3. tensorflow-gpu = 1.9.0
4. tensofboard = 1.9.0
4. numpy >= 1.15.0
5. dipy >= 0.14.0
6. matplotlib>= 2.2.2
7. nibabel >= 1.15.0
8. pandas >= 0.23.4
9. scikit-image >= 0.14.0
10. scikit-learn >= 0.20.0
11. scipy >= 1.1.0
12. seaborn >= 0.9.0
13. SimpleITK >= 1.1.0
14. tabulate >= 0.8.2
15. xlrd >= 1.1.0

### Install requirements
Run `pip install -r requirements.txt`


## How to use the code for training
### Convert your data-set to tfrecords
Request access to Visceral Anatomy 3 data [Anatomy3](hhttp://www.visceral.eu/closed-benchmarks/anatomy3/).

Run the python script `data2tfrecords.py`, make sure you change the paths to `data_folder` and `tf_file`. The data should be in the following format:

```
└── data folder
    ├── 10000004_1
      ├── 10000004_1_CT_wb.nii.gz
      ├── 10000004_1_CT_wb_seg.nii.gz
    ├── 10000007_1
      ├── 10000004_1_CT_wb.nii.gz
      ├── 10000004_1_CT_wb_seg.nii.gz
    |   ....................... 
   
```

You need to generate two files; a training and a validation tfrecord file. Change `excel_file` and other variables accordingly.

### Start the training
Run the python script `train_val.py`. Make sure to change file paths for tfrecord files according to your configuration.

Enable segmentation branch, countour branch and distance branch by changing the variable `branches=['DistanceTransformBranch','EdgesBranch', 'SegBranch']`. `branches=['SegBranch']` means only the segmenation branch is activated.


## How to use the code for inference

Run the python script `inference.py`. Follow the commends in the script to change variables according to your training model and file paths.


## License and Citation

Please cite our paper if it is useful for your research:

    
    @article{navarro2019shape,
    	title={Shape-Aware Complementary-Task Learning for Multi-Organ Segmentation},
    	author={Navarro, Fernando and Shit, Suprosanna and Ezhov, Ivan and Paetzold, Johannes and Gafita, Andrei and Peeken, Jan and Combs, Stephanie and Menze, Bjoern},
    	journal={arXiv preprint arXiv:1908.05099},
    	year={2019}
    }
    
## Code Authors

* **Fernando Navarro**  - [ferchonavarro](https://github.com/ferchonavarro)
* **Suprosanna Shit** - [suprosanna](https://github.com/suprosanna) 

## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that start button on top. Enjoy :)