# MobileDet
MobileNet and YOLOv2 based real-time Person and Vehicle detector.
The final verion of model is called: YOLOv2-MobileNet_Shallow_3Scales
This model can run up to 10fps on NVIDIA TX2 platform

## UAV123 dataset 
### Test with UAV123 video sequences 
 * 0. Download the UAV123 dataset zip file from [link](https://drive.google.com/file/d/0B6sQMCU1i4NbZmFlQmJBVDlLRDg/edit). It is about 4.4G. Unzip it into '~/data/UAV123/'. 
     Download the pre-processed UAV123 dataset from [link](https://drive.google.com/open?id=1_y_T5dEq-jclesTNxQR-MK3cWR8YmQAR) and placed it under '~/data' folder. 
 * 1. Download weights from this google drive [link to UAV123 weights](https://drive.google.com/open?id=1dUjdlRjuWyvyMQKuaF6X9RExKBIRedJt) and place the unzipped folder to 'MobileDet/weights_uav123'
 * 2. Modify the cfg.py as below to configure the model as YOLOv2-MobileNet_Shallow_3Scales
```python
FEATURE_EXTRACTOR = 'mobilenet'
SHALLOW_DETECTOR = True
USE_X0_FEATURE = True
```
 * 3. Run  below sample script to use the model to detect objects from UAV123 images

```python
python test_yolo.py  -m weights_uav123/mobilenet_s3_best.TrueTrue.h5  -t ~/data/UAV123/UAV123_10fps/data_seq/UAV123_10fps/bike3 -o  ~/Videos/bike3 -iou 0.6 -s 0.6
```
* 4. Run below sample script to use calculate the recall and precision of Test group  UAV123 dataset
```python
python recall_precision.py  -m weights_uav123/mobilenet_s3_best.TrueTrue.h5 -d ~/data/uav123.hdf5 -a model_data/uav123_anchors.txt -iou 0.6 -s 0.6
```

### Retrain with other dataset
## Pre-process dataset 
 * 1. To retrain with other dataset, it is required to write a parse script to convert the dataset into HDF5 format. This project has python script for dataset: UAV123, VOC and Okutama Action datasets
 * 2. After creating the HDF5 dataset, it is required to use the 'anchor_boxes.py' to run k-means method to generate a few prior anchor boxes for this dataset 
 * 3. After above two steps, use  'retrain_yolo.py' to retrain the new dataset



### YAD2K
This project is based on: https://github.com/allanzelener/YAD2K


