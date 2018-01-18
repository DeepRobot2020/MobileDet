# MobileDet
MobileNet and YOLOv2 based real-time Person and Vehicle detector.

## UAV123 dataset 
### Test with UAV123 video sequences 
 * 0. Download the UAV123 dataset zip file from [link](https://drive.google.com/file/d/0B6sQMCU1i4NbZmFlQmJBVDlLRDg/edit). It is about 4.4G. Unzip it into '~/data/UAV123/'. 
     Download the pre-processed UAV123 dataset from [link]

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
* 4. Run below sampe script 

## VOC dataset 
### Test with Pascal VOC images 
 * 1. download weights from this google drive [link to voc weights!](https://drive.google.com/open?id=1gv9Gx6v1rfxFW25Vu0C70UFaq_8t8Mhs)