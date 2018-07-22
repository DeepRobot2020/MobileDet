'''
Main configuration file for YOLOv2 Project.
Modify every time one would like to train on a new dataset
'''
import numpy as np

# If a feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
# Most current state-of-the-art models have max-pooling layers (Jan, 2018)
# FEATURE_EXTRACTOR = 'darknet19'
FEATURE_EXTRACTOR = 'mobilenet'
# Number of classes, always 2 (Person and Vehicle)

N_CLASSES = 4
# Number of anchor boxes
N_ANCHORS         = 5
# Training batch size, adjust this value based on hardware
BATCH_SIZE        = 4
# Model input size 
IMAGE_H           = 608 # image height
IMAGE_W           = 608 # image width 

# SHALLOW_DETECTOR is to control how many ConvNet blocks we have in the network 
SHALLOW_DETECTOR = False 
# USE_X0_FEATURE is to control how many scales of feature maps used for detection  
USE_X0_FEATURE = False

# Default anchor boxes for COCO dataset
# YOLO_ANCHORS = np.array(
#     ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
#      (7.88282, 3.52778), (9.77052, 9.16828)))

# Each max-pooling layer will shrink the input by half
#   The origional YOLOv2 contains 5 ConvNet blocks and each includes one max-pooling,
#   therefore the feature map size will be 1/32
#   IF the SHALLOW_DETECTOR is true, the customized YOLOv2 contains 4 ConvNet blocks, 
#   therefore the feature map size will be 1/16  
if SHALLOW_DETECTOR:
    SHRINK_FACTOR     = 16
else:
    SHRINK_FACTOR     = 32

FEAT_H = IMAGE_H // SHRINK_FACTOR
FEAT_W = IMAGE_W // SHRINK_FACTOR



