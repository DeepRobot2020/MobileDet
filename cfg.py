'''
Main configuration file for YOLOv2 Project.
Modify every time one would like to train on a new dataset
'''
import numpy as np

# If a feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
# Most current state-of-the-art models have max-pooling layers (August, 2017)
# FEATURE_EXTRACTOR = 'darknet19'
FEATURE_EXTRACTOR = 'mobilenet'
N_CLASSES         = 2
N_ANCHORS         = 5
BATCH_SIZE        = 4
IMAGE_H           = 608
IMAGE_W           = 608

SHALLOW_DETECTOR = True
USE_X0_FEATURE = True

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

if SHALLOW_DETECTOR:
    SHRINK_FACTOR     = 16
else:
    SHRINK_FACTOR     = 32

FEAT_H = IMAGE_H // SHRINK_FACTOR
FEAT_W = IMAGE_W // SHRINK_FACTOR



