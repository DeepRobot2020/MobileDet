#! /usr/bin/env python
"""Overfit a YOLO_v2 model to a single image from the Pascal VOC dataset.

This is a sample training script used to test the implementation of the
YOLO localization loss function.
"""
import argparse
import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model

from mobiledet.models.keras_yolo import (preprocess_true_boxes, yolo_body_mobilenet,
                                     yolo_eval, yolo_head, yolo_loss, yolo_body_darknet)
from mobiledet.models.keras_darknet19 import darknet_feature_extractor
from mobiledet.utils.draw_boxes import draw_boxes

from cfg import *

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

argparser = argparse.ArgumentParser(
    description='Train YOLO_v2 model to overfit on a single image.')

argparser.add_argument(
    '-d',
    '--data_path',
    help='path to HDF5 file containing pascal voc dataset',
    default='~/data/PascalVOC/VOCdevkit/pascal_voc_07_12_person_vehicle.hdf5')

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/aeryon_anchors.txt')

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default='model_data/aeryon_classes.txt')

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readlines()
            try:
                anchors = [anchor.rstrip().split(',') for anchor in anchors]
                anchors =  sum(anchors, [])
                anchors = [float(x) for x in anchors]
            except:
                anchors = YOLO_ANCHORS
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def _main(args):
    voc_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
    else:
        anchors = YOLO_ANCHORS

    anchors = get_anchors(anchors_path)
    print('Prior anchor boxes:')    
    print(anchors)
    num_anchors = len(anchors)
    voc = h5py.File(voc_path, 'r')
    
    test_id = 31
    image = PIL.Image.open(io.BytesIO(voc['train/images'][test_id]))
    orig_size = np.array([image.width, image.height])
    orig_size = np.expand_dims(orig_size, axis=0)

    net_width    = IMAGE_W
    net_height   = IMAGE_H
    feats_width  = FEAT_W
    feats_height = FEAT_H

    # Image preprocessing.
    image = image.resize((net_width, net_height), PIL.Image.BICUBIC)
    image_data = np.array(image, dtype=np.float)
    image_data /= 255.

    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    boxes = voc['train/boxes'][test_id]
    boxes = boxes.reshape((-1, 5))
    # Get extents as y_min, x_min, y_max, x_max, class for comparision with
    # model output.
    boxes_extents = boxes[:, [2, 1, 4, 3, 0]]

    # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
    boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
    boxes_xy = boxes_xy / orig_size
    boxes_wh = boxes_wh / orig_size
    boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)

    # Precompute detectors_mask and matching_true_boxes for training.
    # Detectors mask is 1 for each spatial position in the final conv layer and
    # anchor that should be active for the given boxes and 0 otherwise.
    # Matching true boxes gives the regression targets for the ground truth box
    # that caused a detector to be active or 0 otherwise.
    detectors_mask_shape = (feats_height, feats_width, num_anchors, 1)
    matching_boxes_shape = (feats_height, feats_width, num_anchors, 5)
    detectors_mask, matching_true_boxes = preprocess_true_boxes(boxes, anchors,
                                                                [net_height, net_width], 
                                                                [feats_height, feats_width])

    # Create model input layers.
    image_input = Input(shape=(net_height , net_width, 3))
    boxes_input = Input(shape=(None, 5))

    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    print('Boxes:')
    print(boxes)
    print('Box corners:')
    print(boxes_extents)
    print('Active detectors:')
    print(np.where(detectors_mask == 1)[:-1])
    print('Matching boxes for active detectors:')
    print(matching_true_boxes[np.where(detectors_mask == 1)[:-1]])

    feature_detector = darknet_feature_extractor(image_input, SHALLOW_DETECTOR)
    model_body = yolo_body_darknet(feature_detector, len(anchors), len(class_names), network_config=[SHALLOW_DETECTOR, USE_X0_FEATURE])
    model_body.summary()

    # TODO: Replace Lambda with custom Keras layer for loss.
    model_loss = Lambda(
        yolo_loss,
        output_shape=(1, ),
        name='yolo_loss',
        arguments={'anchors': anchors,
                    'num_classes': len(class_names)})([
                        model_body.output, boxes_input,
                        detectors_mask_input, matching_boxes_input
                    ])

    model = Model(
        [image_input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    # Add batch dimension for training.
    image_data = np.expand_dims(image_data, axis=0)
    boxes = np.expand_dims(boxes, axis=0)
    detectors_mask = np.expand_dims(detectors_mask, axis=0)
    matching_true_boxes = np.expand_dims(matching_true_boxes, axis=0)

    num_steps = 1000

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              batch_size=1,
              verbose=1,
              epochs=num_steps)
              
    model.save_weights('model_data/trained_stage_1.h5')

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.9)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            model_body.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
    print('Found {} boxes for image.'.format(len(out_boxes)))
    print(out_boxes)

    # Plot image with predicted boxes.
    image_with_boxes = draw_boxes(image_data[0], out_boxes, out_classes,
                                  class_names, out_scores)
    plt.imshow(image_with_boxes, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
