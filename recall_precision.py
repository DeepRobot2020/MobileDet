#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import os.path as osp
import random
import h5py

import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from mobiledet.models.keras_yolo import preprocess_true_boxes
from mobiledet.models.keras_yolo import yolo_eval, yolo_loss, decode_yolo_output, create_model
from mobiledet.models.keras_yolo import yolo_body_darknet, yolo_body_mobilenet                     
from mobiledet.models.keras_yolo import recall_precision


from mobiledet.utils.draw_boxes import draw_boxes

from mobiledet.utils import read_voc_datasets_train_batch, brightness_augment, augment_image
from mobiledet.models.keras_yolo import yolo_get_detector_mask

from cfg import *
from mobiledet.utils import *
from keras.utils import plot_model


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import os
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

import time

parser = argparse.ArgumentParser(
    description='Calculate YOLOv2 recall and precision on test datasets..')
parser.add_argument(
    '-m',
    '--weight_path',
    help='path to trained model weight file')

parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to pascal_anchors.txt',
    default='model_data/uav123_anchors.txt')

parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to drone_classes.txt',
    default='model_data/drone_classes.txt')

parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='images/out')

parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .6',
    default=.3)

parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.3)

def _main(args):
    model_path = os.path.expanduser(args.weight_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    output_path = os.path.expanduser(args.output_path)

    data_path = '~/data/uav123.hdf5'
    data_path = os.path.expanduser(data_path)
    dataset = h5py.File(data_path, 'r')

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    class_names  = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    if SHALLOW_DETECTOR:
        anchors = anchors * 2

    print(class_names)
    print(anchors)

    yolo_model, _ = create_model(anchors, class_names, load_pretrained=True, 
        feature_extractor=FEATURE_EXTRACTOR, pretrained_path=model_path)

    hdf5_images = np.array(dataset['test/images'])

    recall_precision(np.array(dataset['test/images']), np.array(dataset['test/boxes']), 
        yolo_model, anchors, class_names, num_samples=2048)


if __name__ == '__main__':
    _main(parser.parse_args())
