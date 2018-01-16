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
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
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
    model_path = os.path.expanduser(args.model_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    output_path = os.path.expanduser(args.output_path)

    data_path = '~/data/uav123.hdf5'
    data_path = os.path.expanduser(data_path)
    voc = h5py.File(data_path, 'r')

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

    hdf5_images = np.array(voc['test/images'])

    recall_precision(np.array(voc['test/images']), np.array(voc['test/boxes']), 
        yolo_model, anchors, class_names, num_samples=2048)

    return

    model_file_basename, file_extension = os.path.splitext(os.path.basename(model_path))

    model_input = yolo_model.input.name.replace(':0', '') # input_1
    model_output = yolo_model.output.name.replace(':0', '') # conv2d_3/BiasAdd

    sess = K.get_session()
    width, height, channels = int(yolo_model.input.shape[2]), int(yolo_model.input.shape[1]), int(yolo_model.input.shape[3])

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = decode_yolo_output(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    num_samples = 1024
    n_samples = hdf5_images.shape[0]
    sample_list = np.random.choice(n_samples, num_samples, replace=False)

    idx = 0
    for cur_id in sample_list:
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        image = PIL.Image.open(io.BytesIO(hdf5_images[cur_id]))
        
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)

        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        start = time.time()
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        last = (time.time() - start)
        print('Found {} boxes for {}'.format(len(out_boxes), idx))        

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.save(os.path.join(output_path, str(idx)+'.jpg'), quality=90)
        idx += 1
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
