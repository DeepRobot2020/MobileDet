#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import glob
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

from mobiledet.models.keras_yolo import yolo_eval, yolo_loss, decode_yolo_output, create_model
from mobiledet.models.keras_yolo import yolo_body_darknet, yolo_body_mobilenet                     
from mobiledet.models.keras_yolo import recall_precision


from mobiledet.utils.draw_boxes import draw_boxes

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
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    '-m',
    '--weight_path',
    help='path to trained model weight file')

parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to lisa_anchors.txt',
    default='model_data/lisa_anchors.txt')

parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to drone_classes.txt',
    default='model_data/lisa_classes.txt')

parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')

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
    default=.6)

parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .6',
    default=.6)

parser.add_argument(
    '-rp',
    '--calculate_precision_recall',
    type=float,
    help='Calculate precision and recall',
    default=True)

parser.add_argument(
    '-tf',
    '--convert_to_tensorflow',
    type=float,
    help='convert the model to tensorflow and save as protocol buffer',
    default=True)


def freeze(tf_session, model_name, model_input_name, width, height, channels, model_output_name):
    input_binary = True
    graph_def = tf_session.graph.as_graph_def()

    tf.train.Saver().save(tf_session, model_name + '.ckpt')
    tf.train.write_graph(tf_session.graph.as_graph_def(), logdir='.', name=model_name + '.binary.pb', as_text=not input_binary)

    # We save out the graph to disk, and then call the const conversion routine.
    checkpoint_state_name = model_name + ".ckpt.index"
    input_graph_name = model_name + ".binary.pb"
    output_graph_name = model_name + ".pb"

    input_graph_path = os.path.join(".", input_graph_name)
    input_saver_def_path = ""
    input_checkpoint_path = os.path.join(".", model_name + '.ckpt')

    output_node_names = model_output_name
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    output_graph_path = os.path.join('.', output_graph_name)
    clear_devices = False

    freeze_graph(input_graph_path, input_saver_def_path,
                 input_binary, input_checkpoint_path,
                 output_node_names, restore_op_name,
                 filename_tensor_name, output_graph_path,
                 clear_devices, "")

    print("Model loaded from: %s" % model_name)
    print("Output written to: %s" % output_graph_path)
    print("Model input name : %s" % model_input_name)
    print("Model input size : %dx%dx%d (WxHxC)" % (width, height, channels))
    print("Model output name: %s" % model_output_name)


def _main(args):
    model_path = os.path.expanduser(args.weight_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    class_names  = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    if SHALLOW_DETECTOR:
        anchors = anchors * 2

    print(class_names)
    print(anchors)

    yolo_model, yolo_model_for_training = create_model(anchors, class_names, load_pretrained=True, 
        feature_extractor=FEATURE_EXTRACTOR, pretrained_path=model_path, freeze_body=True)

    model_file_basename, file_extension = os.path.splitext(os.path.basename(model_path))

    model_input = yolo_model.input.name.replace(':0', '') # input_1
    model_output = yolo_model.output.name.replace(':0', '') # conv2d_3/BiasAdd

    sess = K.get_session()
    width, height, channels = int(yolo_model.input.shape[2]), int(yolo_model.input.shape[1]), int(yolo_model.input.shape[3])

    # END OF keras specific code
    # freeze(sess, model_file_basename, model_input, width, height, channels, model_output)

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

    idx = 0
    image_files = sorted(glob.glob(test_path + '/*.png')) 

    # import pdb; pdb.set_trace()
    for idx in range(len(image_files)):
        
        image_file = image_files[idx]
        # print(os.path.join(test_path, image_file))
        image = Image.open(image_file)

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
