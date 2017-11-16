#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import os.path as osp
import random

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

from mobiledet.utils.draw_boxes import draw_boxes

from mobiledet.utils import read_voc_datasets_train_batch, brightness_augment, augment_image
from mobiledet.models.keras_yolo import yolo_get_detector_mask

from cfg import *
from keras.utils import plot_model


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import os
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    '-m',
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/aeryon_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/aeryon_classes.txt')
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
    help='threshold for non max suppression IOU, default .3',
    default=.6)

parser.add_argument(
    '-tf',
    '--convert_to_tensorflow',
    type=float,
    help='convert the model to tensorflow and save as protocol buffer',
    default=True)


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

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def convert_to_tensorflow(net_model):
    num_output = 1
    write_graph_def_ascii_flag = True
    prefix_output_node_names_of_final_network = 'output_node'
    output_graph_name = 'constant_graph_weights.pb'
    output_fld = 'tensorflow_model/'
    if not os.path.isdir(output_fld):
        os.mkdir(output_fld)
    K.set_learning_phase(0)

    pred = [None]*num_output
    pred_node_names = [None]*num_output
    print('convert_to_tensorflow=====>: ')
    for i in range(num_output):
        pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])


    print('output nodes names are: ', pred_node_names)
    sess = K.get_session()
    if write_graph_def_ascii_flag:
        f = 'only_the_graph_def.pb.ascii'
        tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
        print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))


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
    model_path = os.path.expanduser(args.model_path)
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

    yolo_model, model = create_model(anchors, class_names, load_pretrained=True, 
        feature_extractor=FEATURE_EXTRACTOR, pretrained_path=model_path)

    plot_model(model, to_file='model.png')
    # convert_to_tensorflow(model)
    model.save('bar.hdf5')

    model_file_basename, file_extension = os.path.splitext(os.path.basename(model_path))

    model_input = model.input[0].name.replace(':0', '')
    model_output = model.output.name.replace(':0', '')

    sess = K.get_session()
    width, height, channels = int(model.input[0].shape[2]), int(model.input[0].shape[1]), int(model.input[0].shape[3])
    # END OF keras specific code
    freeze(sess, model_file_basename, model_input, width, height, channels, model_output)

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
    is_fixed_size = model_image_size != (None, None)

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
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = decode_yolo_output(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    image_files = sorted(os.listdir(test_path))
    for idx in range(len(image_files)):
        image_file = image_files[idx]
        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        image = Image.open(os.path.join(test_path, image_file))
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))

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
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
