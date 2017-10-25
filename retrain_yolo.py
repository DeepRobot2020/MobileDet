"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import h5py

from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from mobiledet.models.keras_yolo import (preprocess_true_boxes, yolo_body_darknet19, yolo_body_darknet_feature,
                                     yolo_body_mobilenet, yolo_eval, yolo_head, yolo_loss)
from mobiledet.utils.draw_boxes import draw_boxes

from mobiledet.utils import read_voc_datasets_train_batch, brightness_augment, augment_image
from mobiledet.models.keras_yolo import yolo_get_detector_mask
from mobiledet.models.keras_darknet19 import darknet19_feature_extractor

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default='~/data/PascalVOC/VOCdevkit/pascal_voc_07_12.hdf5')

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default='model_data/pascal_classes.txt')

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

BATCH_SIZE = 4
IMAGE_H    = 608
IMAGE_W    = 608

FEAT_W = IMAGE_W / 32
FEAT_H = IMAGE_H / 32

def _main(args):
    data_path    = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    voc_classes_path='model_data/pascal_classes.txt'
    voc_class_names  = get_classes(voc_classes_path)

    class_names  = get_classes(classes_path)
    print(voc_class_names)
    print(class_names)
    
    anchors      = get_anchors(anchors_path)
    # custom data saved as a numpy file.
    h5_data = h5py.File(data_path, 'r')
    anchors = YOLO_ANCHORS
    
    H5_TRAIN_BOXES = np.array(h5_data['train/boxes'])
    H5_TRAIN_IMAGES = np.array(h5_data['train/images'])

    H5_VALID_BOXES = np.array(h5_data['val/boxes'])
    H5_VALID_IMAGES = np.array(h5_data['val/images'])
    # clear any previous sesson
    K.clear_session()

    model_body, model = create_model(anchors, class_names, load_pretrained=True)
    train_batch_gen = VOCBatchGenerator(H5_TRAIN_IMAGES, H5_TRAIN_BOXES, IMAGE_W, IMAGE_H, FEAT_W, FEAT_H, anchors, voc_class_names, class_names, jitter=True)
    valid_batch_gen = VOCBatchGenerator(H5_VALID_IMAGES, H5_VALID_BOXES, IMAGE_W, IMAGE_H, FEAT_W, FEAT_H, anchors, voc_class_names, class_names)
    train(
        model,
        class_names,
        anchors, 
        train_batch_gen,
        valid_batch_gen)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def create_model(anchors, class_names, load_pretrained=False, freeze_body=False):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (FEAT_H, FEAT_W, 5, 1)
    matching_boxes_shape = (FEAT_H, FEAT_W, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(IMAGE_H, IMAGE_W, 3))
    boxes_input = Input(shape=(None, 5))

    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    extra_detect_feats = True
    
    feature_model = darknet19_feature_extractor(image_input);
    yolo_model = yolo_body_darknet_feature(feature_model, len(anchors), len(class_names), extra_detection_feature=extra_detect_feats)
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        print("Loading pre-trained weights")
        yolo_path = os.path.join('model_data', 'yolo.h5')
        model_body = load_model(yolo_path)
        # Only load weights before the conv20 layer( the layer right before x4 and x2 space_to_depth conversion)
        model_body = Model(model_body.inputs, model_body.layers[-8].output)               
        model_body.save_weights(topless_yolo_path)
        feature_model.load_weights(topless_yolo_path)
      
    if freeze_body:
        for layer in feature_model.layers:
            layer.trainable = False
            
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
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
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

class VOCBatchGenerator:
    def __init__(self, H5_IMAGES, 
                       H5_BOXES,
                       model_w, 
                       model_h, 
                       feat_w,
                       feat_h, 
                       anchors,
                       voc_class_names,
                       class_names, 
                       jitter=False):
        self.model_w = model_w
        self.model_h = model_h
        self.feat_w = feat_w
        self.feat_h = feat_h
        self.voc_class_names =  voc_class_names
        self.class_names =  class_names        
        self.H5_BOXES = H5_BOXES
        self.H5_IMAGES = H5_IMAGES
        self.anchors = anchors
        self.unique_data_instances = self.H5_IMAGES.shape[0]
        self.jitter = jitter

    def flow_from_hdf5(self):
        anchors = self.anchors   
        while True:
            batch_images = []
            batch_boxes = []
            
            for i in range(BATCH_SIZE):        
                image_data, bboxes = read_voc_datasets_train_batch(self.H5_IMAGES, self.H5_BOXES, self.voc_class_names, self.class_names)
                image_data, bboxes = augment_image(image_data, bboxes, self.model_w, self.model_h, jitter=self.jitter)

                orig_size = np.array([image_data.shape[1], image_data.shape[0]])
                orig_size = np.expand_dims(orig_size, axis=0)
                # normalize the image data 
                image_data /= 255.

                batch_images.append(image_data)
                
                boxes = bboxes.reshape((-1, 5))
                boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
                boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
                boxes_xy = boxes_xy / orig_size
                boxes_wh = boxes_wh / orig_size
                boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)
                batch_boxes.append(boxes)

            # find the max number of boxes
            max_boxes = 0
            for boxz in batch_boxes:
                if boxz.shape[0] > max_boxes:
                    max_boxes = boxz.shape[0]
            
            # add zero pad for training
            for i, boxz in enumerate(batch_boxes):
                if boxz.shape[0]  < max_boxes:
                    zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                    batch_boxes[i] = np.vstack((boxz, zero_padding))

                        
            batch_images = np.array(batch_images)
            batch_boxes = np.array(batch_boxes)
            detectors_mask, matching_true_boxes = yolo_get_detector_mask(batch_boxes, anchors, model_shape=[self.model_h, self.model_w])
            X_train = [batch_images, batch_boxes, detectors_mask, matching_true_boxes]
            y_train = np.zeros(len(batch_images))
            yield X_train, y_train
    

def train(model, class_names, anchors, train_batch_gen, valid_batch_gen, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    train_steps_per_epoch = train_batch_gen.unique_data_instances // BATCH_SIZE
    valid_steps_per_epoch = valid_batch_gen.unique_data_instances // BATCH_SIZE

    print('train_steps_per_epoch=',train_steps_per_epoch);
    print('valid_steps_per_epoch=',valid_steps_per_epoch);
    
    num_epochs = 30 

    model.fit_generator(generator       = train_batch_gen.flow_from_hdf5(),
                        validation_data = valid_batch_gen.flow_from_hdf5(),
                        steps_per_epoch = train_steps_per_epoch,
                        validation_steps= valid_steps_per_epoch,
                        callbacks       = [checkpoint, logging],
                        epochs          = num_epochs,
                        workers=1, 
                        verbose=1)
    model.save_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit_generator(generator       = train_batch_gen.flow_from_hdf5(),
                        validation_data = valid_batch_gen.flow_from_hdf5(),
                        steps_per_epoch = train_steps_per_epoch,
                        validation_steps= valid_steps_per_epoch,
                        callbacks       = [checkpoint, logging],
                        epochs          = num_epochs,
                        workers=1, 
                        verbose=1)

    model.save_weights('trained_stage_2.h5')


    model.fit_generator(generator       = train_batch_gen.flow_from_hdf5(),
                        validation_data = valid_batch_gen.flow_from_hdf5(),
                        steps_per_epoch = train_steps_per_epoch,
                        validation_steps= valid_steps_per_epoch,
                        callbacks       = [checkpoint, logging],
                        epochs          = num_epochs,
                        workers=1, 
                        verbose=1)

    model.save_weights('trained_stage_3.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()
if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
