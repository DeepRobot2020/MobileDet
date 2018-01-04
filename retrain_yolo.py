"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import h5py

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from mobiledet.models.keras_yolo import preprocess_true_boxes, yolo_loss
from mobiledet.models.keras_yolo import yolo_eval, yolo_loss, decode_yolo_output, create_model
from mobiledet.models.keras_yolo import yolo_body_darknet, yolo_body_mobilenet
from mobiledet.models.keras_yolo import yolo_get_detector_mask
from mobiledet.utils.draw_boxes import draw_boxes

from mobiledet.utils import *
from cfg import *

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default='~/data/pascal_voc_07_12_person_vehicle.hdf5')

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'pascal_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to drone_classes.txt',
    default='model_data/drone_classes.txt')



def _main(args):
    data_path    = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    class_names  = get_classes(classes_path)
    anchors      = get_anchors(anchors_path)

    anchors = YOLO_ANCHORS
    if SHRINK_FACTOR == 16:
        anchors = anchors *2
    print('Anchors:')
    print(anchors)
    
    # custom data saved as a numpy file.
    h5_data = h5py.File(data_path, 'r')
  
    train_boxes = np.array(h5_data['train/boxes'])
    train_images = np.array(h5_data['train/images'])

    valid_boxes = np.array(h5_data['valid/boxes'])
    valid_images = np.array(h5_data['valid/images'])
    # clear any previous sesson
    K.clear_session()

    model_body, model = create_model(anchors, class_names, 
        feature_extractor=FEATURE_EXTRACTOR, load_pretrained=True, pretrained_path='/home/jzhang/github/MobileDet/weights/darknet19_s2_best.FalseFalse.h5')

    train_batch_gen = DataBatchGenerator(train_images, train_boxes, IMAGE_W, IMAGE_H, FEAT_W, FEAT_H, anchors, class_names, jitter=True)
    valid_batch_gen = DataBatchGenerator(valid_images, valid_boxes, IMAGE_W, IMAGE_H, FEAT_W, FEAT_H, anchors, class_names, jitter=True)
    train(
        model,
        class_names,
        anchors, 
        train_batch_gen,
        valid_batch_gen)

class DataBatchGenerator:
    def __init__(self, H5_IMAGES, 
                       H5_BOXES,
                       model_w, 
                       model_h, 
                       feat_w,
                       feat_h, 
                       anchors,
                       class_names, 
                       jitter=False):
        self.model_w = model_w
        self.model_h = model_h
        self.feat_w  = feat_w
        self.feat_h  = feat_h
        self.class_names =  class_names        
        self.H5_BOXES    = H5_BOXES
        self.H5_IMAGES   = H5_IMAGES
        self.anchors     = anchors
        self.unique_data_instances = self.H5_IMAGES.shape[0]
        self.jitter = jitter

    def flow_from_hdf5(self):
        while True:
            batch_images = []
            batch_boxes = []
            for i in range(BATCH_SIZE):        
                image_data, bboxes = read_voc_datasets_train_batch(self.H5_IMAGES, self.H5_BOXES)
                image_data, bboxes = augment_image(image_data, bboxes, self.model_w, self.model_h, self.jitter)

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
            detectors_mask, matching_true_boxes = yolo_get_detector_mask(batch_boxes, self.anchors, [self.model_h, self.model_w], [self.feat_h, self.feat_w])
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
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    train_steps_per_epoch = train_batch_gen.unique_data_instances // BATCH_SIZE
    valid_steps_per_epoch = valid_batch_gen.unique_data_instances // BATCH_SIZE

    print('train_steps_per_epoch=',train_steps_per_epoch);
    print('valid_steps_per_epoch=',valid_steps_per_epoch);
    
    num_epochs = 25
    weight_name = 'weights/{}_s1_best.{}{}.h5'.format(FEATURE_EXTRACTOR, SHALLOW_DETECTOR, USE_X0_FEATURE)

    checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', save_weights_only=True, save_best_only=True)
    model.fit_generator(generator       = train_batch_gen.flow_from_hdf5(),
                        validation_data = valid_batch_gen.flow_from_hdf5(),
                        steps_per_epoch = train_steps_per_epoch,
                        validation_steps= valid_steps_per_epoch,
                        callbacks       = [checkpoint, logging],
                        epochs          = num_epochs,
                        workers=1, 
                        verbose=1)
    model.save_weights('weights/{}_s1.{}.{}.h5'.format(FEATURE_EXTRACTOR, SHALLOW_DETECTOR, USE_X0_FEATURE))

    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-06)
    model.compile(
        optimizer=adam, loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    weight_name = 'weights/{}_s2_best.{}{}.h5'.format(FEATURE_EXTRACTOR, SHALLOW_DETECTOR, USE_X0_FEATURE)
    checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', save_weights_only=True, save_best_only=True)
    model.fit_generator(generator       = train_batch_gen.flow_from_hdf5(),
                        validation_data = valid_batch_gen.flow_from_hdf5(),
                        steps_per_epoch = train_steps_per_epoch,
                        validation_steps= valid_steps_per_epoch,
                        callbacks       = [checkpoint, logging],
                        epochs          = num_epochs,
                        workers=1, 
                        verbose=1)

    model.save_weights('weights/{}_s2.{}.{}.h5'.format(FEATURE_EXTRACTOR, SHALLOW_DETECTOR, USE_X0_FEATURE))
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-06)
    model.compile(
        optimizer=adam, loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  
    
    weight_name = 'weights/{}_s2_best.{}{}.h5'.format(FEATURE_EXTRACTOR, SHALLOW_DETECTOR, USE_X0_FEATURE)
    checkpoint = ModelCheckpoint(weight_name, monitor='val_loss', save_weights_only=True, save_best_only=True)
    model.fit_generator(generator       = train_batch_gen.flow_from_hdf5(),
                        validation_data = valid_batch_gen.flow_from_hdf5(),
                        steps_per_epoch = train_steps_per_epoch,
                        validation_steps= valid_steps_per_epoch,
                        callbacks       = [checkpoint, logging],
                        epochs          = num_epochs,
                        workers=1, 
                        verbose=1)
    model.save_weights('weights/{}_s3.{}.{}.h5'.format(FEATURE_EXTRACTOR, SHALLOW_DETECTOR, USE_X0_FEATURE))

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

    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = decode_yolo_output(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction
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
