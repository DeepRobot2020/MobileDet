"""YOLO_v2 Model Defined in Keras."""
import sys
import numpy as np
import os

import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation
from keras.applications.mobilenet import MobileNet


# from ..utils import compose
from .keras_darknet19 import (DarknetConv2D, DarknetConv2D_BN_Leaky,
                              darknet19)
from .keras_mobilenet import _depthwise_conv_block, mobile_net
from cfg import *

sys.path.append('..')

def yolo_get_detector_mask(boxes, anchors, model_shape=[416, 416], feature_shape=[19, 19]):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, model_shape, feature_shape)
    return np.array(detectors_mask), np.array(matching_true_boxes)

def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x4(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=4."""
    # Import currently required to make Lambda work.
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=4)

def relu_6(x):
    return K.relu(x, max_value=6)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def space_to_depth_x4_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 4, input_shape[2] // 4, 16 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    16 * input_shape[3])

def yolo_body_darknet(inputs, num_anchors, num_classes, weights='yolov2', network_config=[False, False]):
    """Create YOLO_V2 model CNN body in Keras."""
    fine_grained_layers = [17, 27, 43]

    shallow_detector, use_x0 = network_config

    if shallow_detector:
        fine_grained_layers = fine_grained_layers[0:2]
        num_final_layers = 512
        final_feature_layer = 43
    else:
        fine_grained_layers = fine_grained_layers[1:]
        num_final_layers = 1024
        final_feature_layer = -1

    feature_model = darknet19(inputs, include_top=False)
    feature_model = Model(inputs=feature_model.input, outputs=feature_model.layers[final_feature_layer].output)
    
    if weights == 'yolov2':
        print("Loading pre-trained yolov2 weights")
        # Save topless yolo:
        yolo_path = os.path.join('model_data', 'yolo.h5')
        trained_model = load_model(yolo_path)
        # trained_model = Model(trained_model.inputs, trained_model.output)   
        trained_layers = trained_model.layers
        feature_layers = feature_model.layers
        for i in range(0, min(len(feature_layers), len(trained_layers))):
            weights = trained_layers[i].get_weights()
            feature_layers[i].set_weights(weights)
            
    x2 = feature_model.output
    x1 = feature_model.layers[fine_grained_layers[1]].output
    x0 = feature_model.layers[fine_grained_layers[0]].output

    x2 = DarknetConv2D_BN_Leaky(num_final_layers, (3, 3))(x2)    
    x2 = DarknetConv2D_BN_Leaky(num_final_layers, (3, 3))(x2)    

    x1 = DarknetConv2D_BN_Leaky(64, (1, 1))(x1)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    x1_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth_x2')(x1)
        
    x0 = DarknetConv2D_BN_Leaky(16, (1, 1))(x0)
    # TODO:  #304Allow Keras Lambda to use func arguments for output_shape?
    x0_reshaped = Lambda(
        space_to_depth_x4,
        output_shape=space_to_depth_x4_output_shape,
        name='space_to_depth_x4')(x0)   

    if use_x0:
        x = concatenate([x0_reshaped, x1_reshaped, x2])
    else:
        x = concatenate([x1_reshaped, x2])
    x = DarknetConv2D_BN_Leaky(num_final_layers, (3, 3))(x)
    
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)
    return Model(feature_model.inputs, x)

def yolo_body_mobilenet(inputs, num_anchors, num_classes, weights='imagenet', network_config=[False, False]):    
    """
    Mobile Detector Implementation
    :param feature_extractor:
    :param num_classes:
    :param num_anchors:
    :return:
    """
    fine_grained_layers = [17, 27, 43]
    shallow_detector, use_x0 = network_config

    if shallow_detector:
        fine_grained_layers = fine_grained_layers[0:2]
        num_final_layers = 512
        final_feature_layer = 69
    else:
        fine_grained_layers = fine_grained_layers[1:]
        num_final_layers = 1024
        final_feature_layer = -1
        
    feature_model = MobileNet(input_tensor=inputs, include_top=False, weights=None)
    feature_model = Model(inputs=feature_model.input, outputs=feature_model.layers[final_feature_layer].output)

    if weights == 'imagenet':
        print('Loading pretrained weights from ImageNet...')
        trained_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        trained_layers = trained_model.layers
        feature_layers = feature_model.layers
        for i in range(0, min(len(feature_layers), len(trained_layers))):
            weights = trained_layers[i].get_weights()
            feature_layers[i].set_weights(weights)

    x2 = feature_model.output
    x1 = feature_model.layers[fine_grained_layers[1]].output
    x0 = feature_model.layers[fine_grained_layers[0]].output

    x2 = _depthwise_conv_block(x2, num_final_layers, 1.0, block_id=14)
    x2 = _depthwise_conv_block(x2, num_final_layers, 1.0, block_id=15)

    # Reroute x1
    x1 = Conv2D(64, (1, 1), padding='same', use_bias=False, strides=(1, 1))(x1)
    x1 = BatchNormalization()(x1)
    # To keep keras to tensorflow conversion happy 
    x1 = Lambda(relu_6)(x1)

    x1_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth_x2')(x1)

    # Reroute x0
    x0 = Conv2D(16, (1, 1), padding='same', use_bias=False, strides=(1, 1))(x0)
    x0 = BatchNormalization()(x0)
    x0 = Lambda(relu_6)(x0)

    x0_reshaped = Lambda(
        space_to_depth_x4,
        output_shape=space_to_depth_x4_output_shape,
        name='space_to_depth_x4')(x0)

    if use_x0:
        x = concatenate([x0_reshaped, x1_reshaped, x2])
    else:
        x = concatenate([x1_reshaped, x2])

    x = _depthwise_conv_block(x, num_final_layers, 1.0, block_id=16)

    x = Conv2D(num_anchors * (num_classes + 5), (1, 1))(x)

    model = Model(inputs=feature_model.input, outputs=x)
    return model


def decode_yolo_output(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_loss(args,
              anchors,
              num_classes,
              rescore_confidence=False,
              print_loss=False):
    """YOLO localization loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    print_loss : bool, default=False
        If True then use a tf.Print() to print the loss components.

    Returns
    -------
    mean_loss : float
        mean localization loss across minibatch
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xy, pred_wh, pred_confidence, pred_class_prob = decode_yolo_output(
        yolo_output, anchors, num_classes)

    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo_head.
    yolo_output_shape = K.shape(yolo_output)
    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        num_classes + 5
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio.

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))

    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))

    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
        
    if print_loss:
        total_loss = tf.Print(
            total_loss, [
                total_loss, confidence_loss_sum, classification_loss_sum,
                coordinates_loss_sum
            ],
            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

    return total_loss


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes


def preprocess_true_boxes(true_boxes, anchors, image_size, feature_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.
    feature_size : array-like
        List of feature dimensions in form of h, w in pixels.
    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    feature_height, feature_width = feature_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = int(feature_height)
    conv_width = int(feature_width)

    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array(
            [conv_height, conv_width, conv_height, conv_width])

        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k
        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes

def create_model(anchors, class_names, feature_extractor='darknet19', 
    load_pretrained=False, pretrained_path=None, freeze_body=False):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''
    num_anchors = len(anchors)

    detectors_mask_shape = (FEAT_H, FEAT_W, num_anchors, 1)
    matching_boxes_shape = (FEAT_H, FEAT_W, num_anchors, 5)

    # Create model input layers.
    image_input = Input(shape=(IMAGE_H, IMAGE_W, 3))
    boxes_input = Input(shape=(None, 5))

    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    if feature_extractor == 'darknet19':
        yolo_model = yolo_body_darknet(image_input, len(anchors), len(class_names), weights=None,
            network_config=[SHALLOW_DETECTOR, USE_X0_FEATURE])
    elif feature_extractor == 'mobilenet':
        yolo_model = yolo_body_mobilenet(image_input, len(anchors), len(class_names), weights=None,
            network_config=[SHALLOW_DETECTOR, USE_X0_FEATURE])
    else:
        assert(False)
        
    # yolo_model.summary()
    
    if load_pretrained:
        if pretrained_path:
            yolo_model.load_weights(pretrained_path)
        else:
            print('No pretrained weights!')
            
    if freeze_body:
        for layer in yolo_model.layers:
            layer.trainable = False

    model_body = Model(image_input, yolo_model.output)
    # model_body.summary()
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
