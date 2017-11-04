"""Miscellaneous utility functions."""

from functools import reduce
import cv2
import copy
import numpy as np
import io
import PIL

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def brightness_augment(image):
    image = np.array(image, dtype=np.uint8)
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = 0.5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1[:, :, 2][image1[:, :, 2] > 255]  = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def _remap_object_boxes(boxes, class_names, target_class_names):
    """ Remap original object labels into the interested target object labels
    """
    drone_box = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        voc_label = class_names[box[0]]  
        if voc_label in target_class_names:
            dbox = copy.deepcopy(box)
            dbox[0] = target_class_names.index(voc_label)
            drone_box.append(dbox)
    return np.array(drone_box)

def read_voc_datasets_train_batch(data_images, data_boxes):  
    idx = np.random.choice(data_images.shape[0], replace=False)
    batch_image = data_images[idx]
    batch_boxes = data_boxes[idx]
    batch_boxes = batch_boxes.reshape((-1, 5))    
    image = PIL.Image.open(io.BytesIO(batch_image))
    orig_size = np.array([image.width, image.height])
    orig_size = np.expand_dims(orig_size, axis=0)
    image_data = np.array(image, dtype=np.float)           
    return np.array(image_data), np.array(batch_boxes)

def augment_image(image_data, bboxes, model_width, model_height, jitter=False):
    h, w, c = image_data.shape
    if jitter:        
        scale = np.random.uniform() / 10. + 1.
        image_data = cv2.resize(image_data, (0,0), fx = scale, fy = scale)    
        ## translate the image
        max_offx = (scale-1.) * w
        max_offy = (scale-1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        image_data = image_data[offy : (offy + h), offx : (offx + w)]        
        flip = np.random.binomial(1, .5)
        if flip > 0.5: 
            image_data = cv2.flip(image_data, 1)
            
    image_data = cv2.resize(image_data, (model_height, model_width))
    bboxes2 = copy.deepcopy(bboxes)
    for bbox in bboxes2:
        for attr in (1,3): # adjust xmin and xmax
            if jitter: bbox[attr] = int(bbox[attr] * scale - offx)
            bbox[attr] = int(bbox[attr] * float(model_width) / w)     
            bbox[attr] = max(min(bbox[attr], model_width), 0)
   
        for attr in 2,4: # adjust ymin and ymax
            if jitter: bbox[attr] = int(bbox[attr] * scale - offy)
            bbox[attr] = int(bbox[attr] * float(model_height) / h)
            bbox[attr] = max(min(bbox[attr], model_height), 0)
        if jitter and flip > 0.5:
            xmin = bbox[1]
            bbox[1] = model_width - bbox[3]
            bbox[3] = model_width - xmin  
    return image_data, bboxes2
