"""
Convert Okutama Action dataset to VOC similar dataset 
Store the processed data into HDF5 file
The original dataset is for object predestria tracking and people action understasnding
We remap the orignal labels by only keeping object detection labels 
"""
# Original Labels: Each line contains 10+ columns, separated by spaces. The definition of these columns are:
#     Track ID. All rows with the same ID belong to the same person for 180 frames. Then the person gets a new idea for the next 180 frames. We will soon release an update to make the IDs consistant.
#     xmin. The top left x-coordinate of the bounding box.
#     ymin. The top left y-coordinate of the bounding box.
#     xmax. The bottom right x-coordinate of the bounding box.
#     ymax. The bottom right y-coordinate of the bounding box.
#     frame. The frame that this annotation represents.
#     lost. If 1, the annotation is outside of the view screen.
#     occluded. If 1, the annotation is occluded.
#     generated. If 1, the annotation was automatically interpolated.
#     label. The label for this annotation, enclosed in quotation marks.
#     (+) actions. Each column after this is an action.

# There are two label files for each video; 
# one for single-action detection and one for multi-action detection.
#  Note that labels for single-action detection has been created from the multi-action detection labels 
# (for more details please refer to our publication). 
# For pedestrian detection task, the columns describing the actions should be ignored.

# Object detection Labels:
#     labels. Always be 0 ('person') for this dataset 
#     xmin. The top left x-coordinate of the bounding box.
#     ymin. The top left y-coordinate of the bounding box.
#     xmax. The bottom right x-coordinate of the bounding box.
#     ymax. The bottom right y-coordinate of the bounding box.

import numpy as np
import os 
import glob 
import cv2
import argparse
import fnmatch
import h5py
import random 
import copy 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

classes = ["person", "bus", "car", "train"]

parser = argparse.ArgumentParser(
    description='Merge multiple HDF5 datasets into a singleone.')

parser.add_argument(
    '-o',
    '--data_output',
    help='path to output HDF5',
    default='~/data/')

parser.add_argument(
    '-i',
    '--input_hdf5s',
    help='path to input hdf5 files',
    nargs=argparse.ONE_OR_MORE)

parser.add_argument(
    '-d',
    '--draw',
    help='draw bound boxes on each image and output to /tmp',
    default=False)

def draw_bboxes(image, bboxes):
    """Draw the bounding boxes on raw or jpg images"""
    decoded_image = copy.deepcopy(image)
    decoded_image = cv2.imdecode(decoded_image, 1)
    if bboxes is None:
        return decoded_image
    corners = bboxes[:, 1:]
    corners = np.array(corners, dtype=np.int)
    for corner in corners:
        cv2.rectangle(decoded_image, (corner[0], corner[1]),(corner[2], corner[3]), (0,255,0), 5)
    return decoded_image

def draw_on_images(dataset_images, dataset_boxes, out_dir='/tmp/combined/'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(dataset_images.shape[0]):
        boxes = np.array(dataset_boxes[i]).reshape(-1, 5)
        img = draw_bboxes(dataset_images[i], boxes)
        out_img_path = os.path.join(out_dir, str(i)+'.jpg')
        cv2.imwrite(out_img_path, img)
    return 

def _main(args):
    draw_enabled = args.draw
    output_path = os.path.expanduser(args.data_output)

    input_hdf5 = []
    num_datasets = len(args.input_hdf5s)
    if num_datasets < 2:
        print('Number of hd5f files is :' + str(num_datasets))
        print('Nothing to combine')
        return 
    
    num_samples_images_train = 0
    num_samples_bboxes_train = 0
    num_samples_images_valid = 0
    num_samples_bboxes_valid = 0
     
    for f in args.input_hdf5s:
        in_file = h5py.File(f, 'r')
        input_hdf5.append(in_file)
        num_samples_images_train += in_file['train/images'].shape[0]
        num_samples_bboxes_train += in_file['train/boxes'].shape[0]
        num_samples_images_valid += in_file['valid/images'].shape[0]
        num_samples_bboxes_valid += in_file['valid/boxes'].shape[0]

    # images and boxes must have the same size
    assert(num_samples_images_train == num_samples_bboxes_train)
    assert(num_samples_images_valid == num_samples_bboxes_valid)
    
    num_samples_train = num_samples_images_train
    num_samples_valid = num_samples_images_valid
    
    # Create HDF5 dataset structure
    print('Creating output HDF5 dataset structure.')
    print('Total train: ' + str(num_samples_train))
    print('Total valid: ' + str(num_samples_valid))

    if not os.path.exists(output_path):
        print('Creating ' + output_path)
        os.mkdir(output_path)

    fname = os.path.join(output_path, 'combined.hdf5')
    if os.path.exists(fname):
        print('Removing old ' + fname)
        os.remove(fname)

    # Create HDF5 dataset structure
    print('Creating HDF5 dataset structure.')
    combined = h5py.File(fname, 'w')

    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))  # variable length uint8
    int32_dt = h5py.special_dtype(
        vlen=np.dtype('int32'))  # variable length uint8

    vlen_int_dt = h5py.special_dtype(
        vlen=np.dtype(int))  # variable length default int

    train_group = combined.create_group('train')  
    valid_group = combined.create_group('valid')  

    # store class list for reference class ids as csv fixed-length numpy string
    combined.attrs['classes'] = np.string_(str.join(',', classes))

    # store images as variable length uint8 arrays
    dataset_train_images = train_group.create_dataset(
        'images', shape=(num_samples_train, ), dtype=uint8_dt)

    dataset_valid_images = valid_group.create_dataset(
        'images', shape=(num_samples_valid, ), dtype=uint8_dt)

    # store images as variable length uint8 arrays
    dataset_train_boxes = train_group.create_dataset(
        'boxes', shape=(num_samples_train, ), dtype=int32_dt)

    dataset_valid_boxes = valid_group.create_dataset(
        'boxes', shape=(num_samples_valid, ), dtype=int32_dt)

    # combine the input hdf5 into the new dataset
    # combine train and valid data
    Xtrain = []
    ytrain = []
    Xvalid = []
    yvalid = []
    # Note: this might use a large chunk of memory as all the data are loaded into memory first and then 
    # we randomly shuffe it
    for hdf5 in input_hdf5:
        for i in range(hdf5['train/images'].shape[0]):
            Xtrain.append(hdf5['train/images'][i])
            ytrain.append(hdf5['train/boxes'][i])
        for i in range(hdf5['valid/images'].shape[0]):
            Xvalid.append(hdf5['valid/images'][i])
            yvalid.append(hdf5['valid/boxes'][i])

    Xtrain, ytrain = shuffle(Xtrain, ytrain)
    Xvalid, yvalid = shuffle(Xvalid, yvalid)

    for i in range(num_samples_train):
        dataset_train_images[i] = Xtrain[i]
        dataset_train_boxes[i] = ytrain[i]
    for i in range(num_samples_valid):
        dataset_valid_images[i] = Xvalid[i]
        dataset_valid_boxes[i] = yvalid[i]

    if draw_enabled:
        num_check = 1000
        draw_on_images(dataset_train_images[0:num_check], dataset_train_boxes[0:num_check], '/tmp/train')
        draw_on_images(dataset_valid_images[0:num_check], dataset_valid_boxes[0:num_check], '/tmp/valid')
    
    combined.close()
    for hdf5 in input_hdf5:
        hdf5.close()
    print('Done combining')

if __name__ == '__main__':
    _main(parser.parse_args())