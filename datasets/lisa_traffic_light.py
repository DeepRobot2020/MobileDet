# https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset/home

import os 
import glob 
import cv2
import argparse
import fnmatch
import h5py
import csv
import pandas as pd
import numpy as np
import copy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

LISA_UDACITY_CLASSES = ["stop", "go", "warning", "donotcare"]


parser = argparse.ArgumentParser(
    description='Convert LISA dataset to HDF5.')

parser.add_argument(
    '-p',
    '--seq_path',
    help='path to UAV123 dataseq',
    default='~/data/UAV123/UAV123_10fps/data_seq/UAV123_10fps/')

parser.add_argument(
    '-a',
    '--anno_path',
    help='path to UAV123 annotation',
    default='~/data/UAV123/UAV123_10fps/anno/UAV123_10fps/')

parser.add_argument(
    '-s',
    '--selected',
    help='path to UAV123 selected',
    default='datasets/UAV123_selected/')

parser.add_argument(
    '-f',
    '--hdf5_path',
    help='path to output UAV123 hdf5',
    default='~/data/UAV123/UAV123_10fps/')

parser.add_argument(
    '-d',
    '--verify_only',
    help='verify the UAV HDF5 dataset',
    default=False)



def parse_lisa_annotations(annotations_path, lisa_seq_names):
    ''' Parse LISA annotations
    '''
    csv_files_dict = {}
    for fname in lisa_seq_names:
        csv_path = os.path.join(annotations_path, fname)
        csv_file = os.path.join(csv_path, 'frameAnnotationsBOX.csv')
        csv_file = os.path.expanduser(csv_file)
        if not os.path.exists(csv_file):
            print(csv_file  + 'does not exit')
        else:
            print('found anno file ' + csv_file)
        csv_files_dict[fname] = csv_file
    return csv_files_dict


def load_images_from_csv(annotations_path, images_path, lisa_seq_names):
    ''' Parse the csvs and update image paths
    '''
    csv_files_dict = parse_lisa_annotations(annotations_path, lisa_seq_names)
    images_path = os.path.expanduser(images_path)
    lisa_dict = {}
    for seq in lisa_seq_names:
        csv_file = csv_files_dict[seq]
        col_name = 'Filename'
        col_tag = 'Annotation tag'
        col_x0 = 'Upper left corner X'
        col_y0 = 'Upper left corner Y'
        col_x1 = 'Lower right corner X'
        col_y1 = 'Lower right corner Y'
        col_names = [col_name, col_tag, col_x0, col_y0, col_x1, col_y1]
        signs_of_interest = ['stop', 'go', 'warning']
        ann_csv = pd.read_csv(csv_file, sep=';', usecols=col_names)
        # update the path and change the traffic lights which do not care label 
        # we will use those as negative training
        for i in range(ann_csv.shape[0]):
            org_name = ann_csv[col_name].values[i]
            img_name = org_name.split('/')[-1]
            seq_image_path = os.path.join(images_path, seq)
            seq_image_path = os.path.join(seq_image_path, 'frames')
            image_path = os.path.join(seq_image_path, img_name)
            ann_csv[col_name].values[i] = image_path
            if ann_csv[col_tag].values[i] not in signs_of_interest:
                ann_csv[col_tag].values[i] = 'donotcare'
            # convert to training bbox
            label = ann_csv[col_tag].values[i]
            bbox = [LISA_UDACITY_CLASSES.index(label), 
                    int(ann_csv[col_x0].values[i]),
                    int(ann_csv[col_y0].values[i]),
                    int(ann_csv[col_x1].values[i]),
                    int(ann_csv[col_y1].values[i])]
            if image_path not in lisa_dict:
                lisa_dict[image_path] = []
            lisa_dict[image_path].append(bbox)
    return lisa_dict


def add_to_dataset(lisa_dict, keys, dataset_images, dataset_boxes):
    dataset_images.resize(len(keys), axis=0)
    dataset_boxes.resize(len(keys), axis=0)

    for i, image_file in enumerate(keys):
        if not os.path.exists(image_file):
            print('warn: cannot find image file: ' + image_file)
            continue
        image_boxes = np.array(lisa_dict[image_file])
        if len(image_boxes) == 0:
            print('warn: no bbox for image file: ' + image_file)
        with open(image_file, 'rb') as in_file:
            image_data = in_file.read()
            image_data = np.fromstring(image_data, dtype='uint8')
            dataset_images[i] = image_data
            dataset_boxes[i] = image_boxes.flatten('C')

def draw_bboxes(image, bboxes):
    decoded_image = copy.deepcopy(image)
    if image.shape[0] > 3180:
        decoded_image = cv2.imdecode(image, 1)
    if bboxes is None:
        return decoded_image
    corners = bboxes[:, 1:]
    corners = np.array(corners, dtype=np.int)
    for corner in corners:
        cv2.rectangle(decoded_image, (corner[0], corner[1]),(corner[2], corner[3]), (0,255,0), 10)
    return decoded_image

def draw_on_images(dataset_images, dataset_boxes, out_dir='/tmp/'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(dataset_images.shape[0]):
        boxes = np.array(dataset_boxes[i]).reshape(-1, 5)
        img = draw_bboxes(dataset_images[i], boxes)
        out_img_path = os.path.join(out_dir, str(i)+'.jpg')
        cv2.imwrite(out_img_path, img)

def _main(args):
    LISA_ANNO_PATH ='~/data/LISA/Annotations'
    LISA_IMAGE_PATH ='~/data/LISA/lisa-traffic-light-dataset'
    LISA_SEQS = ['daySequence1', 'daySequence2']
    hdf5_path = os.path.join('/tmp', 'lisa.hdf5')
    # Convert CSV to 
    lisa_dict = load_images_from_csv(LISA_ANNO_PATH, LISA_IMAGE_PATH, LISA_SEQS)
    keys = list(lisa_dict.keys())
    keys_train, keys_valid = train_test_split(keys, shuffle=True, test_size = 0.25)
    

    lisa_h5file = h5py.File(hdf5_path, 'w')

    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))  # variable length uint8

    uint32_dt = h5py.special_dtype(
        vlen=np.dtype('uint32'))  # variable length uint8

    train_group = lisa_h5file.create_group('train')
    valid_group = lisa_h5file.create_group('valid')

    # store class list for reference class ids as csv fixed-length numpy string
    lisa_h5file.attrs['classes'] = np.string_(str.join(',', LISA_UDACITY_CLASSES))

    # store images as variable length uint8 arrays
    dataset_train_images = train_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)

    dataset_valid_images = valid_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)


    dataset_train_boxes = train_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=uint32_dt)

    dataset_valid_boxes = valid_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=uint32_dt)

    add_to_dataset(lisa_dict, keys_train, dataset_train_images, dataset_train_boxes)
    add_to_dataset(lisa_dict, keys_valid, dataset_valid_images, dataset_valid_boxes)

    lisa_h5file.close()

    print("Verifying the HD5 data....")
    if not os.path.exists(hdf5_path):
        print(hdf5_path + " does not exits!")
        return 
    uav123 = h5py.File(hdf5_path, 'r')
    print("Verifying the training data....")
    draw_on_images(uav123['train/images'], uav123['train/boxes'])
    print("Verifying the validation data....")
    draw_on_images(uav123['valid/images'], uav123['valid/boxes'])
    uav123.close()

if __name__ == '__main__':
    _main(parser.parse_args())
