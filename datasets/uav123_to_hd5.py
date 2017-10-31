"""
Convert UAV123 dataset to VOC similar dataset 
Store the processed data into HDF5 file
The original dataset is for UAV object tracking
"""

import numpy as np
import os 
import glob 
import cv2
import argparse
import fnmatch
import h5py
import random 
import copy 
import re

from sklearn.model_selection import train_test_split

classes = ['person', 'car']

parser = argparse.ArgumentParser(
    description='Convert UAV123 dataset to HDF5.')

parser.add_argument(
    '-p',
    '--path_dataseq',
    help='path to UAV123 dataseq',
    default='~/data/UAV123/UAV123_10fps/data_seq/UAV123_10fps/')

parser.add_argument(
    '-p',
    '--path_anno',
    help='path to UAV123 annotation',
    default='~/data/UAV123/UAV123_10fps/anno/UAV123_10fps/')

seq_path = '~/data/UAV123/UAV123_10fps/data_seq/UAV123_10fps/'
ann_path = '~/data/UAV123/UAV123_10fps/anno/UAV123_10fps/'

seq_path = os.path.expanduser(seq_path)
ann_path = os.path.expanduser(ann_path)
  
def find_car_person_folders(seq_path):    
    """ Find folders with car or person pictures
    """
    assert(os.path.exists(seq_path))
    folders = os.listdir(seq_path)
    if len(folders):
        print('No folders in ' + seq_path)
    car_person_folders = []
    for folder in folders:
        person_match = re.search(r'person\d+$', folder)
        car_match = re.search(r'car\d+$', folder)
        if person_match is not None:
            car_person_folders.append(person_match.group(0))
            print('Adding folder ' + person_match.group(0))
        elif car_match is not None:
            car_person_folders.append(car_match.group(0))
            print('Adding folder ' + car_match.group(0))
    return sorted(car_person_folders)

def find_car_person_anns(ann_path):
    """Find annoation files related to car and persons
    """
    car_re = r'car\d+(_\d+)?.txt'
    person_re = r'person\d+(_\d+)?.txt'
    car_person_ann = []
    for ann in os.listdir(ann_path):
        car_match = re.match(car_re, ann)
        person_match = re.match(person_re, ann)
        if car_match:
            car_person_ann.append(car_match.group(0))
            print('Adding Car anno ' + car_match.group(0))
        elif person_match:
            car_person_ann.append(person_match.group(0))
            print('Adding Person anno ' + person_match.group(0))
    return sorted(car_person_ann)

def parse_anno_to_bboxes(ann_path, ann_file):
    """Parse a annotation file into bound boxs (batch_size, xc, yc, w, h) 
    """
    ann_file = os.path.join(ann_path, ann_file)
    assert(os.path.exists(ann_file))
    bboxes = []
    with open(ann_file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            line = line.split(',')
            try:
                line = [int(a) for a in line]
            except ValueError:
                line = [0, 0, 0, 0]
            line = np.array(line)
            bboxes.append(line)
    return np.array(bboxes)

# imgs, anns = match_dataseq_anno(seq_path, ann_path)

def match_dataseq_anno(seq_path, ann_path):
    car_person_folders = find_car_person_folders(seq_path)
    car_person_anns = find_car_person_anns(ann_path)
    annotations = []
    object_images = []
    for folder in car_person_folders:
        imgs = os.listdir(os.path.join(seq_path, folder))
        imgs = sorted(imgs)
        imgs = [os.path.join(seq_path, folder, img) for img in imgs]
        print('Folder name: ' + folder)
        print('Total images: ' + str(len(imgs)))
        ann_name1 = folder + '.txt'
        ann_name2 = folder + '_'
        anns = [ann for ann in car_person_anns if ann_name1 in ann or ann_name2 in ann]
        anns = sorted(anns)
        anns = [os.path.join(ann_path, ann) for ann in anns]
        ann_data = ''.join([open(f).read() for f in anns])
        ann_data = ann_data.split('\n')
        parsed_anns = []
        for ann in ann_data:
            ann = ann.split(',')
            try:
                ann = [int (a) for a in ann]
            except ValueError:
                ann = [0, 0, 0, 0]
            parsed_anns.append(ann)
        annotations.append(np.array(parsed_anns))
        object_images.append(imgs)
        print('Total annotations: ' + str(len(parsed_anns)))
    return object_images, annotations

def draw(images, bboxes, name_hint='debug'):
    xmin, ymin = bboxes[:,0],bboxes[:,1]
    xmax, ymax = xmin + bboxes[:,2], ymin + bboxes[:,3] 
    corners = np.concatenate((xmin.reshape(-1,1), ymin.reshape(-1,1), xmax.reshape(-1,1), ymax.reshape(-1,1)), axis=1)
    corners = np.array(corners, dtype=np.int)
    for i in range(min(len(images), len(bboxes))):
        img = cv2.imread(images[i])
        corner = corners[i]
        cv2.rectangle(img, (corner[0], corner[1]),(corner[2], corner[3]), (0,255,0), 10)
        out_dir = os.path.join('/tmp', name_hint)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_img_path = os.path.join(out_dir, str(i)+'.jpg')
        cv2.imwrite(out_img_path, img)
    return

for idx in range(41):
    if idx == 7:
        continue
    draw(imgs[idx], anns[idx], str(idx))


def _main(args):
    videos_path = os.path.expanduser(args.path_to_video)
    labels_path = os.path.expanduser(args.path_to_labels)
    hdf5_path   = os.path.expanduser(args.path_to_hdf5)
    if not os.path.exists(hdf5_path):
        print('Creating ' + hdf5_path)
        os.mkdir(hdf5_path)
    # Create HDF5 dataset structure
    print('Creating HDF5 dataset structure.')
    fname = os.path.join(hdf5_path, 'OkutamaAction.hdf5')
    oa_h5file = h5py.File(fname, 'w')
    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))  # variable length uint8
    float32_dt = h5py.special_dtype(
        vlen=np.dtype('float32'))  # variable length uint8

    vlen_int_dt = h5py.special_dtype(
        vlen=np.dtype(int))  # variable length default int

    train_group = oa_h5file.create_group('train')  
    valid_group = oa_h5file.create_group('valid')  

    # store class list for reference class ids as csv fixed-length numpy string
    oa_h5file.attrs['classes'] = np.string_(str.join(',', classes))

    # store images as variable length uint8 arrays
    dataset_train_images = train_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)

    dataset_valid_images = valid_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)

    # store images as variable length uint8 arrays
    dataset_train_boxes = train_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=float32_dt)

    dataset_valid_boxes = valid_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)

    # Get all the label txt files, there is supposed to have one per video
    label_files = glob.glob(labels_path + '/*.txt')
    assert(len(label_files) > 0)
    print('Total number of lablel files is: ' + str(len(label_files)))
    for lfile in label_files:
        video_file = get_video_file(lfile, videos_path)
        print('Processing video file ' + video_file.split('/')[-1])
        orig_shape, jpg_images = get_video_frames(video_file)
        print('Processing label file ' + lfile.split('/')[-1])
        bboxes_dict = get_bounding_boxes(lfile, orig_shape)
        Xtrain, ytrain = select_images_boxes(jpg_images, bboxes_dict)
        if debug:
            out_dir = video_file.split('/')[-1]
            out_dir = os.path.join('/tmp', out_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for i in range(len(Xtrain)):
                img = draw(Xtrain[i], ytrain[i])
                out_img_path = os.path.join(out_dir, str(i)+'.jpg')
                cv2.imwrite(out_img_path, img)
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xtrain, ytrain, test_size=0.33, random_state=42)   
        print('Adding ' + str(len(Xtrain)) + ' training data')
        add_to_dataset(dataset_train_images, dataset_train_boxes, Xtrain, ytrain)
        print('Adding ' + str(len(Xvalid)) + ' validation data')
        add_to_dataset(dataset_valid_images, dataset_valid_boxes, Xvalid, yvalid)
        
    print('Closing HDF5 file.')
    oa_h5file.close()
    print('Done.')

if __name__ == '__main__':
    _main(parser.parse_args())