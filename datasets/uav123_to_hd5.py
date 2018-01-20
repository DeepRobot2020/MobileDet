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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

classes = ['person', 'car']

parser = argparse.ArgumentParser(
    description='Convert UAV123 dataset to HDF5.')

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


def find_car_person_folders(seq_path):    
    """ Find folders with car or person pictures
    """
    assert(os.path.exists(seq_path))
    folders = os.listdir(seq_path)
    if len(folders) == 0:
        print('No folders in ' + seq_path)
        return
    car_person_folders = []
    group_person_folders = []
    for folder in folders:
        person_match = re.search(r'person\d+$', folder)
        car_match = re.search(r'car\d+$', folder)
        # group_match = re.search(r'group\d+$', folder)
        if person_match is not None:
            car_person_folders.append(person_match.group(0))
            print('Adding folder ' + person_match.group(0))
        elif car_match is not None:
            car_person_folders.append(car_match.group(0))
            print('Adding folder ' + car_match.group(0))
        # elif group_match is not None:
        #     group_person_folders.append(group_match.group(0))
        #     print('Adding folder ' + group_match.group(0))
    car_person_folders = sorted(car_person_folders)
    # group_person_folders = sorted(group_person_folders)
    return car_person_folders

def find_car_person_anns(ann_path):
    """Find annoation files related to car and persons
    """
    car_re = r'car\d+(_\d+)?.txt'
    person_re = r'person\d+(_\d+)?.txt'
    # group_re = r'group\d+(_\d+)?.txt'
    car_person_ann = []
    group_ann = []
    for ann in os.listdir(ann_path):
        car_match = re.match(car_re, ann)
        person_match = re.match(person_re, ann)
        # group_match = re.match(group_re, ann)
        if car_match:
            car_person_ann.append(car_match.group(0))
            print('Adding Car anno ' + car_match.group(0))
        elif person_match:
            car_person_ann.append(person_match.group(0))
            print('Adding Person anno ' + person_match.group(0))
        # elif group_match:
        #     group_ann.append(group_match.group(0))
        #     print('Adding Person anno ' + group_match.group(0))
    car_person_ann = sorted(car_person_ann)
    # group_ann = sorted(group_ann)
    return car_person_ann

def match_dataseq_anno(seq_path, ann_path):
    """ Find the label file for each video folder. Note that for some videos there are multiple 
        Note: There is a new line missing in car7 label file and it has been manually fixed
    """
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
        parsed_anns = np.array(parsed_anns)
        # covert the (xmin, ymin, w, h) to (xmin,ymin, xmax,ymax)
        parsed_anns[:,2] = parsed_anns[:,0] + parsed_anns[:,2]
        parsed_anns[:,3] = parsed_anns[:,1] + parsed_anns[:,3]
        label = np.zeros((parsed_anns.shape[0], 1), dtype=np.int)
        if 'car' in folder.lower():
            label.fill(classes.index('car'))
        elif 'person' in folder.lower():
            label.fill(classes.index('person'))
        else:
            print('warning: unknown label')
            continue
        # final label is (class, xmin,ymin, xmax,ymax), the same as voc parse script
        parsed_anns = np.concatenate((label, parsed_anns), axis=1)
        print('Total annotations: ' + str(len(parsed_anns)))
        if len(parsed_anns) != len(imgs):
            print('warning: image and anno have different size. Turncating') 
            num = min(len(parsed_anns),len(imgs))
            parsed_anns = parsed_anns[:num,:]
            imgs = imgs[:num]
        annotations.append(parsed_anns)
        object_images.append(imgs)
    return object_images, annotations, car_person_folders


def select_object_detection_images(list_videos, list_annos, list_folders, selection_folder = '~/data/UAV123/UAV123_selected'):
    """ The original UAV123 dataset was for object tracking purpose and usually only one of 
        the object is labelled. To train an object detection network, we need remove the images where 
        not all the objects are labelled. Otherwise, it might take longer time for model to converge. 
    """
    selection_folder = os.path.expanduser(selection_folder)
    assert(os.path.exists(selection_folder))
    assert(len(list_videos) == len(list_annos))
    assert(len(list_videos) > 0)
    out_images = []
    out_annos = []
    out_folders = []
    selected_video_name = os.listdir(selection_folder)
    selected_video_name = [f.split('.')[0] for f in selected_video_name]
    selected_video_name = sorted(selected_video_name)

    selected_images = dict.fromkeys(selected_video_name)
    for video_file in selected_video_name:
        label_name = video_file + '.txt'
        label_name = os.path.join(selection_folder, label_name)
        with open(label_name, 'r') as f:
            img_idxs = [line.rstrip() for line in f]
        selected_images[video_file] = img_idxs
    # import pdb; pdb.set_trace()

    for i in range(len(list_folders)):
        if list_folders[i] not in selected_images:
            continue
        raw_video = list_videos[i] # this in fact is a list of image
        raw_anns = list_annos[i]
        video = []
        anns =[]
        for j in range(len(raw_video)):
            if raw_anns[j][2] * raw_anns[j][3] < 16: # w*h > 16 pixes
                continue
            image_name = raw_video[j].split('/')[-1]
            if image_name not in selected_images[list_folders[i]]:
                continue
            video.append(raw_video[j])
            anns.append(raw_anns[j])
        print('Adding folder', list_folders[i], len(video))
        out_images.append(video)
        out_annos.append(np.array(anns))
        out_folders.append(list_folders[i])
    return out_images, out_annos, out_folders

    

def balance_video_annos(videos, annos, max_allowed_sample=100):
    """ The number if images for each videos has large variance and this might
        cause the deep learning model overfit on certain type of images. To overcome 
        the issue of using video data for image detection we only allow max_allowed_sample
        of images for each video and mark them as balanced_video and balanced_annos
        The remaining one will be marked as unbalanced_video and annos and can be used for validation purpose
    Parameters
    ----------
    videos : list
        List of list of images. 
    annos : list
        List of list of images
    max_allowed_sample: int
        The maximum allow image samples for each video clip
    Returns
    -------
    balance_videos : List
        List of list of images
    balance_videos : List
        List of list of annos.
    unbalance_videos : List
        List of list of images.
    unbalance_videos : List
        List of list of anno.
    """
    assert(len(videos) == len(annos))
    balance_images   = []
    balance_labels   = []
    unbalance_images = []
    unbalance_labels = [] 
    min_samples = min([len(video) for video in videos])
    max_allowed_sample = min(min_samples, max_allowed_sample)
    max_allowed_sample = max(max_allowed_sample, 100)
    for i in range(len(videos)):
        images = np.array(videos[i])
        labels = np.array(annos[i])  
        # TODO: for better performance, sample images with fixed interval 
        if len(images) > max_allowed_sample:
            rnd_idxs = sorted(np.random.choice(len(images), max_allowed_sample, replace=False))
            balance_images.extend(images[rnd_idxs].tolist())
            balance_labels.extend(labels[rnd_idxs].tolist())
            unbalance_images.extend([images[j] for j in range(len(images)) if not (j in rnd_idxs)])
            unbalance_labels.extend([labels[j] for j in range(len(labels)) if not (j in rnd_idxs)])
        else:
            balance_images.extend(images)
            balance_labels.extend(labels)
    return balance_images, np.array(balance_labels), unbalance_images, np.array(unbalance_labels)

def get_image_for_id(images, image_id):
    assert(image_id < len(images))
    fname = images[image_id]
    with open(fname, 'rb') as in_file:
        data = in_file.read()
    # Use of encoding based on: https://github.com/h5py/h5py/issues/745
    return np.fromstring(data, dtype='uint8')

def add_to_dataset(dataset_images, dataset_boxes, images, bboxes, start=0):
    """ Store image and bboxes data into dataset
    """
    current_rows = len(bboxes)
    total_rows = current_rows + dataset_images.shape[0]
    dataset_images.resize(total_rows, axis=0)
    dataset_boxes.resize(total_rows, axis=0)
    for i in range(min(len(images), len(bboxes))):
        dataset_images[start + i] = get_image_for_id(images, i)
        dataset_boxes[start + i] = bboxes[i].flatten('C')
    return i

def draw_on_image_files(images, bboxes, name_hint='debug'):
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

def draw_on_images(dataset_images, dataset_boxes, out_dir='/tmp/uav123/'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(dataset_images.shape[0]):
        boxes = np.array(dataset_boxes[i]).reshape(-1, 5)
        img = draw_bboxes(dataset_images[i], boxes)
        out_img_path = os.path.join(out_dir, str(i)+'.jpg')
        cv2.imwrite(out_img_path, img)
    return 

def _main(args):
    verify_only = args.verify_only
    seq_path = os.path.expanduser(args.seq_path)
    anno_path = os.path.expanduser(args.anno_path)
    hdf5_path   = os.path.expanduser(args.hdf5_path)
    uav123_selected = args.selected

    assert(os.path.exists(seq_path))
    assert(os.path.exists(anno_path))
    if verify_only:
        hdf5_path = os.path.join(hdf5_path, 'UAV123.hdf5')
        print("Verifying the HD5 data....")
        if not os.path.exists(hdf5_path):
           print(hdf5_path + " does not exits!")
           return 
        uav123 = h5py.File(hdf5_path, 'r')
        print("Verifying the training data....")
        draw_on_images(uav123['train/images'], uav123['train/boxes'])
        print("Verifying the validation data....")
        draw_on_images(uav123['valid/images'], uav123['valid/boxes'])
        print("Verification is done")
        return
    list_videos, list_annos, list_folders = match_dataseq_anno(seq_path, anno_path)
    print(len(list_videos), len(list_annos))
    videos, annos, folders = select_object_detection_images(
        list_videos, list_annos, list_folders, uav123_selected)

    print('Total number of images: '+ str(sum([len(i) for i in videos])))
    balance_images, balance_annos, unbalance_images, unbalance_annos = balance_video_annos(videos, annos)
    Xtrain, ytrain = shuffle(balance_images, balance_annos, random_state=0)
    Xvalid, Xtest, yvalid, ytest = train_test_split(unbalance_images, unbalance_annos, test_size=0.4, random_state=42)

    # We will use balance_images, balance_annos as train data
    # and select a portion from unbalance_images, unbalance_annos to use as validation and test data 
    if not os.path.exists(hdf5_path):
        print('Creating ' + hdf5_path)
        os.mkdir(hdf5_path)
    # Create HDF5 dataset structure
    print('Creating HDF5 dataset structure.')
    fname = os.path.join(hdf5_path, 'UAV123.hdf5')

    if os.path.exists(fname):
        print('Removing old HDF5')
        os.remove(fname)
    uav123_h5file = h5py.File(fname, 'w')
    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))  # variable length uint8
    uint32_dt = h5py.special_dtype(
        vlen=np.dtype('uint32'))  # variable length uint8
    vlen_int_dt = h5py.special_dtype(
        vlen=np.dtype(int))  # variable length default int
    train_group = uav123_h5file.create_group('train')  
    valid_group = uav123_h5file.create_group('valid')  
    test_group = uav123_h5file.create_group('test')  
    

    uav123_h5file.attrs['classes'] = np.string_(str.join(',', classes))

    # store images as variable length uint8 arrays
    dataset_train_images = train_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)

    dataset_valid_images = valid_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)

    dataset_test_images = test_group.create_dataset(
        'images', shape=(0, ), maxshape=(None, ), dtype=uint8_dt)


    dataset_train_boxes = train_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=uint32_dt)

    dataset_valid_boxes = valid_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=uint32_dt)

    dataset_test_boxes = test_group.create_dataset(
        'boxes', shape=(0, ), maxshape=(None, ), dtype=uint32_dt)

    print('Adding ' + str(len(Xtrain)) + ' training data')
    add_to_dataset(dataset_train_images, dataset_train_boxes, Xtrain, ytrain, start=0)
    print('Adding ' + str(len(Xvalid)) + ' validation data')
    add_to_dataset(dataset_valid_images, dataset_valid_boxes, Xvalid, yvalid, start=0) 
    print('Adding ' + str(len(Xvalid)) + ' test data')
    add_to_dataset(dataset_test_images, dataset_test_boxes, Xtest, ytest, start=0) 

    print('Closing HDF5 file.')
    uav123_h5file.close()
    print('Done.')

if __name__ == '__main__':
    _main(parser.parse_args())
