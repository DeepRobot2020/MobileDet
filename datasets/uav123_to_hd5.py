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

classes = ["person"]
debug = False

parser = argparse.ArgumentParser(
    description='Convert Okutama Action dataset to HDF5.')

parser.add_argument(
    '-p',
    '--path_to_uav123',
    help='path to Okutama Action videos directory',
    default='~/data/UAV123/UAV123_10fps')


def find_car_person_folders(seq_path):    
    """ Find folders with car or person pictures
    """
    assert(os.path.exists(seq_path))
    folders = os.listdir(seq_path)
    if len(folders):
        print('No folders in ' + seq_path)
    person_folders = []
    car_folders = []
    for folder in folders:
        person_match = re.search(r'person\d+$', folder)
        car_match = re.search(r'car\d+$', folder)
        if person_match is not None:
            person_folders.append(person_match.group(0))
            print('Adding folder ' + person_match.group(0))
        elif car_match is not None:
            car_folders.append(car_match.group(0))
            print('Adding folder ' + car_match.group(0))
    return person_folders, car_folders

def get_video_file(label_file, video_path):  
    """ Get the video file associated with the label file
    Parameters
    ----------
    video_path : str
        Path to a video directory.
    label_file : str
        The label file containing boundling box information
    Returns
    -------
    video_file : str
        Video file name matching the label file
    """
    assert(os.path.exists(label_file))
    assert(os.path.exists(video_path))
    video_file = None
    video_prefix = label_file.split('/')[-1].split('.txt')[0]
    for file in os.listdir(video_path):
        if fnmatch.fnmatch(file, video_prefix + '.*'):
            video_file = file
    video_file = os.path.join(video_path, video_file)
    assert(os.path.exists(video_file))# this should never happens 
    return video_file


def get_video_frames(video_path):
    """ Convert video file into a list of jpeg images
    Parameters
    ----------
    video_path : str
        Path to a video directory.
    Returns
    -------
    orig_shape : tuple
        Original shape of images (w,h)
    images: list of numpy array
        Encoded jpeg images of all the frames of the video file 
    """
    assert(os.path.exists(video_path))
    frames = []
    count = 0
    vidcap = cv2.VideoCapture(video_path)
    orig_shape = None
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if not orig_shape:
            orig_shape = image.shape[:2][::-1]
        # encode into jpeg and save frame into a list
        ret, image = cv2.imencode('.jpg', image)
        frames.append(image.flatten('C'))
        count += 1
    print("{} images are extacted in {}.".format(count,video_path))
    return orig_shape, frames

def get_bounding_boxes(label_file, orig_size):
    """ Convert the original huam action tracking labels into Object detection labels (bboxes + object type)
    Parameters
    ----------
    label_file : str
        Path to a the label file containing all the original labels.
    orig_size: tuple
        The orignal size of video frames 
    Returns
    -------
    bboxes_dict : dict
        A dictionary (frame_id:np.array representing bboxes of all human objects of a frame id
        Note that the frame_id is not necessary starting from zero or continouous as there might no objects on a frame
    """
    assert(os.path.exists(label_file))
    label_lines = None
    bboxes_dict = {}
    with open(label_file, "r") as labels:
        lines = labels.readlines()
        label_lines = [np.array(line.rstrip().split()[1:9], dtype=np.int) for line in lines]
        label_lines = np.array(label_lines)
        for line in label_lines:
            img_id = int(line[4])
            lost = bool(line[5])
            occluded = bool(line[6])
            generated = bool(line[7])
            if lost or generated or occluded:
                continue
            if img_id not in bboxes_dict:
                bboxes_dict[img_id] = set()           
            bboxes_dict[img_id].add(tuple(line[:4]))
        # Now convert the values to numpy array 
        for key, boxes in bboxes_dict.iteritems():
            boxes = np.array(list(boxes))
            boxes_xy = 0.5 * (boxes[:, 2:4] + boxes[:, 0:2])
            boxes_wh = 1.0*(boxes[:, 2:4] - boxes[:, 0:2])
            boxes_xy = boxes_xy / orig_size
            boxes_wh = boxes_wh / orig_size
            boxes = np.concatenate((boxes_xy, boxes_wh), axis=1)
            bboxes_dict[key] = boxes
    return bboxes_dict

def draw(image, bboxes):
    decoded_image = copy.deepcopy(image)
    if image.shape[0] > 3180:
        decoded_image = cv2.imdecode(image, 1)
    if bboxes is None:
        return decoded_image
    h, w = decoded_image.shape[:2]
    xmin = w*(bboxes[:,0] - 0.5 * bboxes[:,2]).reshape(-1, 1)
    ymin = h*(bboxes[:,1] - 0.5 * bboxes[:,3]).reshape(-1, 1) 
    xmax = w*(bboxes[:,0] + 0.5 * bboxes[:,2]).reshape(-1, 1)
    ymax = h*(bboxes[:,1] + 0.5 * bboxes[:,3]).reshape(-1, 1)
    corners = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
    corners = np.array(corners, dtype=np.int)
    for corner in corners:
        cv2.rectangle(decoded_image, (corner[0], corner[1]),(corner[2], corner[3]), (0,255,0), 10)
    return decoded_image



def add_to_dataset(dataset_images, dataset_boxes, images, boxes, start=0):    
    """Add all images and bboxes to given datasets."""
    current_rows = len(boxes)
    total_rows = current_rows + dataset_images.shape[0]
    dataset_images.resize(total_rows, axis=0)
    dataset_boxes.resize(total_rows, axis=0)
    for i in range(current_rows):
        flatten_boxes = boxes[i].flatten('C')
        dataset_boxes[start + i] = flatten_boxes
        dataset_images[start + i] = images[i].flatten('C')
    return i

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