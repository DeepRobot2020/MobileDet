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

classes = ["person"]
debug = False

parser = argparse.ArgumentParser(
    description='Convert Okutama Action dataset to HDF5.')

parser.add_argument(
    '-p',
    '--path_to_video',
    help='path to Okutama Action videos directory',
    default='~/data/OkutamaAction/Videos')

parser.add_argument(
    '-l',
    '--path_to_labels',
    help='path to Okutama Action label directory',
    default='~/data/OkutamaAction/Labels')

parser.add_argument(
    '-o',
    '--path_to_hdf5',
    help='path to output HDF5',
    default='~/data/OkutamaAction/hdf5')

parser.add_argument(
    '-d',
    '--verify_enabled',
    help='path to classes file, defaults to pascal_classes.txt',
    default=False)

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

def split_image_25(sbboxes, image, xmin, ymin, xmax, ymax,  w=3840, h=2160):
    """ Map the bboxes from original wxh to the new cooridnates 
    ----------
    boxes : numpy array N x 4
        Original central xc, yc, wc, hc
    image_size: tuple
        The orignal size of video frames (w,h) 
    Returns
    -------
    region : dict
        A dictionary (frame_id:np.array representing bboxes of all human objects of a frame id
        Note that the frame_id is not necessary starting from zero or continouous as there might no objects on a frame
        xmin' = ((xc - 0.5 *wc) * w - xmin) / w'
        ymin' = ((yc - 0.5 *hc) * h - xmin) / h'
        xmax' = ((xc + 0.5 *wc) * w - xmin) / w'
        ymax' = ((yc + 0.5 *hc) * h - xmin) / h'
        w'  = xmax - xmin
        h'  = ymax - ymin 
    """
    img_shape = image.shape
    width_prime = int(xmax - xmin)
    height_prime = int(ymax - ymin)
    if sbboxes.shape[0] == 0:
        print('warnning: no objects on this quarter of image')
        return None, None
    xmin_prime = ((sbboxes[:,0] - 0.5 * sbboxes[:,2])*w - xmin) / float(width_prime)
    ymin_prime = ((sbboxes[:,1] - 0.5 * sbboxes[:,3])*h - ymin) / float(height_prime)
    xmax_prime = ((sbboxes[:,0] + 0.5 * sbboxes[:,2])*w - xmin) / float(width_prime)
    ymax_prime = ((sbboxes[:,1] + 0.5 * sbboxes[:,3])*h - ymin) / float(height_prime)
    xc_prime = 0.5 * (xmin_prime + xmax_prime).reshape(-1, 1)
    yc_prime = 0.5 * (ymin_prime + ymax_prime).reshape(-1, 1)
    wc_prime = 1.0 * (xmax_prime - xmin_prime).reshape(-1, 1)
    hc_prime = 1.0 * (ymax_prime - ymin_prime).reshape(-1, 1)
    bboxes_prime = np.concatenate((xc_prime, yc_prime, wc_prime, hc_prime), axis=1)
    image_cropped = image[ymin: ymax, xmin:xmax]
    image_cropped = cv2.resize(image_cropped, (w//2, h//2))
    ret, jpg_img = cv2.imencode('.jpg', image_cropped)
    return jpg_img, bboxes_prime

def split_2160p_to_1080p(image, bboxes):
    """ Split the original 3840x2160 into 5 pictures
    The first 4 images are from spliting the orignal images equally like below
    0 | 1
    ------
    2 | 3
    The last image 4 is the minimum 1920x1080 image convering all the objects
        Parameters
        ----------
        images : The jpeg encoded image
            Lists of all images for a video
        Returns
        -------
    """
    decoded_image = cv2.imdecode(image, 1)
    imwidth, imheight = 3840, 2160
    width, height = 1920, 1080
    # print('Decoded jpeg image into shape: ', decoded_image.shape)
    split_bboxes = []
    # bboxes lies on frame 0
    quarter_boxes = bboxes[np.where((bboxes[:, 0] < 0.5 ) * (bboxes[:, 1] < 0.5))]
    split_bboxes.append(quarter_boxes)
    # bboxes lies on frame 1
    quarter_boxes = bboxes[np.where((bboxes[:, 0] >= 0.5 ) * (bboxes[:, 1] < 0.5))]
    split_bboxes.append(quarter_boxes)
    # bboxes lies on frame 2
    quarter_boxes = bboxes[np.where((bboxes[:, 0] < 0.5 ) * (bboxes[:, 1] >= 0.5))]
    split_bboxes.append(quarter_boxes)
    # bboxes lies on frame 3
    quarter_boxes = bboxes[np.where((bboxes[:, 0] >= 0.5 ) * (bboxes[:, 1] >= 0.5))]
    split_bboxes.append(quarter_boxes)
    # Sanity checking the result 
    assert(sum([item.shape[0] for item in split_bboxes])== bboxes.shape[0])
    # The fifth image is the 1920x1080 image have the same center with original image but cover 
    # all the human objects
    split_bboxes.append(bboxes)
    output_images = []
    output_boxes = []
    for i in range(5):
        sbb = split_bboxes[i]   
        region_xmin = 0 if (i % 2 == 0) else width
        region_ymin = 0 if (i < 2) else height
        if i == 4:
            region_xmin = imwidth // 4
            region_ymin = imheight // 4
        region_xmax = region_xmin +  width
        region_ymax = region_ymin +  height
        # print('Processing quarter region (ymin,xmin), (ymax, xmax): ', (region_ymin, region_xmin), (region_ymax, region_xmax))
        # print('Number of boxes lies in this region: ', sbb.shape[0])
        if sbb.shape[0] == 0:
            continue 
        # found the largest area which can cover all the bboxes of this splited region
        bbxmin = np.min(sbb[:,0] - 0.5 * sbb[:,2])*imwidth
        bbxmax = np.max(sbb[:,0] + 0.5 * sbb[:,2])*imwidth
        bbymin = np.min(sbb[:,1] - 0.5 * sbb[:,3])*imheight
        bbymax = np.max(sbb[:,1] + 0.5 * sbb[:,3])*imheight
        bbxmin = int(min(bbxmin, region_xmin))
        bbymin = int(min(bbymin, region_ymin))
        bbxmax = int(max(bbxmax, region_xmax))
        bbymax = int(max(bbymax, region_ymax))
        quart_img, quart_boxes =  split_image_25(sbb, decoded_image, bbxmin, bbymin, bbxmax, bbymax)
        # print('Max region can cover all the boxes of this region: ', (bbymin, bbxmin), (bbymax, bbxmax))
        output_images.append(quart_img)
        output_boxes.append(quart_boxes)
    return output_images, output_boxes
  

def select_images_boxes(images, bboxes_dict):
    """ There a a lot of reduntent images taken from a video
        Randomly select a portion of images and their corresponding bboxes
    Parameters
    ----------
    images : list of jpeg images
        Lists of all images for a video
    bboxes_dict: dict
        A dict of [frame_id: bounding boxes]
    Returns
    -------
    train_images : list
    train_boxes  : list    
    """
    img_ids_with_bboxes = bboxes_dict.keys()
    total = len(img_ids_with_bboxes)
    img_ids_with_bboxes = sorted(img_ids_with_bboxes)
    keys = [i for i in img_ids_with_bboxes if i < len(images)]
    train_images = [images[i] for i in keys]
    train_boxes = [bboxes_dict[i] for i in keys]
    splited_train_imgs = []
    splited_train_boxes = []
    for i in range(len(train_images)):
        simages, sboxes = split_2160p_to_1080p(train_images[i], train_boxes[i])
        for img in simages:
            splited_train_imgs.append(img)
        for box in sboxes:
            splited_train_boxes.append(box)
    # trick to flatten a list of list
    splited_train_imgs = sum([], splited_train_imgs)
    splited_train_boxes = sum([], splited_train_boxes) 
    return splited_train_imgs, splited_train_boxes


def add_to_dataset(dataset_images, dataset_boxes, images, boxes, start=0):    
    """Add all images and bboxes to given datasets."""
    current_rows = len(boxes)
    total_rows = current_rows + dataset_images.shape[0]
    dataset_images.resize(total_rows, axis=0)
    dataset_boxes.resize(total_rows, axis=0)
    for i in range(current_rows):
        dataset_boxes[start + i] = boxes[i].flatten('C')
        dataset_images[start + i] = images[i].flatten('C')
    return i

def draw_bboxes(image, bboxes):
    """Draw the bounding boxes on raw or jpg images"""
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

def draw_on_images(dataset_images, dataset_boxes, out_dir='/tmp/okutana/'):
    if os.path.exists(out_dir):
        os.rmdir(out_dir)
    os.mkdir(out_dir)
    for i in range(dataset_images.shape[0]):
        img = draw_bboxes(dataset_images[i], dataset_boxes[i])
        out_img_path = os.path.join(out_dir, str(i)+'.jpg')
        cv2.imwrite(out_img_path, img)
    return 

def covert_bboxes_to_voc_style(list_bboxes, h=1080, w=1920):
    """Conver bboxes to voc style bboxes (label, xmin, ymin, xmax, ymax)"""
    voc_bboxes = []
    for bboxes in list_bboxes:
        xmin = w*(bboxes[:,0] - 0.5 * bboxes[:,2]).reshape(-1, 1)
        ymin = h*(bboxes[:,1] - 0.5 * bboxes[:,3]).reshape(-1, 1) 
        xmax = w*(bboxes[:,0] + 0.5 * bboxes[:,2]).reshape(-1, 1)
        ymax = h*(bboxes[:,1] + 0.5 * bboxes[:,3]).reshape(-1, 1)
        corners = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
        corners = np.array(corners, dtype=np.int)
        # Generate label information 
        label = np.zeros((bboxes.shape[0], 1), dtype=np.int)
        label.fill(classes.index('person'))
        voc_bboxes.append(np.concatenate((label, corners), axis=1))
    return voc_bboxes


def _main(args):
    videos_path = os.path.expanduser(args.path_to_video)
    labels_path = os.path.expanduser(args.path_to_labels)
    hdf5_path   = os.path.expanduser(args.path_to_hdf5)
    verify_enabled = args.verify_enabled

    if verify_enabled:
        print("Verifying the HD5 data....")
        if not os.path.exists(hdf5_path):
           print(hdf5_path + " does not exits!")
           return 
        oa = h5py.File(hdf5_path, 'r')
        print("Verifying the training data....")
        draw_on_images(oa['train/images'], oa['train/boxes'])
        print("Verifying the validation data....")
        draw_on_images(oa['valid/images'], oa['valid/boxes'])
        print("Verification is done")
        return
    fname = os.path.join(hdf5_path, 'OkutamaAction.hdf5')
    if os.path.exists(fname):
        os.remove(fname)
    if not os.path.exists(hdf5_path):
        print('Creating ' + hdf5_path)
        os.mkdir(hdf5_path)
    # Create HDF5 dataset structure
    print('Creating HDF5 dataset structure.')

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
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xtrain, ytrain, test_size=0.33, random_state=42)   
        # Convert the bboxes to be the same sytle as voc parse script
        ytrain = covert_bboxes_to_voc_style(ytrain);
        yvalid = covert_bboxes_to_voc_style(yvalid);
        print('Adding ' + str(len(Xtrain)) + ' training data')
        add_to_dataset(dataset_train_images, dataset_train_boxes, Xtrain, ytrain)
        print('Adding ' + str(len(Xvalid)) + ' validation data')
        add_to_dataset(dataset_valid_images, dataset_valid_boxes, Xvalid, yvalid)   
        break 
    print('Closing HDF5 file.')
    oa_h5file.close()
    print('Done.')

if __name__ == '__main__':
    _main(parser.parse_args())