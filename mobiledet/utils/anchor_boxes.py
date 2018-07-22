"""
This script generates anchor boxes for a custom dataset. It will generate:
   + an anchor text file (depending on number of anchors, default = 5)

Example usage:
-------------
python anchor_boxes.py
 --path       /path/to/dataset.hdf5
 --output_dir ./
 --num_anchors  5
"""
import os
import csv
import numpy as np
import cv2, io
import h5py
from PIL import Image
from cfg import *
from box import box_iou, Box
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate custom anchor boxes")

parser.add_argument('-i', '--input_hdf5',
                    help="Path to input hdf5 file", type=str, default=None)

parser.add_argument('-o', '--output_dir',
                    help="Path to output directory", type=str, default='./')

parser.add_argument('-n', '--number_anchors',
                    help="Number of anchors [default = 5]", type=int, default=5)


def hdf5_read_image_boxes(data_images, data_boxes, idx):  
    image = data_images[idx]
    boxes = data_boxes[idx]
    boxes = boxes.reshape((-1, 5))    
    image = Image.open(io.BytesIO(image))
    image_data = np.array(image, dtype=np.float)           
    return np.array(image), np.array(boxes)

def main():
    args = parser.parse_args()
    input_hdf5 = args.input_hdf5
    output_dir = args.output_dir
    number_anchors = args.number_anchors
    h5_data = h5py.File(input_hdf5, 'r')

    # #################################
    # Generate Anchors and Categories #
    # #################################
    train_boxes = np.array(h5_data['train/boxes'])
    train_images = np.array(h5_data['train/images'])

    gt_boxes = []
    n_small = 0
    average_iou = []
    print("Calculating Anchors using K-mean Clustering....")
    if number_anchors in range(2, 16):
        for i in range(train_images.shape[0]):
            img, boxes = hdf5_read_image_boxes(train_images, train_boxes, i)
            img_height, img_width = img.shape[:2]
            orig_size = np.array([img_width, img_height], dtype=np.float)
            orig_size = np.expand_dims(orig_size, axis=0)
            boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
            boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
            # boxes_xy = boxes_xy / orig_size
            # boxes_wh = boxes_wh / orig_size
            boxes = np.concatenate((boxes_xy, boxes_wh), axis=1)
            for box in boxes:
                xc, yc, w, h = box[0], box[1], box[2], box[3]
                aspect_ratio = [IMAGE_W / float(img_width), IMAGE_H / float(img_height)]
                feature_width = float(w) * aspect_ratio[0] / 32
                feature_height = float(h) * aspect_ratio[1] / 32
                if feature_width < 1 and feature_height < 1:
                    n_small += 1
                box = Box(0, 0, feature_width, feature_height)
                gt_boxes.append(box)
        print('Total boxes: ' + str(len(gt_boxes)))
        print('Total small: ' + str(n_small))

        anchors, avg_iou = k_mean_cluster(number_anchors, gt_boxes)
        print("Number of anchors: {:2} | Average IoU:{:-4f}\n\n ".format(number_anchors, avg_iou))
        anchors_file = os.path.join(output_dir, str(number_anchors) + '_' + str(round(avg_iou, 2)) + '_anchors.txt')
        average_iou.append(avg_iou)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(anchors_file, 'w') as f:
            for anchor in anchors:
                f.write("({:5f}, {:5f})\n".format(anchor.w, anchor.h))
    print('average_iou:', average_iou);

def k_mean_cluster(n_anchors, gt_boxes, loss_convergence=1e-6):
    """
    Cluster anchors.
    """
    # initialize random centroids
    centroid_indices = np.random.choice(len(gt_boxes), n_anchors)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(gt_boxes[centroid_index])

    # iterate k-means
    anchors, avg_iou, loss = run_k_mean(n_anchors, gt_boxes, centroids)

    while True:
        anchors, avg_iou, curr_loss = run_k_mean(n_anchors, gt_boxes, anchors)
        if abs(loss - curr_loss) < loss_convergence:
            break
        loss = curr_loss

    return anchors, avg_iou


def run_k_mean(n_anchors, boxes, centroids):
    '''
    Perform K-mean clustering on training ground truth to generate anchors.
    In the paper, authors argues that generating anchors through anchors would improve Recall of the network

    NOTE: Euclidean distance produces larger errors for larger boxes. Therefore, YOLOv2 did not use Euclidean distance
          to measure calculate loss. Instead, it uses the following formula:
          d(box, centroid)= 1 - IoU (box, centroid)

    :param n_anchors:
    :param boxes:
    :param centroids:
    :return:
        new_centroids: set of new anchors
        groups:        wth?
        loss:          compared to current bboxes
    '''
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0

        for i, centroid in enumerate(centroids):
            distance = 1 - box_iou(box, centroid)  # Used in YOLO9000
            if distance < min_distance:
                min_distance = distance
                group_index = i

        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        if len(groups[i]) == 0:
            continue
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    iou = 0
    counter = 0
    for i, anchor in enumerate(new_centroids):
        for gt_box in groups[i]:
            iou += box_iou(gt_box, anchor)
            counter += 1

    avg_iou = iou / counter
    return new_centroids, avg_iou, loss


if __name__ == '__main__':
    main()
