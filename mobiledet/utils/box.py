import numpy as np

class Box(object):
    def __init__(self, xc, yc, w, h, c=None, classes=None):
        self.x = xc
        self.y = yc
        self.w = w
        self.h = h

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score
        self.label = -1
        self.score = -1

    def to_array(self):
        return np.array([self.x, self.y, self.w, self.h, int(self.get_label())])

    def to_opencv_format(self):
        """
        Convert bounding box to OpenCV format
        :return:  [[(x1, y1), (x2, y2)]] (numpy int)
        """
        x1 = int(self.x - self.w/2)
        y1 = int(self.y - self.h/2)
        x2 = int(self.x + self.w/2)
        y2 = int(self.y + self.h/2)
        opencv_box = [[(x1, y1), (x2, y2)]]

        return opencv_box

    def to_relative_size(self, img_size=(1280, 960)):
        """
        
        :param img_size: 
        :return: 
        """
        width, height = img_size
        xc = self.x/(1. * width)
        yc = self.y/(1. * height)
        w  = self.w/(1. * width)
        h  = self.h/(1. * height)
        return xc, yc, w, h

    def to_abs_size(self, img_size=(1280, 960)):
        """

        :param img_size: 
        :return: 
        """
        width, height = img_size

        if self.x > 1.0:  # Make sure current box is in relative format
            return self.x, self.y, self.w, self.h

        self.x = self.x   * width
        self.y = self.y   * height
        self.w  = self.w   * width
        self.h  = self.h   * height
        return self.x, self.y, self.w, self.h



    def __str__(self):
        return "{}, {}, {}, {}".format(self.x, self.y, self.w, self.h)

    def __repr__(self):
        return str(self)


def box_iou(b1, b2):
    intersect = box_intersection(b1, b2)
    union = box_union(b1, b2)
    iou = float(intersect / union)
    return iou


def box_intersection(b1, b2):
    w = overlap(b1.x, b1.w, b2.x, b2.w)
    h = overlap(b1.x, b1.h, b2.x, b2.h)
    if (w < 0) or (h < 0): return 0
    area = w * h
    return area


def overlap(x1, w1, x2, w2):
    l1 = x1 - (w1 / 2.)
    l2 = x2 - (w2 / 2.)
    r1 = x1 + (w1 / 2.)
    r2 = x2 + (w2 / 2.)
    left = l1 if l1 >= l2 else l2
    right = r1 if r1 <= r2 else r2
    return right - left


def box_union(b1, b2):
    intersect = box_intersection(b1, b2)
    union = (b1.w * b1.h) + (b2.w * b2.h) - intersect
    return union


def convert_bbox(x1, y1, x2, y2):
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    xc = float(x1) + w / 2.
    yc = float(y1) + h / 2.
    return xc, yc, w, h


def scale_rel_box(img_size, box):
    """
    Scale bounding box relative to image size
    """
    width, height, _ = img_size
    dw = 1. / width
    dh = 1. / height
    xc = box.x * dw
    yc = box.y * dh
    w  = box.w * dw
    h  = box.h * dh
    return xc, yc, w, h

