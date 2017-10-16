# coding: utf-8

"""
      Whiteboard-Pirate
  Because pirates say AR a lot.

  dirk p. janssen, fall 2017
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def tightbluemask(image, clean=True):
    """
    Detect the blue grid that we use to write names in.  Returns a mask (grayscale image) which is >0 in the area
    of the blue grid.

    This routine uses color selection in a HSV space, you may need to tweak this if you have a different setup.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold = np.array([[103, 95, 30], [120, 240, 125]])
    mask = cv2.inRange(hsv, threshold[0, :], threshold[1, :])
    # close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # remove speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# simplify contours, find one that is a rectangle
def find_outer_rect(cnts, img=None):
    """
    Find the one large rectangle in a list of contours, returns an approximated contour
    and optionally an image of the contour.  The one large rectangle is the outside of our blue grid.

    If an optional image is passed, this routine will return a filled 3-channel image of the found contour.
    In channel 0, the area inside the outer rect is 0, 255 elsewhere.
    """

    if not img is None:
        dst = np.zeros(img.shape, np.uint8)
        dst[:] = (255, 240, 240)
    else:
        dst = None
    for c in cnts:
        peri = 0.01 * cv2.arcLength(c, True)  # approximate such that points are never > 0.01*perimeter away
        approx = cv2.approxPolyDP(c, peri, True)
        print ("Found shape with sides: %d" % len(approx))
        if len(approx) == 4:
            approx = np.reshape(approx, [4, 2])  # drop 2nd dimension
            xsize = approx[:, 0].max() - approx[:, 0].min()
            ysize = approx[:, 1].max() - approx[:, 1].min()
            if xsize > 300 and ysize > 150:
                print ("-- Found: %d x %d" % (xsize, ysize))
                if not img is None:
                    cv2.drawContours(dst, [approx], -1, (0, 255, 0), cv2.FILLED)
                break
            else:
                print ("-- Rejected rectangle: %d x %d" % (xsize, ysize))
    else:
        raise ValueError("No outer rectangle found")
    return approx, dst



def find_inner_rectangles(img):
    """
    Find the inner four boxes (rectangles) in our grid.  Returns a list of four contours and a hierarchy structure
    (see opencv findContours).  This throws an assertion error if (parts of) the grid cannot be found.
    """
    _, cnts, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for i in range(len(cnts)):
        contour = cnts[i]
        _, _, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 50:
            contours.append(contour)
        else:
            hier[0, i, -1] = -2  # mark for deletion

    # remove smaller contours that were marked for deletion
    hier = np.delete(hier, np.where(hier[0, :, -1] == -2)[0], 1)

    assert len(contours) == 5, "there should be five contours: found %d" % len(contours)
    assert hier[0, 0, -1] == -1, "first contour should be the outer one, ie it has no parent"
    assert np.all(hier[0, 1:5, -1] == 0), "all other contours should have the first contour as their parent"
    hier = np.delete(hier, 0, 1)  # remove parent from list of hier
    contours = contours[1:]  # remove parent from contours
    contours.sort(key=lambda a: a[:, 0, 1].min())  # sort contours by minimum y value (top down)
    return contours, hier



# Inspired by/copied from http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def order_points(pts):
    """
    Given a contour of four points, order them in top-left, top-right, bottom-right, bottom-left order
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, wh):
    """
    Given a source image and a list of 4 contour points, return a perspective warped image
    of dimension WxH
    """
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [wh[0] - 1, 0],
        [wh[0] - 1, wh[1] - 1],
        [0, wh[1] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, wh)
    return warped


class Box(object):
    """
    Data class for image of the four squares (boxes) on our white board. This is basically a bunch, ie a
    wrapper around whatever attributes we decide to throw in here.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return "Box< %r >" % self.__dict__



# TODO this all assumes the first box is the largest one, which we don't test or ensure

def get_inner_rect_contents(contours, image):
    """
    For each inner rectangle in 'contours', extract a standard size rectangle with
    the (masked) contents of this rectangle from 'image'.  Returns a list of cropped images in Box data structures.
    """
    boxes = []
    wh = (475, 100)  # scaled from what we have
    kernel = np.ones((5, 5), np.uint8)
    for contour in contours:
        # compute mask from contour by filling
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        # make mask about 5px smaller on all sides
        mask = cv2.erode(mask, kernel)
        peri = cv2.arcLength(contour, True)  # approximate such that points are never > 0.05*perimeter away
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        # create white box and copy image in where mask is true
        box = Box(image=np.zeros_like(image), perimeter=peri, contour=contour, approx=approx)
        box.image.fill(255)
        box.image[mask == 255] = image[mask == 255]
        # find simpler contour
        print ("Found shape with sides: %d" % len(approx))
        box.image = four_point_transform(box.image, approx[:, 0, :], wh)
        boxes.append(box)
    return boxes


class Classify_Boxes(object):
    modulepath = os.path.split(__file__)[0] + '/'
    namefound = cv2.cvtColor(cv2.imread(modulepath + 'namefound50.png'), cv2.COLOR_BGR2GRAY)
    namemissing = cv2.cvtColor(cv2.imread(modulepath + 'namemissing50.png'), cv2.COLOR_BGR2GRAY)

    def run(self, boxes):
        for b in boxes:
            bwimg = cv2.cvtColor(b.image[10:(b.image.shape[0]-10), 30:(b.image.shape[1]-30)], cv2.COLOR_BGR2GRAY) ## TODO remove crop
            blur = cv2.GaussianBlur(bwimg, (5, 5), 0)
            bg = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            dst = np.dstack([bg, bg, bg])
            image = cv2.Canny(bg, 10, 200)
            cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            cnts = [c for c in cnts if cv2.contourArea(c) >= 20]
            b.hascontent = len(cnts) > 1
            if b.hascontent:
                color = (255, 0, 0)
                icon = self.namefound
            else:
                color = (0, 0, 255)
                icon = self.namemissing
            cv2.drawContours(dst, cnts, -1, color, 3)
            dst[20:66, -60:-10, :] = np.dstack([icon, icon, icon])
            b.contentdetection = dst
        return boxes

classify_boxes = Classify_Boxes().run



def pirate(imagepath):
    """
    Return a reconstructed, perspective warped and annotated version of the blue grid.
    """
    img = cv2.imread(imagepath)

    # step 1, extract the right kind of blue
    # image = img
    mask = tightbluemask(img)
    bluewhite = img.copy()
    bluewhite[mask == 0, :] = (255, 255, 255)

    # step 2 :extract raw contours
    dst = np.zeros(img.shape, np.uint8)
    dst[:] = (240, 240, 240)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    print("Found %d outer contours" % len(cnts))
    cv2.drawContours(dst, cnts, -1, (0, 255, 0), 3)

    # step 3: get outer rectangle
    (outerRectangle, filledMask) = find_outer_rect(cnts, img)

    # step 4: remove area outside of outer rectangle from mask (ie set to 0)
    mask[filledMask[:, :, 0] > 0] = 0

    # step 5: find inner rectangles
    contours, hier = find_inner_rectangles(mask)
    assert len(contours) == 4, "Expected four inner rectangles, found %d " % len(contours)

    # step 6: get contents of inner rectangles (boxes)
    boxes = get_inner_rect_contents(contours, img)

    # step 7: classify
    classify_boxes(boxes)

    # step 8: reconstruct image
    bluestraight = four_point_transform(bluewhite, outerRectangle, (480, 550))
    bluestraight[30:(30 + 80), 26:(26 + 415)] = boxes[0].contentdetection
    bluestraight[180:(180 + 80), 26:(26 + 415)] = boxes[1].contentdetection
    bluestraight[290:(290 + 80), 26:(26 + 415)] = boxes[2].contentdetection
    bluestraight[430:(430 + 80), 26:(26 + 415)] = boxes[3].contentdetection

    return bluestraight







