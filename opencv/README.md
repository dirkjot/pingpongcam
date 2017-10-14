

# Augmented Reality Ping Pong Waiting List

This uses opencv and gstreamer.  Very much a work in progress.

Using Python 3.6 and OpenCV 3 (latest)


## Look into

- smoke encoder/decoder, like motion jpg but with change detection.
- facialblur gstreamer plugin





## On my laptop 

to use Ipython with cv2.imshow, use this:

import sys
sys.path.remove('/usr/local/lib/python2.7/dist-packages')
import cv2
import numpy as np
cv2.startWindowThread()
cv2.namedWindow('prev')


# cv2.imshow('prev', image)
# not waitKey necesssary

