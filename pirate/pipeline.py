from .annotate import annotate


import cv2
import re
import time
import sys



bomb = cv2.imread("pirate/boom800.png")
bomb_counter = 1
framerate = "1/2" # 1 frame every 2 seconds

def check_supported_versions():
    if not sys.version_info.major == 3:
        raise SystemError("Only Python3 is supported, cannot continue")
    if not cv2.__version__.startswith('3.'):
        raise SystemError("Only OpenCV version 3 is supported, cannot continue")
    if not re.search("GStreamer:\W+base:\W+YES", cv2.getBuildInformation()):
        raise SystemError("GStreamer support not enabled in OpenCV2, cannot continue")


_livestream = None
def get_livestream(frame=None):
    global _livestream
    if not _livestream:
        assert frame is not None, "Must pass frame to first call of get_livestream"
        framesize = tuple(reversed(frame.shape[:2]))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        print("Found frame dimensions: ", framesize)

        # NOTE NOTE cv2.VideoWriter is overloaded.  You need the version with 5 arguments to specify you want the Gstreamer
        # backend.  https://docs.opencv.org/3.3.0/dd/d9e/classcv_1_1VideoWriter.html
        # Obviously, need to build opencv with Gstreamer support for this to work.

        livestream = cv2.VideoWriter('appsrc ! videorate ! videoconvert !  video/x-raw ,  framerate=%s ' % framerate +
                                        ' ! videoconvert ! videorate   ! textoverlay text="vC" valignment=top halignment=right font-desc="Sans, 36" ' 
                                        ' ! clockoverlay ! x264enc tune=zerolatency ! mpegtsmux ! hlssink max-files=5 async-handling=true target-duration=5 '
                                        ' playlist-location="stills/playlist.m3u8"  location="stills/segment%05d.ts"',
                                        cv2.CAP_GSTREAMER, fourcc, 1.0, framesize)

        livestream = cv2.VideoWriter('appsrc ! videorate ! videoconvert ! video/x-raw, framerate=%s ! timeoverlay ! osxvideosink ' % framerate,
                                     cv2.CAP_GSTREAMER, fourcc, 1.0, framesize)

        _livestream = livestream
    return _livestream



def pipeline(simulate=False, camera_number=0, loop=10):
    """

    :param simulate: Use stills from 'stills/frame%3d.png' instead of camera input
    :param camera_number: Index of the web cam to use, defaults to 0
    :return: None

    This function will start up a gstreamer pipeline reading from the specified webcam (or a directory of stills),
    annotate the frames received, and create a live stream in the current directory (files playlist.m3u8 and segment*.ts)

    """
    check_supported_versions()
    prevState = None
    livestream = None
    then = time.monotonic()


    if not simulate:
        camera_capture = cv2.VideoCapture(
            "avfvideosrc device-index=%d ! videorate ! videoconvert ! videoscale !  video/x-raw, framerate=%s, width=800, height=600   !  appsink" % (
                camera_number, framerate))
    else:
        camera_capture = cv2.VideoCapture(
            'multifilesrc location="stills/frame%%3d.png" start-index=100 stop-index=600 loop=TRUE caps="image/png,framerate=%s' % framerate +
            ' ! pngdec ! videorate ! videoconvert !  video/x-raw ,  framerate=%s   !  appsink' % framerate)

    try:
        for i in range(loop):
            now = time.monotonic()
            print("TIME ELAPSED", now-then)
            then = now

            ret, frame = camera_capture.read()
            if not ret:
                print("Camera not ready, sleeping")
                time.sleep(1)
            else:
                if not livestream:
                    livestream = get_livestream(frame)
                try:
                    frame, prevState = annotate(frame, prevState)
                    livestream.write(frame)
                except ValueError as e:
                    global bomb_counter
                    print("Could not parse frame (bomb%04d.png)" % bomb_counter, e)
                    livestream.write(bomb)
                    cv2.imwrite("stills/bomb%04d.png" % bomb_counter, frame)
                    bomb_counter += 1

    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
        pass


    camera_capture.release()
    livestream.release()
    print("Done")

