'''capture.py'''
import cv2, sys

video_capture = cv2.VideoCapture(
    "avfvideosrc device-index=0 ! videorate ! videoconvert !  video/x-raw ,  framerate=1/1   !  appsink")

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

# NOTE NOTE cv2.VideoWriter is overloaded.  You need the version with 5 arguments to specify you want the Gstreamer
# backend.  https://docs.opencv.org/3.3.0/dd/d9e/classcv_1_1VideoWriter.html
# Obviously, need to build opencv with Gstreamer support for this to work.

gstreamer_out = cv2.VideoWriter('appsrc ! videorate ! videoconvert !  video/x-raw ,  framerate=1/1  ! osxvideosink ',
                                cv2.CAP_GSTREAMER, fourcc, 1.0, (1280, 720))



#while True :
for i in range(5):
    ret, frame = video_capture.read()
    # for testing, turn the blue channel up to the max:
    frame[:, :, 0] = 255
    gstreamer_out.write(frame)

video_capture.release()
gstreamer_out.release()



