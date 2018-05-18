'''capture.py'''
import cv2, sys
import numpy as np

video_capture = cv2.VideoCapture(
   'multifilesrc location="stills/frame%3d.png" start-index=300 stop-index=900 loop=FALSE caps="image/png,framerate=3/1" '
   ' ! pngdec ! videoconvert ! videorate   ! textoverlay text="vA" valignment=top halignment=right font-desc="Sans, 36" !  video/x-raw ,  framerate=1/1   !  appsink')


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
gstreamer_out = cv2.VideoWriter('appsrc ! videorate ! videoconvert !  video/x-raw ,  framerate=1/1  ! osxvideosink ',
                                cv2.CAP_GSTREAMER, fourcc, 1.0, (800, 600))


frameshape = None

for i in range(5):
    ret, frame = video_capture.read()
    if not frameshape:
        frameshape = tuple(reversed(frame.shape[:2]))
        print ("Frame shape", frameshape)
    # for testing, add blue gradient
    frame[:, :, 0] = np.reshape(np.linspace(0,255, 800*600), (600,800))
    gstreamer_out.write(frame)

video_capture.release()
gstreamer_out.release()



