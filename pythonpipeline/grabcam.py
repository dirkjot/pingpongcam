'''
grabcam.py

Read a few frames from the webcam and output them to stdout.  Use a bash pipeline to play them back with VLC (see below)
'''
import cv2, sys

video_capture = cv2.VideoCapture(
    "avfvideosrc device-index=0 ! videorate ! videoconvert !  video/x-raw ,  framerate=1/1   !  appsink")




#while True :
for i in range(5):
    ret, frame = video_capture.read()
    # for testing, turn the blue channel up to the max:
    frame[:, :, 0] = 255
    sys.stdout.buffer.write( frame.tostring() )

video_capture.release()

# confirm this with
# rm foo.avi; python3 grabcam.py  | ffmpeg -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 1 -i - foo.avi ; vlc foo.avi


