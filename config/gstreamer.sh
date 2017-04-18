#!/bin/bash 


boardcam=${1:-1}
scenecama=${2:-2}
scenecamb=${3:-3}

echo "Playing streams:  boardcam=$boardcam,  scene A=$scenecama,  scene B=$scenecamb"

rm -f playlist.m3u8 segment000*.ts

gst-launch-1.0 \
    videomixer name=mix sink_0::alpha=1  sink_1::ypos=460 sink_1::xpos=620 sink_1::alpha=.5  sink_2::ypos=320 sink_2::xpos=620 !   \
           videoconvert ! x264enc tune=zerolatency  ! \
           mpegtsmux ! hlssink max-files=5 async-handling=true target-duration=5   \
    avfvideosrc device-index=${boardcam} !  video/x-raw , width=800 !  mix.   \
    avfvideosrc device-index=${scenecama} !  videoscale method=0 ! video/x-raw, width=160, height=120 ! mix.  \
    avfvideosrc device-index=${scenecamb} !  video/x-raw, width=160, height=120 ! mix.
