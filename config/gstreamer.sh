#!/bin/bash

rm -f playlist.m3u8 segment000*.ts

gst-launch-1.0 \
    videomixer name=mix sink_0::alpha=1  sink_1::ypos=460 sink_1::xpos=620 sink_1::alpha=.5  !   videoconvert ! x264enc tune=zerolatency  ! \
           mpegtsmux ! hlssink max-files=5 async-handling=true target-duration=5   \
    avfvideosrc device-index=0 !  video/x-raw , width=800 !  mix.   \
    avfvideosrc device-index=1 !  videoscale method=0 ! video/x-raw, width=160 ! mix.
