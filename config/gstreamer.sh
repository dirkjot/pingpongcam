#!/bin/bash 
export PATH=$PATH:/usr/local/bin

cd /usr/local/var/www

date > "startup.time.txt"

cleanup () {
  rm -f playlist.m3u8 segment*.ts
  echo "Cleanup done"
}

cleanup

trap cleanup EXIT HUP INT QUIT PIPE TERM

boardcam=${1:-2}
scenecama=${2:-0}
scenecamb=${3:-1}

echo "Playing streams:  boardcam=$boardcam,  scene A=$scenecama,  scene B=$scenecamb"



gst-launch-1.0 \
    videomixer name=mix sink_0::alpha=1  sink_1::ypos=450 sink_1::xpos=620   sink_2::ypos=450 sink_2::xpos=440 !   \
           videoconvert ! clockoverlay halignment=right valignment=top ! \
           x264enc tune=zerolatency  ! \
           mpegtsmux ! hlssink max-files=5 async-handling=true target-duration=5   \
    avfvideosrc device-index=${boardcam} !  videorate ! video/x-raw , width=800, framerate=1/1 !  mix.   \
    avfvideosrc device-index=${scenecama} !  videorate ! videoscale method=0 ! video/x-raw, width=160, height=120, framerate=5/1 ! mix.  \
    avfvideosrc device-index=${scenecamb} !  videorate ! videoscale method=0 ! video/x-raw, width=160, height=120, framerate=5/1 ! mix.
