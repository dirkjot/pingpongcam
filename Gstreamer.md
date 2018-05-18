
# Gstreamer notes (started May 2018)


#### General 
- Test whether playing an mp4 works:
  `GST_DEBUG="*:2" gst-launch-1.0 -v playbin uri="file:///Users/pivotal/Downloads/big_buck_bunny.mp4" `
  first download the clip from [github](https://github.com/mediaelement/mediaelement-files/blob/master/big_buck_bunny.mp4)
- general debugging: `GST_DEBUG="*:2"` sets debug for all elements to level 2
- generic sources and sinks for debugging:  `videotestsrc`,  `fakesink`, `filesink location=xx.mpg`
- MacOS specifics:  read from camera  `avfvideosrc device-index=0`,   show video : `gst-launch-1.0 videotestsrc ! osxvideosink`
- You can read a sequential stream of images with multifilesrc, see below
- If stuff doesn't work, use a `fakesink` and start removing 



#### Gstreamer example: Playing from a directory with images

This assumes you have png stills in `frame300.png` to `frame400.png`:
```
gst-launch-1.0 multifilesrc location="frame%3d.png" start-index=300 stop-index=400 caps="image/png,framerate=30/1" ! pngdec ! videoconvert ! videorate   ! osxvideosink
```


#### Gstreamer: Creating a local HTTP stream from png files

**First**, create a `stills` directory and populate it with png images.  (if you use a different directory name or path, update `indexlocal.html`).

**Second**, run this command from within that directory

```bash
gst-launch-1.0 multifilesrc location="frame%3d.png" start-index=300 stop-index=900 loop=TRUE caps="image/png,framerate=3/1" \
   ! pngdec ! videoconvert ! videorate   ! textoverlay text="vA" valignment=top halignment=right font-desc="Sans, 36"  \
   ! timeoverlay ! x264enc tune=zerolatency ! mpegtsmux ! hlssink max-files=5 async-handling=true target-duration=5
```

1. Read the png images still300.png to still900.png, send them out with 3 frames per second, and keep looping that
1. decode png; convert format and framerate
1. Add a `vA` to the top right; add the current time in the video to the top left
1. encode with x264, with low latency profile
1. mux into mpeg container
1. send to hlssink

This will create a playlist (`m3u8`) and various `segment*.ts` files in your current directory. 

**Third**, run a python webserver: `python3 -m http.server`

**Finally**, open Chrome (not Safari) to `localhost:8000:indexlocal.html`.  This file will send the hls viewer to the
web browser, than point it to the `stills/playlist.m3u8` file.  If the browser cannot find hlsjs, try `npm install`. 



#### Confirm GStreamer support in OpenCV

```
$ python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -A6 -i gstreamer

    GStreamer:
      base:                      YES (ver 1.14.0)
      video:                     YES (ver 1.14.0)
      app:                       YES (ver 1.14.0)
      riff:                      YES (ver 1.14.0)
      pbutils:                   YES (ver 1.14.0)
```
