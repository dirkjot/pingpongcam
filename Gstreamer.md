
# Gstreamer notes (started May 2018)

- Test whether playing an mp4 works:
  `GST_DEBUG="*:2" gst-launch-1.0 -v playbin uri="file:///Users/pivotal/Downloads/big_buck_bunny.mp4" `
  first download the clip from https://github.com/mediaelement/mediaelement-files/blob/master/big_buck_bunny.mp4
- general debugging: `GST_DEBUG="*:2"` sets all elements to level 2
- generic sources and sinks for debugging:  `videotestsrc`,  `fakesink`, `filesink location=xx.mpg`
- MacOS specifics:  read from camera  `avfvideosrc device-index=0`,   show video : `gst-launch-1.0 videotestsrc ! osxvideosink`
- MacOS loop
