# pingpongcam
Streaming three webcams through Gstreamer and Nginx so you can see the waiting list and the table


This was created for viewing our ping pong whiteboard and the table in one convenient web page.  It uses three
800x600 webcams to composite an image of the white board with the two views of the table superimposed.  

## installation

This may install some extra stuff, I have not optimized this yet.  I ran this with Gstreamer 1.10 and Nginx 1.10.3:

```
brew install gstreamer
brew install gst-libav 
brew install gst-plugins-base --with-opencv  # just guessing
brew install  gst-plugins-good  --with-jpeg --with-libdv --with-check --with-gdk-pixbuf
brew install gst-plugins-bad --with-opencv --with-rtmpdump
brew install gst-plugins-ugly --with-x264 --with-libshout --with-dirac 
brew install nginx
```

Shallow copy this repo into nginx's html directory then run the script:
```
git clone https://github.com/dirkjot/pingpongcam.git  
cd pingpongcam
npm install
cp * /usr/local/Cellar/nginx/1.10.3/html/
cp -R node_modules /usr/local/Cellar/nginx/1.10.3/html/

# you probably want to set Nginx config after config/nginx.conf
# edit file /usr/local/etc/nginx/nginx.conf

# then run the script
./config/gstreamer.sh

```



Software to install on MacOS:
```bash
brew install ipython
pip3 install jupyter matplotlib scikit-image
brew install pygobject3 --with-python@2 gtk+3
```
