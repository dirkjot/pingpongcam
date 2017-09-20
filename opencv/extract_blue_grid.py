
# coding: utf-8

# In[1]:


LIVENOTEBOOK = False
import cv2
cv2.__version__



# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim 
def imshow(img):  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
if LIVENOTEBOOK:
    get_ipython().magic('matplotlib inline')
    img = cv2.imread("../reference/frame17978.png")
    # test plotting
    imshow(img)


# In[3]:


def tightbluemask(image, clean=True):
    """Returns a mask (grayscale image) which is >0 in the area of the blue grid."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold  = np.array([[103,95,30],[120,240,125]])
    mask = cv2.inRange(hsv, threshold[0,:], threshold[1,:])

    if clean:
        # close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # remove speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask



# In[4]:


# step 1, extract the right kind of blue
if LIVENOTEBOOK:
    image = img
    mask = tightbluemask(img)
    bluewhite = img.copy()
    bluewhite[mask==0,:] = (255,255,255)
    imshow(bluewhite)


# In[5]:


# step 2 :extract raw contours
if LIVENOTEBOOK:
    dst = np.zeros(image.shape, np.uint8)
    dst[:]=(240,240,240)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    print("Found %d outer contours" % len(cnts))
    cv2.drawContours(dst, cnts, -1, (0,255,0), 3)
    plt.imshow(dst)


# In[6]:


# simplify contours, find one that is a rectangle
def find_outer_rect(cnts, img=None):
    """Find the one large rectangle in a list of contours, returns an approximated contour 
    and optionally an image of the contour
    
    If an optional image is passed, this routine will return a filled 3-channel image of the found contour.
    In channel 0, the area inside the outer rect is 0, 255 elsewhere."""
    
    if not img is None:
        dst = np.zeros(img.shape, np.uint8)
        dst[:] = (255,240,240)
    else:
        dst = None
    for c in cnts:
        peri = 0.01 * cv2.arcLength(c, True)  # approximate such that new perimeter is 1% of old one
        approx = cv2.approxPolyDP(c, peri, True)
        print ("Found shape with sides: %d" % len(approx))
        if len(approx) == 4:
                approx = np.reshape(approx,[4,2]) # drop 2nd dimension 
                xsize = approx[:,0].max() - approx[:,0].min()
                ysize = approx[:,1].max() - approx[:,1].min()
                if xsize > 300 and ysize > 150:
                    print ("-- Found: %d x %d" % (xsize, ysize))
                    if not img is None:
                        cv2.drawContours(dst, [approx], -1, (0,255,0), cv2.FILLED)
                    break
                else:
                    print ("-- Rejected rectangle: %d x %d" % (xsize, ysize))
    else:
        raise ValueError("No outer rectangle found")
    return approx, dst

# step 3: get outer rectangle
if LIVENOTEBOOK:
    (outerRectangle, filledMask) = find_outer_rect(cnts, img)
    imshow(filledMask)


# In[7]:


# step 4: remove area outside of outer rectangle from mask (ie set to 0)
if LIVENOTEBOOK:
    mask[filledMask[:,:,0]>0]=0
    plt.imshow(mask)


# In[8]:


def find_inner_rectangles(img):
    # find inner rectangles
    _, cnts, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for i in range(len(cnts)):
        contour = cnts[i]
        _,_,w,h = cv2.boundingRect(contour)
        if w>100 and h>50:
            contours.append(contour)
        else:
            hier[0, i, :] = -2 # mark for deletion

    hier = np.delete(hier, np.where(hier[0,:,-1] == -2)[0], 1)

    assert len(contours) == 5, "there should be five contours: found %d" % len(contours)
    assert hier[0,0,-1] == -1, "first contour should be the outer one, ie it has no parent"
    assert np.all(hier[0,1:5,-1] == 0), "all other contours should have the first contour as their parent"
    hier = np.delete(hier, 0, 1) # remove parent from list of hier
    contours = contours[1:] # remove parent from contours
    contours.sort(key=lambda a: a[:,0,1].min()) # sort contours by minimum y value (top down)
    return contours, hier

# step 5: find inner rectangles
if LIVENOTEBOOK:
    contours, hier = find_inner_rectangles(mask)
    assert len(contours) == 4, "Expected four inner rectangles, found %d " % len(contours)


# In[ ]:





# In[9]:


def get_inner_rect_contents(contours, image):
    """for each inner rectangle in 'contours', extract a standard size rectange with 
    the (masked) contents of this rectangle from 'image'.  
    Returns a list of cropped images and a list of original contours translated to 
    those images. """

    # determine size of largest (topmost) bounding box
    _,_,bbw,bbh = cv2.boundingRect(contours[0])
    boxes = []
    translated_contours = []
    kernel =  np.ones((5,5), np.uint8)
    for contour in contours:
        # compute mask from contour by filling
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        # make mask about 5px smaller on all sides
        mask = cv2.erode(mask, kernel)
        # create white box and copy image in where mask is true
        box = np.zeros_like(image)
        box.fill(255)
        box[mask == 255] = image[mask == 255] 
        # crop to bounding box
        bbx, bby, w, h = cv2.boundingRect(contour)
        assert bbw >= w and bbh >= h, "all boxes should be smaller than the top one"
        box = box[bby:(bby+bbh), bbx:(bbx+bbw), :]
        boxes.append(box)
        translated_contours.append(contour - np.array([bby, bbx]))
    return boxes, translated_contours
    
# step 6: get contents of inner rectangles (boxes)
if LIVENOTEBOOK:
    boxes = get_inner_rect_contents(contours, img)
    imshow(np.concatenate(boxes))


# In[27]:


def get_contents(imagepath):
    img = cv2.imread(imagepath)

    # step 1, extract the right kind of blue
    # image = img
    mask = tightbluemask(img)
    bluewhite = img.copy()
    bluewhite[mask==0,:] = (255,255,255)
    
    # step 2 :extract raw contours
    #dst = np.zeros(img.shape, np.uint8)
    #dst[:]=(240,240,240)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    print("Found %d outer contours" % len(cnts))
    #cv2.drawContours(dst, cnts, -1, (0,255,0), 3)
    
    # step 3: get outer rectangle
    (outerRectangle, filledMask) = find_outer_rect(cnts, img)
    
    # step 4: remove area outside of outer rectangle from mask (ie set to 0)
    mask[filledMask[:,:,0]>0]=0
    
    # step 5: find inner rectangles
    contours, hier = find_inner_rectangles(mask)
    assert len(contours) == 4, "Expected four inner rectangles, found %d " % len(contours)

    # step 6: get contents of inner rectangles (boxes)
    boxes, contours = get_inner_rect_contents(contours, img)
    return boxes, contours
    

