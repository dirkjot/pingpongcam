
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
def imshow(img):  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
get_ipython().run_line_magic('matplotlib', 'inline')
cv2.__version__


# In[5]:


from annotate import *


# In[78]:


def referenceImages():
    for fn in glob.glob('../reference/frame*.png'):
        if not "-" in fn:
            yield fn


# In[38]:


def imageload(imagename, stepname):
    image = cv2.imread(imagename)
    assert image.shape[0]>100, "height less than 100?"
    return image


# In[40]:


def comparisonload(imagename, stepname):
    compname = imagename.replace(".png", "-" + stepname + ".png")
    comparison = cv2.imread(compname)
    return comparison


# In[21]:


def compreplace(imagename, stepname, newImage):
    compname = imagename.replace(".png", "-" + stepname + ".png")
    cv2.imwrite(compname, newImage)


# In[72]:


def imagecompare(imagename, stepname, fixit, subject):
    comparison = comparisonload(imagename, stepname)
    applesToapples = (subject is not None 
                      and comparison is not None 
                      and np.all(subject.shape == comparison.shape) )
    if applesToapples and np.all(cv2.compare(subject, comparison, cv2.CMP_EQ)):
        print("Pass %s %s" % (stepname, imagename))
    else:
        if fixit:
            compreplace(imagename, stepname, subject)
            print("Replaced after fail %s %s" % (stepname, imagename))
        else:
            print("Failed %s %s" % (stepname, imagename))


# In[95]:


def test_step1(imagename, fixit=False):
    global mask, bluewhite, image, _imagename
    _imagename = imagename
    stepname = 'step1'
    image = imageload(imagename, stepname)
    # 
    mask = tightbluemask(image)
    bluewhite = image.copy()
    bluewhite[mask == 0, :] = (255, 255, 255)
    bluewhite = Box(image=bluewhite)
    #
    imagecompare(imagename, stepname, fixit, bluewhite.image)
            


# In[104]:



test_step1("../reference/frame17978.png", False)


# In[92]:


def test_step2(imagename, fixit=False):
    global image, dst, cnts, mask
    stepname = "step2"
    assert _imagename == imagename
    #
    dst, cnts = extract_raw_contours(image, mask)
    #
    imagecompare(imagename, stepname, fixit, dst)
    assert len(cnts) == 5


# In[107]:


test_step2(_imagename, False)


# In[108]:





# In[71]:





# In[98]:


def extract_raw_contours(image, mask):
    dst = np.zeros(image.shape, np.uint8)
    dst[:] = (240, 240, 240)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    print("Found %d outer contours" % len(cnts))
    cv2.drawContours(dst, cnts, -1, (0, 255, 0), 3)
    return dst, cnts

