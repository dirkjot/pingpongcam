
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


# In[114]:


class TestEnvironment(object):
    "capture the variables we use to communicate between steps"
    def __init__(self, imagename, fixit=False, **kwargs):
        self.__dict__.update(kwargs)
        self.imagename = imagename
        self.fixit = fixit
    def __str__(self):
        return "TestEnv< %r >" % self.__dict__
    


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


# In[170]:


def imagecompare(imagename, stepname, fixit, subject):
    comparison = comparisonload(imagename, stepname)
    if subject is None:
        print ("Empty subject in comparison")
        return False
    if comparison is None and not fixit:
        print ("Empty comparison")
        return False
    if comparison is not None and not np.all(subject.shape == comparison.shape):
        if len(subject.shape) == 2 and len(comparison.shape) == 3:
            comparison = comparison[:,:,0]
            print ("Removed channel 2 and 3 from comparison image")
        else:
            print ("Dimensions not comparable in comparison")
            return False
    if comparison is not None and np.all(cv2.compare(subject, comparison, cv2.CMP_EQ)):
        print("Pass %s %s" % (stepname, imagename))
        return True
    else:
        if fixit:
            compreplace(imagename, stepname, subject)
            print("Replaced after fail %s %s" % (stepname, imagename))
            return True
        else:
            print("Failed %s %s" % (stepname, imagename))
            return False


# In[176]:


def test_step1(testenv, fixit=False):
    stepname = 'step1'
    testenv.image = imageload(testenv.imagename, stepname)
    # 
    testenv.mask, testenv.bluewhite = tightbluemask(testenv.image)
    #
    imagecompare(testenv.imagename, stepname, testenv.fixit or fixit, testenv.bluewhite.image)
    
testenv = TestEnvironment("../reference/frame17978.png")
test_step1(testenv)


# In[177]:


def test_step2(testenv, fixit=False):
    stepname = "step2"
    #
    testenv.dst, testenv.cnts = extract_raw_contours(testenv.image, testenv.mask)
    #
    imagecompare(testenv.imagename, stepname, testenv.fixit or fixit, testenv.dst)
    assert len(testenv.cnts) == 5

test_step2(testenv)


# In[172]:





# In[186]:


def test_step3(testenv, fixit=False):
    stepname = "step3"
    #
    testenv.bluewhite.contour, testenv.mask = find_outer_rect(testenv.cnts, testenv.image, testenv.mask)
    #
    imagecompare(testenv.imagename, stepname, testenv.fixit or fixit, testenv.mask)    

test_step3(testenv)


# In[187]:


def test_step5(testenv):
    stepname = "step5"
    #
    testenv.contours, testenv.hier = find_inner_rectangles(testenv.mask)
    #
    assert len(testenv.contours) == 4

test_step5(testenv)
  


# In[188]:


def test_step6(testenv, fixit=False):
    stepname = "step6"
    #
    testenv.boxes = get_inner_rect_contents(testenv.contours, testenv.image)
    #
    boxedimg = np.concatenate([b.image for b in testenv.boxes])
    imagecompare(testenv.imagename, stepname, (testenv.fixit or fixit), boxedimg)
    
test_step6(testenv)


# In[189]:


def test_step7(testenv, fixit=False):
    stepname = "step7"
    #
    classify_boxes(testenv.boxes)
    #
    boxedimg = np.concatenate([b.contentdetection for b in testenv.boxes])
    imagecompare(testenv.imagename, stepname, (testenv.fixit or fixit), boxedimg)
    
test_step7(testenv)


# In[193]:


def test_step9(testenv, fixit=False):
    stepname = "step9"
    #
    addTimeIndicators(testenv.boxes, None)
    testenv.bluestraight = reconstruct(testenv.boxes, testenv.bluewhite)
    #
    imagecompare(testenv.imagename, stepname, (testenv.fixit or fixit), testenv.bluestraight)
    
test_step9(testenv)
    


# In[202]:


def testAll(fixit=False):
    for name in referenceImages():
        print(name)
        try:
            testenv = TestEnvironment(name, fixit)
            test_step1(testenv)
            test_step2(testenv)
            test_step3(testenv)
            test_step5(testenv)
            test_step6(testenv)
            test_step7(testenv)
            test_step9(testenv)
        except AssertionError:
            print ("Failure caught, moving on", name)
testAll(True)

