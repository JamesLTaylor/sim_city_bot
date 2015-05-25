import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyscreenshot

def nothing(x):
    pass

def filteronwhite(img):
    """
    Threshold only on light gray/white
    """
    img2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    img = img.astype(np.float)
    avg = np.mean(img, axis=2)
    ind = np.logical_and(np.abs(img[:,:,0]-avg)<10, np.abs(img[:,:,1]-avg)<10)
    ind = np.logical_and(ind, np.abs(img[:,:,2]-avg)<10)
    ind = np.logical_and(ind, avg>200)
    img2[ind] = 255

def findroads(img):
    """
    night: 15 24 49, 149 128 30  (RGB, HSL) 
    evening: 35 32 33, 227 11 32
    day: 47 56 60, 132 29 50
    """
    lower = (30, 22, 12)
    upper = (62, 58, 49)    
    
    ret = cv2.inRange(img, lower, upper)
    return ret
    
def variablecanny(img):
    """
    395 680 seems to catch the dotted line on the edge during day time (and at night)
    """

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = findroads(img)
    
    cv2.namedWindow('edge detect')
    
    cv2.createTrackbar('threshold','edge detect',128,1000, nothing)
    cv2.createTrackbar('min val','edge detect',100,1000, nothing)
    cv2.createTrackbar('max val','edge detect',200,1000, nothing)
    
    while(1):
        minval = cv2.getTrackbarPos('min val','edge detect')
        maxval = cv2.getTrackbarPos('max val','edge detect')
        edges = cv2.Canny(img1,minval,maxval)
        cv2.imshow('edge detect',edges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    
def realtime():
    cv2.namedWindow('edge detect')
    while(1):
        im = pyscreenshot.grab(bbox = (23,45,982,603))
        in_data = np.asarray(im, dtype=np.uint8)
        img = in_data[:,:,::-1]
        edges = cv2.Canny(img,395,680)
        cv2.imshow('edge detect',edges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    #img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\night1.png")
    #img = img[51:611,15:973]    
    #variablecanny(img)
    realtime()