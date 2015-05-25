import win32api, win32con
import time
import math
import numpy as np
import pyscreenshot
import cv2
import datetime

def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(.1)
    
def dragmove(x, y, nx, ny, slow = False):    
    length = math.sqrt((x-nx)**2 + (y-ny)**2)
    print(slow)
    if slow:
        steps = int(math.ceil(length/2))
    else:
        steps = int(math.ceil(length/10))
    xs = (np.round(np.linspace(x, nx, steps))).astype(int)
    ys = (np.round(np.linspace(y, ny, steps))).astype(int)   
    
    for i in range(len(xs)-1):
        (cx, cy) = win32api.GetCursorPos()
        
        # was seeing some funny behaviour where the mouse does not move the exact
        # number of pixels, so this code keeps correcting for where it is meant to be
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, xs[i+1]-cx,ys[i+1]-cy,0,0)
        if slow:
            time.sleep(.01)
        else:
            time.sleep(.01)
        
        
  
    
    
def drag(x, y, nx, ny, slow=False):
    x = int(x)
    y = int(y)
    win32api.SetCursorPos((x,y))    
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(.2)
    
    if hasattr(nx, '__iter__'):
        for i in range(nx):
            if i==0:
                dragmove(x, y, nx[0], ny[0], slow=slow)
            else:
                dragmove(nx[i-1], ny[i-1], nx[i], nx[i], slow=slow)
    else:
        dragmove(x, y, nx, ny, slow=slow)
        
    time.sleep(.2)
    (cx, cy) = win32api.GetCursorPos()
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,cx,cy,0,0)
    print("moving from " + str((x,y)) + " to " + str((nx, ny)) + ".  Ended at " + str((cx, cy)))    
    time.sleep(.2) # make sure there is a delay before the next mouse down takes place
    
    #win32api.SetCursorPos((nx,ny))  


def moveslow():
    win32api.SetCursorPos((250,250))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,250,250,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,250,250,0,0) 
    time.sleep(.1)
    
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,250,250,0,0)
    for i in range(20):   
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 5,5,0,0)
        win32api.SetCursorPos((250+i*5,250+i*5))
        time.sleep(.05)
        
    time.sleep(.2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,350,350,0,0)
   
    
def movefast():
    win32api.SetCursorPos((250,250))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,250,250,0,0)
    time.sleep(.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,250,250,0,0) 
    time.sleep(.1)
    
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,250,250,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 100,100,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,350,350,0,0)
    win32api.SetCursorPos((350,350))
    
def findtemplate(img):
    template = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\helmettick.png")
    h, w, ignore = template.shape
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    
    x = loc[1][0]
    y = loc[0][0]
    img = np.array(img)
    
    """cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 2)    
    
    cv2.imshow('a',img)
    cv2.waitKey()
    cv2.destroyAllWindows() """   
    
    return (x+w/2, y+h/2)
  
    
    
if __name__ == "__main__":
    (cx, cy) = win32api.GetCursorPos()
    win32api.SetCursorPos((cx,cy))
    time.sleep(0.5)
    click(cx,cy)
    
    im = pyscreenshot.grab(bbox = (23,45,982,603))
    in_data = np.asarray(im, dtype=np.uint8)
    img = in_data[:,:,::-1]
    
    (nx, ny) = findtemplate(img)
    win32api.SetCursorPos((nx+23,ny+45))
    time.sleep(0.5)
    click(nx+23,ny+45)
    time.sleep(1)
    drag
    #drag(cx,cy,cx+80,cy+120)
    # win32api.SetCursorPos((cx+80,cy+120))
    
    #time.sleep(0.5)
    #win32api.SetCursorPos((250,317))
  