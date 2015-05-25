"""
* get screen coordinates

* move to find left and and bottom dashed line

* move left line to right of screen to get top vanishing point

* move bottom line to top of screen to get left vanishing point

* use intersection between dashes and vanishing points to get the screen grid structure
"""

import contour
import personal.robot as robot
import perspective
import win32api, win32con
import time
import math
import numpy as np
import pyscreenshot
import cv2


screenpos = (0,0,0,0)

def getscreen():
    return (7, 30, 1356, 872)
    sys.stdin.read(1)
    (x1, y1) = win32api.GetCursorPos()
    sys.stdin.read(1)
    (x2, y2) = win32api.GetCursorPos()
    
    print((x1,y1,x2,y2))
    return (x1,y1,x2,y2)
    
def screenshot():
    im = pyscreenshot.grab(bbox = screenpos)
    in_data = np.asarray(im, dtype=np.uint8)
    img = np.array(in_data[:,:,::-1])
    return img
    
def leftmost(contours):
    x = 10000
    y = 10000
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if contours[i][j][0][0] < x:
                x = contours[i][j][0][0]
                y = contours[i][j][0][1]
            
    return (x, y)
    
def rightmost(contours):
    x = -10000
    y = -10000
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if contours[i][j][0][0] > x:
                x = contours[i][j][0][0]
                y = contours[i][j][0][1]
            
    return (x, y)    
    
    
def findleftdash():
    img = screenshot()
    lines = contour.get_dashed(img)
    print((bool(lines[0]),bool(lines[1])))
    """
    # move right 4 times then left once so we don't pick up the right edge by mistake
    for i in range(4):
        robot.drag(screenpos[2] - 100, screenpos[3]-20, screenpos[0] + 100, screenpos[3]-20)
    robot.drag(screenpos[0] + 100, screenpos[3]-20, screenpos[2] - 100, screenpos[3]-20)
    # move up just in case we are in the ocean
    robot.drag(screenpos[0] + 100, screenpos[1]+200, screenpos[0] + 100, screenpos[3]-100)
    
    img = screenshot()
    lines = contour.get_dashed(img)
    """
    
    # find the left edge
    count = 0
    while not lines[0] and count<=6:
        count = count + 1
        # if we find a horizontal edge, move along that
        if lines[1]:
            (x, y) = leftmost(lines[1]['contours'])
            robot.drag(x + screenpos[0], y + screenpos[1], screenpos[2] - 100, screenpos[3]-100)
        else:
            robot.drag(screenpos[0] + 100, screenpos[3]-200, screenpos[2] - 100, screenpos[3]-20)
        img = screenshot()
        lines = contour.get_dashed(img)
        print((bool(lines[0]),bool(lines[1])))
        
    # put the left edge right on the edge
    (x, y) = leftmost(lines[0]['contours'])
    robot.drag(x + screenpos[0], y + screenpos[1], screenpos[0] + 50, screenpos[3]-50)
    img = screenshot()
    lines = contour.get_dashed(img)
    
    # slide the screen up to find the bottom edge
    count = 0
    while (not lines[1] or (lines[1] and len(lines[1]['contours'])<8)) and count<=25: # we don't have the bottom yet
        count = count+1
        if not lines[0]:
            cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\noleft.png", img)
        (lx, ly) = leftmost(lines[0]['contours'])
        (rx, ry) = rightmost(lines[0]['contours'])
        print((lx, ly))
        print((rx, ry))
        nx = lx + 60
        ny = ly - 60 * (float(ly)-ry)/(rx-lx)
        robot.drag(lx + screenpos[0], ly + screenpos[1], nx + screenpos[0], ny + screenpos[1])
        img = screenshot()
        lines = contour.get_dashed(img)
        
#def moveblocks(up, right, )        
        
    
   
if __name__ == "__main__":
    
    screenpos = getscreen()
    
    robot.click(screenpos[0]+100, screenpos[1]-10) # click in title bar to get focus
    findleftdash()
    
    # now move right
    robot.drag(screenpos[0] + 100, screenpos[3]-100, screenpos[2] - 100, screenpos[3]-100)
    imgright = screenshot()
    robot.drag(screenpos[2] - 100, screenpos[3]-100, screenpos[0] + 100, screenpos[3]-100)
    
    
    # move up
    robot.drag(screenpos[0] + 300, screenpos[3]-200, screenpos[0] + 300, screenpos[1]+100)
    imgup = screenshot()
    robot.drag(screenpos[0] + 300, screenpos[1]+100, screenpos[0] + 300, screenpos[3]-200)
      
    img = screenshot()
    
    grid = perspective.Grid.fromimg(img, imgright, imgup, show = False)
    grid.setscreen(screenpos[0], screenpos[1])
    
    img = screenshot()
    grid.draw(img)        
    robot.click(screenpos[0]+100, screenpos[1]-10) # click in title bar to get focus
    
    img = screenshot()
    grid.updatemids()
    #grid.draw(img)
    
    """
    for i in range(3):
        robot.drag(screenpos[0] + 300, screenpos[1]+600, screenpos[0] + 400, screenpos[1]+700, slow=True)

    img = screenshot()
    grid.draw(img)
    robot.click(screenpos[0]+100, screenpos[1]-10) # click in title bar to get focus
    for i in range(3):
        robot.drag(screenpos[0] + 400, screenpos[1]+700, screenpos[0] + 300, screenpos[1]+600, slow=True)
    img = screenshot()
    grid.draw(img)    
    robot.click(screenpos[0]+100, screenpos[1]-10) # click in title bar to get focus    
      
    """
    for i in range(5):
        grid.move(5,0)
        grid.updatemids()
        #img = screenshot()
        #grid.draw(img)
        #robot.click(screenpos[0]+100, screenpos[1]-10) # click in title bar to get focus
    for i in range(5):
        grid.move(-5,0)
        grid.updatemids()
        #img = screenshot()
        #grid.draw(img)
        #robot.click(screenpos[0]+100, screenpos[1]-10) # click in title bar to get focus
    img = screenshot()
    grid.draw(img)
    