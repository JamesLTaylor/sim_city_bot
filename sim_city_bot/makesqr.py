import cv2
import numpy as np

mousedown = False # true if mouse is pressed

lines = np.float32(np.zeros((4,2)))

ix,iy = -1,-1
line = 0



def change(event,x,y,flags,param):
    global ix,iy,imgoverlay,imgtemp, mousedown, line

    if event == cv2.EVENT_LBUTTONDOWN:
        mousedown = True        
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mousedown == True:
            imgtemp = np.zeros(img.shape, np.uint8)
            cv2.line(imgtemp, (ix, iy), (x,y), (255,255,255))
            
            

    elif event == cv2.EVENT_LBUTTONUP:
        mousedown = False
        imgoverlay = cv2.add(imgoverlay, imgtemp)
        lines[line,:] = np.array([ix,iy]).astype('float32')
        lines[line+1,:] = np.array([x,y]).astype('float32')
        line = line + 2


img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\day1.png")
imgoverlay = np.zeros(img.shape, np.uint8)
imgtemp = np.zeros(img.shape, np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image',change)

while(1):
    try:
        cv2.imshow('image',cv2.add(cv2.add(img,imgoverlay), imgtemp))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):        
            pts1 = lines
            midx1 = (lines[0,0] + lines[1,0])/2
            midx2 = (lines[2,0] + lines[3,0])/2
            lines2 = np.copy(lines)
            lines2[0,0] = midx1
            lines2[1,0] = midx1
            lines2[2,0] = midx2
            lines2[3,0] = midx2
            print(lines)
            print(lines2)
            M = cv2.getPerspectiveTransform(lines,lines2)
            img = cv2.warpPerspective(cv2.add(img,imgoverlay),M, (img.shape[1], img.shape[0]))
            imgoverlay = np.zeros(img.shape, np.uint8)
            imgtemp = np.zeros(img.shape, np.uint8)
            line = 0
    
        elif k == 27:
            break
    except:
        cv2.destroyAllWindows()
        raise

cv2.destroyAllWindows()