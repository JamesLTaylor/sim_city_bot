import contour
import robot
import win32api, win32con
import time
import math
import numpy as np
import pyscreenshot
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, vanishtop, midsbottom, vanishleft, midsleft, origin):
        self.vanishtop = vanishtop
        self.vanishleft = vanishleft
        self.midsbottom = np.vstack((origin, midsbottom))
        self.midsleft = np.vstack((origin, midsleft))
        self.origin = origin
        self.err_estimate = np.array([0, 0])
        if len(midsbottom)<8 or len(midsleft)<6:
            raise Exception("not enough grid points")
        
    def setscreen(self, x, y):
        self.x = x
        self.y = y
        
    
    @classmethod
    def fromimg(cls, img, imgright, imgup, show):
        gridvals = getgrid(img, imgright, imgup, show)
        (m1, c1) = getmc(img, vertical = True)
        (m2, c2) = getmc(img, vertical = False)        
        origin = intercept(m1, c1, m2, c2)
        return cls(gridvals['vanishtop'], gridvals['midsbottom'], gridvals['vanishleft'], gridvals['midsleft'], origin)
    
    def xy(self, row, col):
        """
        gives the xy coordinates of the vertex at the top righ corner of block(row, col)
        """
        (m1, c1) = mcfrom2p(self.vanishtop[0], self.vanishtop[1], self.midsbottom[col,0], self.midsbottom[col,1])
        (m2, c2) = mcfrom2p(self.vanishleft[0], self.vanishleft[1], self.midsleft[row,0], self.midsleft[row,1])
        
        return intercept(m1, c1, m2, c2)
    
    def move(self, rows, cols):
        (x2, y2) = self.xy(0, 0)
        (x1, y1) = self.xy(np.abs(rows), np.abs(cols))
        
        if rows>0:
            startpoint = (int(x1 + self.x - self.err_estimate[0]), int(y1 + self.y - self.err_estimate[1]))
            endpoint = (int(x2 + self.x), int(y2 + self.y))
            im = pyscreenshot.grab(bbox = (startpoint[0]-25, startpoint[1]-25, startpoint[0]+25, startpoint[1]+25))
            in_data = np.asarray(im, dtype=np.uint8)
            img = np.array(in_data[:,:,::-1])
            cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\move\\up_start.PNG", img)            
            
            robot.drag(startpoint[0], startpoint[1], endpoint[0], endpoint[1], slow=True)
            #robot.drag(x1+self.x, y1+self.y, x2+self.x, y1+self.y, slow=True)
            #robot.drag(x2+self.x, y1+self.y, x2+self.x, y2+self.y, slow=True)

            im = pyscreenshot.grab(bbox = (endpoint[0]-25, endpoint[1]-25, endpoint[0]+25, endpoint[1]+25))
            in_data = np.asarray(im, dtype=np.uint8)
            img = np.array(in_data[:,:,::-1])
            cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\move\\up_end.PNG", img)            
            
        else:
            startpoint = (int(x2 + self.x - self.err_estimate[0]), int(y2 + self.y - self.err_estimate[1]))
            endpoint = (int(x1 + self.x), int(y1 + self.y))
            im = pyscreenshot.grab(bbox = (startpoint[0]-25, startpoint[1]-25, startpoint[0]+25, startpoint[1]+25))
            in_data = np.asarray(im, dtype=np.uint8)
            img = np.array(in_data[:,:,::-1])
            cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\move\\down_start.PNG", img)
            robot.drag(x2+self.x, y2+self.y, x1+self.x, y1+self.y, slow=True)
            #robot.drag(x2+self.x, y2+self.y, x1+self.x, y2+self.y, slow=True)
            #robot.drag(x1+self.x, y2+self.y, x1+self.x, y1+self.y, slow=True)
            im = pyscreenshot.grab(bbox = (endpoint[0]-25, endpoint[1]-25, endpoint[0]+25, endpoint[1]+25))
            in_data = np.asarray(im, dtype=np.uint8)
            img = np.array(in_data[:,:,::-1])
            cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\move\\down_end.PNG", img)       
            
        
    def updatemids(self):
        """
        
        """
        # if we happen to be at the bottom then the first one needs to be 
        # obtained in a different way
        newmids = np.copy(self.midsleft)
        for i in range(1,len(self.midsleft)):
            x1 = self.midsleft[i,0]
            y1 = self.midsleft[i,1]
            bbox = (int(x1+self.x-25), int(y1+self.y-25), int(x1+self.x+25), int(y1+self.y+25))
            im = pyscreenshot.grab(bbox = bbox)
            in_data = np.asarray(im, dtype=np.uint8)
            img = np.array(in_data[:,:,::-1])
            cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\move\\update" + str(i) + ".PNG", img)     
            newmids[i] =  getmidsmall(img) + np.array([bbox[0]-self.x, bbox[1]-self.y])
        
        print(self.midsleft-newmids)
        err = self.midsleft-newmids
        err = err[1:,:]
        err = err[err[:,0]<1000]
        
        if len(err)>3:
            err = np.mean(err, axis=0)
            self.err_estimate = err
            print("UPDATED to!" + str(err))
            
        
    def draw(self, img):
        gridimg = drawgrid(img, self.vanishtop, self.midsbottom, vertical = True, show = False)
        gridimg = drawgrid(gridimg, self.vanishleft, self.midsleft, vertical = False, show = True)
        
def show(img):  
    b,g,r = cv2.split(img)  
    rgb_img = cv2.merge([r,g,b])  
    plt.imshow(rgb_img)

def getquad(contour):    
    epsilon = 0.2*cv2.arcLength(contour,True)    
    approx = cv2.approxPolyDP(contour,epsilon,True)
    
    count = 0
    while len(approx)<4 and count<12:
        count += 1
        epsilon = epsilon*0.9
        approx = cv2.approxPolyDP(contour,epsilon,True)
        
    return approx
    
        
def getmidsmall(img):
    """
    
    
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,contours,_ = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours)!=2:
        print("can't find two regions")
        return np.array([-1000, -1000])
    
    # find a row the divides the two white areas
    blackrows = np.all(th2==0, axis=1)
    
    if not np.any(blackrows):
        print("can't find a gap between the two regions")
        return np.array([-1000, -1000])
       
    approx1 = getquad(contours[0])
    try:
        approx1 = np.reshape(approx1,(4,2))
    except:
        print("can't find quadrilateral")
        return np.array([-1000, -1000])
        
    approx1 = approx1[np.argsort(approx1[:,1])]

    approx2 = getquad(contours[1])
    try:
        approx2 = np.reshape(approx2,(4,2))
    except:
        print("can't find quadrilateral")
        return np.array([-1000, -1000]) 
        
    approx2 = approx2[np.argsort(approx2[:,1])]
    
    if approx1[0,1] > approx2[0,1]:
        temp = approx1
        approx1 = approx2
        approx2 = temp
    
    midx = (approx1[2,0] + approx1[3,0] + approx2[0,0] + approx2[1,0])/4.0
    midy = (approx1[2,1] + approx1[3,1] + approx2[0,1] + approx2[1,1])/4.0
    
    return np.array([midx, midy])
    

def getmc_regress(img):
    lines = contour.get_dashed(img)
    cnts = np.asarray(lines[0]['contours'])
    c = np.reshape(cnts, (np.size(cnts, 0)*np.size(cnts, 1), 2))
    
    x = c[:,0]
    y = c[:,1]
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]

    x1 = np.min(x)
    x2 = np.max(x)
    y1 = int(m*x1+c)
    y2 = int(m*x2+c)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)    
    
    cv2.imshow('fitted',img)
    cv2.waitKey()
    cv2.destroyAllWindows()    
    
    return (m, c)
    
def getmc(img, vertical):
    """ Get the slope and intercept for the dashed line at the edge of the board
    """
    
    lines = contour.get_dashed(img)
    if vertical:
        cnts = np.asarray(lines[0]['contours'])
    else:
        cnts = np.asarray(lines[1]['contours'])
    c = np.reshape(cnts, (np.size(cnts, 0)*np.size(cnts, 1), 2))
    
    x = c[:,0]
    y = c[:,1]
    
    if vertical: # sort by y
        ind = np.argsort(c[:,1])
    else:
        ind = np.argsort(c[:,0])
        
    c = c[ind]

    x1 = (c[0,0]+c[1,0])/2
    y1 = (c[0,1]+c[1,1])/2
    x2 = (c[-1,0]+c[-2,0])/2
    y2 = (c[-1,1]+c[-2,1])/2
    
    if np.abs((x1-x2))<1e-5:
        m = np.inf
        c = x1
    else:
        m = (float(y1)-y2)/(x1-x2)
        c = y1 - m * x1

    #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cv2.imshow('fitted',img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()    
    
    return (m, c) 
    
    
def intercept(m1, c1, m2, c2):
    """ Get the intercept between two lines
    """
    if m1==np.inf:
        x = c1
        y = m2*x+c2
    elif m2==np.inf:
        x=c2
        y = m1*x + c1        
    else:
        x= -(c1 - c2) / (m1 - m2)
        y = m1 * x + c1
        
    return (x, y)
    
def fillmissing(leftp):
    dist1 = np.power((leftp[:-1,0]-leftp[1:,0]),2) + np.power((leftp[:-1,1]-leftp[1:,1]),2)
    dist1 = np.sqrt(dist1)
    d  = np.sort(dist1)
    ratio = np.abs(d[1:]-d[:-1])
    ratio = np.divide(ratio, d[:-1])
    if np.any(ratio>0.2):
        ind = np.argmax(np.abs(d[1:]-d[:-1]))
        m = np.mean(d[:ind+1])
        ngaps = np.round(dist1/m)
        newleft = leftp[0]
        for i in range(len(ngaps)):
            gap = ngaps[i]
            if int(gap)==1:
                newleft = np.append(newleft, leftp[i+1])
            elif int(gap)==2:
                newleft = np.append(newleft, (leftp[i]+leftp[i+1])/2)
                newleft = np.append(newleft, leftp[i+1])
            elif int(gap)==3:
                newleft = np.append(newleft, leftp[i] + (leftp[i+1]-leftp[i])*(1.0/3.0))
                newleft = np.append(newleft, leftp[i] + (leftp[i+1]-leftp[i])*(2.0/3.0))
                newleft = np.append(newleft, leftp[i+1])
            else:
                raise Exception("Gap too big")
        leftp = np.reshape(newleft, (len(newleft)/2,2))
    return leftp

    
def midgaps(img, vertical, show): 
    """ The points between the dashes on the lines at the edge of the screen
    """
    try:
        original_img = np.copy(img)
        cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\start.PNG", img) 
        lines = contour.get_dashed(img)
        if vertical:
            cnts = np.asarray(lines[0]['contours'])
        else:
            cnts = np.asarray(lines[1]['contours'])
        
        leftp = np.zeros((len(cnts),2))
        rightp = np.zeros((len(cnts),2))
        print(len(cnts))
        for i in range(len(cnts)):
            cnt = np.reshape(cnts[i],(4,2))
            if vertical:
                cnt = cnt[np.argsort(cnt[:,1])]
            else:
                cnt = cnt[np.argsort(cnt[:,0])]
            x1 = (cnt[0,0]+cnt[1,0])/2
            y1 = (cnt[0,1]+cnt[1,1])/2
            x2 = (cnt[-1,0]+cnt[-2,0])/2
            y2 = (cnt[-1,1]+cnt[-2,1])/2
            leftp[i] = np.array([x1, y1])
            rightp[i] = np.array([x2, y2])
            if show:
                cv2.circle(img, (x1, y1), 2, (0,0,255), -1)
                cv2.circle(img, (x2, y2), 2, (0,0,255), -1)
        
        leftp = leftp[np.argsort(leftp[:,0])]
        rightp = rightp[np.argsort(rightp[:,0])]
        
        leftp = fillmissing(leftp)
        rightp = fillmissing(rightp)        
                
        midx = (rightp[:-1,0]+leftp[1:,0])/2
        midy = (rightp[:-1,1]+leftp[1:,1])/2
        
        newmids = np.vstack((midx, midy)).T
        
        if show:
            for (x, y) in zip(midx, midy):
                cv2.circle(img, (int(x), int(y)), 3, (0,255,255), -1)
                    
        if show:
            for (x, y) in zip(newmids[:,0], newmids[:,1]):
                cv2.circle(img, (int(x), int(y)), 5, (0,255,0), 2)
        
            cv2.imshow('fitted',img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        #if vertical:
            #newmids = np.flipud(newmids)
     
        
        return newmids
    except:
        cv2.imwrite("C:\\Dev\\Trunk\\python\\james\\image\\error.PNG", original_img)    
        raise
        
    
def mcfrom2p(x1, y1, x2, y2):    
        m = (float(y1)-y2)/(x1-x2)
        c = y1 - m * x1
        return (m, c)
    
def drawgrid(img, vpoint, mids, vertical, show):
    if vertical:
        y1 = 1
        y2 = img.shape[0]
    
        for i in range(len(mids)):
            (m, c) = mcfrom2p(vpoint[0], vpoint[1], mids[i,0], mids[i,1])
            x1 = int((y1 - c)/m)
            x2 = int((y2 - c)/m)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        x1 = 1
        x2 = img.shape[1]
    
        for i in range(len(mids)):
            (m, c) = mcfrom2p(vpoint[0], vpoint[1], mids[i,0], mids[i,1])
            y1 = int(x1 * m + c)
            y2 = int(x2 * m + c)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    if show:    
        cv2.imshow('fitted',img)
        cv2.waitKey()
        cv2.destroyAllWindows() 

    return img
    
def getgrid(img, imgright, imgup, show):
    """ 
    Get the vanishing points and intercepts for the grid
    
    :param img: an image with the dashed lines forming the bottom left corner
    :param imgright: img shifted right so that the vertical line is still visible
    :param imgup: img shifted up so that the horizonatal line is still visible    
    """
    (m1, c1) = getmc(img, vertical = True)
    (m2, c2) = getmc(imgright, vertical = True)
    (vx1, vy1) = intercept(m1, c1, m2, c2)    
    midsbottom = midgaps(img, vertical = False, show = False)
    
    gridimg = drawgrid(img, (vx1, vy1), midsbottom, vertical = True, show = False)
    
    (m1, c1) = getmc(img, vertical = False)
    (m2, c2) = getmc(imgup, vertical = False)    
    (vx2, vy2) = intercept(m1, c1, m2, c2)    
    midsleft = midgaps(img, vertical = True, show = False)
    
    gridimg = drawgrid(gridimg, (vx2, vy2), midsleft, vertical = False, show = show)
    
    return {'vanishtop': (vx1, vy1), 'midsbottom': midsbottom, 'vanishleft':(vx2, vy2), 'midsleft':midsleft}
    
    
def test1():
    img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\move\\update1.PNG")
    getmidsmall(img)
    
def test2():
    img = cv2.imread("C:\\Dev\\python\\sim_city_bot\\sim_city_bot\\\images\\leftcornernight.png")
    imgright = cv2.imread("C:\\Dev\\python\\sim_city_bot\\sim_city_bot\\\images\\leftcornernight_right.png")
    imgup = cv2.imread("C:\\Dev\\python\\sim_city_bot\\sim_city_bot\\\images\\leftcornerevening_up.png")
    
    grid = Grid.fromimg(img, imgright, imgup, show = False)
    grid.setscreen(0, 0)    


if __name__ == "__main__":
    """thisimg = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\error.PNG") 
    midsbottom = midgaps(thisimg, vertical = False, show = True)    
    """
    test2()

    
