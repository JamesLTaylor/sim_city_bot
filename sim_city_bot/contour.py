import cv2
import numpy as np

def realtimethresh(img):
    """
    day works well with 200, night works better with 155
    after equalization night looks better with 240, which works pretty well for day too
    """
    cv2.namedWindow('thresh')
    cv2.createTrackbar('thresh','thresh',128,255, lambda x : x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    while(1):
        thresh = cv2.getTrackbarPos('thresh','thresh')
        ret,thresh_img = cv2.threshold(gray,thresh,255,0)
        
        # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)        
        
        cv2.imshow('thresh',thresh_img)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    
def groupclose(data, distance):
    """
    """
    grouped = []
    for (pos, datum) in enumerate(data):
        found = False
        if len(grouped)>0:  
            for i in range(len(grouped)):                
                if np.abs(datum-grouped[i]['avg'])<distance and not found:
                    entry = grouped[i]
                    entry['count'] += 1
                    entry['total'] += datum
                    entry['avg'] = entry['total'] / entry['count']
                    entry['data'].append(datum)
                    entry['inds'].append(pos)
                    found = True

        if not found:
            entry = {'avg':datum, 'count':1, 'total':datum, 'data':[datum], 'inds':[pos]}
            grouped.append(entry)
    
    return grouped
                    
            
    
def get_dashed(img):
    """    
    returns a list of 1 or 2 elements, each element is the contours around the 
    dashes of the dashed line, there will be at least 5
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ret,thresh_img = cv2.threshold(gray,240,255,0)    
    #cv2.imshow('thresh',thresh_img)
    #_,contours,_ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    _,contours,_ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []
    angles = []
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.02*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.putText(img, str(i), (x+w, y+h), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0])
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if (len(approx)==4 and  
            len(cnt)>70 and len(cnt)<150): #can be approximated by a rectangle and has the length of the typical dashes
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            if (ma/MA) > 4 and (ma/MA) < 8.5: # another check on shape so that it is suitably long and thin
                x,y,w,h = cv2.boundingRect(cnt)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                #cv2.drawContours(img,cnt,1,(0,0,255),2)
                candidates.append(approx)
                angles.append(angle)
        
    grouped = groupclose(angles, 1)   
    results = [{},{}]
    for i in range(len(grouped)):
        if grouped[i]['count']>=3:
            group = {}
            group['contours'] = []
            group['angle'] = grouped[i]['avg']
            
            for j in range(grouped[i]['count']):
                cnt = candidates[grouped[i]['inds'][j]]
                group['contours'].append(cnt)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #cv2.drawContours(img,[box],0,(255,0,0),2)
            if group['angle'] < 45 or group['angle'] > 135:
                results[0] = group
            else:
                results[1] = group
                
    #cv2.imshow('dottedline',img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
    return results
                

    
if __name__ == "__main__":
    #img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\day2.png")
    #img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\leftcornerday.png")
    #img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\night2.png")
    img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\dashed_at_top.png")
    #realtimethresh(img)
    get_dashed(img)


