import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\day1.png")
template = cv2.imread("C:\\Dev\\Trunk\\python\\james\\image\\helmettick.png")

h, w, ignore = template.shape


res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    

cv2.imshow('a',img)
cv2.waitKey()
cv2.destroyAllWindows()
 