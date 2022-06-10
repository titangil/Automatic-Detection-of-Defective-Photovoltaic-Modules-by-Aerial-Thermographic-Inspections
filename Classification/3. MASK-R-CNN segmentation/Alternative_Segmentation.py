import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import os
directory = os.fsencode("")



global a
a = np.empty((0,2), int)



def draw_circle(event,x,y,flags,param):
    global a
    
    global mouseX,mouseY
    
    if  event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),5,(0,0,255),-1)
        mouseX,mouseY = x,y
       # print(x,y)
       
        a = np.append(a, np.array([[x,y]]), axis=0)

cv.namedWindow("image", cv.WINDOW_NORMAL)
      
img = cv.imread("")
img_copy =  img.copy()
rows,cols,ch = img.shape
#print(rows,cols)

cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)



cv.destroyAllWindows()

b = np.float32(a)

offset = 200
pts2 = np.float32([ [offset+0,offset+0],
                    [offset+1200,offset+0],
                    [offset+0,offset+200],
                    [offset+1200,offset+200] ]) 


M = [[ 1.88441982e+00, -2.81272470e-02 , 1.99991364e+02],
     [-1.64444376e-01 , 8.79878656e-01 , 1.33939785e+02],
     [-3.36981969e-04 ,-1.10909115e-05 , 1.00000000e+00]]
M = np.array(M)
print(""+str(type(M)))
