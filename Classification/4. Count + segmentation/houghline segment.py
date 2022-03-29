# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:40:58 2022

@author: titan
"""

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import math




img = cv2.imread("testimg3.png")


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows,cols = gray.shape
new_pt = np.float32([[0, 0],[cols, 0],[0,rows],[cols, rows]])

ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
cnst = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       
cnst = imutils.grab_contours(cnst)   
for c in cnst:
    approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    if len(approx) == 4:
        screenCnt = approx
        #print(screenCnt.ndim)
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 2)   
        break
screenCnt = np.array(screenCnt[:,0,:],dtype=np.float32 )
p1, p2, p3, p4 = screenCnt[0], screenCnt[1], screenCnt[2], screenCnt[3]
cont1 = img[0:864,]
cnt1,cnt2,cnt3,cnt4 = (np.array([[0,0], [int(cols/2),0], [int(cols/2),int(rows/2)], [0,int(rows/2)]]),
                       np.array([[int(cols/2),int(rows/2)], [int(cols/2),0], [int(cols),int(0)], [int(cols),int(rows/2)]]),
                       np.array([[int(cols/2),int(rows/2)], [int(cols/2),rows], [int(0),int(rows)], [int(0),int(rows/2)]]),
                       np.array([[int(cols/2),int(rows/2)], [int(cols/2),rows], [int(cols),int(rows)], [int(cols),int(rows/2)]]))
cnt = (cnt1,cnt2,cnt3,cnt4)
psort = []
for c in cnt:
    for p in screenCnt:
        result = cv2.pointPolygonTest(c, p, False)
        if result == -1:continue
        else:psort.append(p)                                                    #Sort 4 points corner

'''print(screenCnt)        
print(np.array(psort))'''
        


M = cv2.getPerspectiveTransform(np.array(psort), new_pt)


dst = cv2.warpPerspective(img, M, (cols, rows))
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.uint8)
offset = 30
blur = cv2.GaussianBlur(dst,(5,5),1)
edges = cv2.Canny(blur,0,100)

zero = np.zeros((rows,cols))
dst_color = cv2.merge([dst,dst,dst])
lines = cv2.HoughLines(edges, 1, np.pi/100,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    cv2.line(zero,(x1,y1),(x2,y2),(255,255,255),3)
lines = cv2.HoughLines(edges,5,np.pi/1,100)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    cv2.line(zero,(x1,y1),(x2,y2),(255,255,255),3)



zero = 255-zero
zero = np.uint8(zero)
ret,th1 = cv2.threshold(zero,127,255,cv2.THRESH_BINARY)
cnst2 = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
points = []    
#cnst2=sorted(cnst2,key=cv2.contourArea,reverse=False)[:3]
cnst2 = imutils.grab_contours(cnst2) 
t = 1

print(len(cnst2))


for c in cnst2:
        
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        '''if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(dst, [screenCnt], -1, (0, 255, 0), 2)  '''
            
        
        area = cv2.contourArea(c)
        
      
    
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
    # set values as what you need in the situation
            cX, cY = 0, 0
        
        
        x,y,w,h=cv2.boundingRect(c)
        points.append((x,y,w,h))
        #print(points)
        '''cv2.drawContours(dst_color, [c],-1, (0,255,0), 2)      ### show lines, dots, and numbers  ####
        cv2.circle(dst_color,(cX,cY),4,(255,0,0),-1)
        cv2.putText(dst_color,str(t),(cX-20,cY-7),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1 )'''
        points = sorted(points,key=lambda data:(data[0]),reverse = False)
        #print(dst.shape)
        #print(cX,cY)
        t+=1
        '''if t == 72:
            x, y, width, height = cv2.boundingRect(c)
            cv2.imshow('test',dst_color[y:y+height, x:x+width])
            cv2.imwrite('D:/Works/Project/Code/Houghline/pic/test.jpg', dst_color[y:y+height, x:x+width])'''
       

point1 = []
point2 = []
for c in points:
    #print(c[0],c[1])
    #print(c[1])
    if c[1] > dst.shape[0]/2:
        point2.append((c[0],c[1],c[2],c[3]))
    else:
        point1.append((c[0],c[1],c[2],c[3]))
t = 0      
for c in point1:
    '''cv2.circle(dst_color,(c[0],c[1]),4,(255,0,0),-1)
    cv2.putText(dst_color,str(t),(c[0],c[1]+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1 )
    cv2.rectangle(dst_color, (c[0],c[1]),(c[0]+c[2],c[1]+c[3]), (255,0,255), 1)'''
    
    if t < 10:
        cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(0,'+str(t)+').jpg', dst_color[ c[1]:c[1]+c[3],  c[0]:c[0]+c[2] ] )
    else:
        cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(0,'+str(t)+').jpg', dst_color[ c[1]:c[1]+c[3],  c[0]:c[0]+c[2] ] )
    t+=1

for c in point2:
    '''cv2.circle(dst_color,(c[0],c[1]),4,(255,0,0),-1)
    cv2.putText(dst_color,str(t),(c[0],c[1]+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1 )
    cv2.rectangle(dst_color, (c[0],c[1]),(c[0]+c[2],c[1]+c[3]), (255,0,255), 1)'''
    cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(1,'+str(int((t)-len(cnst2)/2))+').jpg', dst_color[ c[1]:c[1]+c[3],  c[0]:c[0]+c[2] ] )
    t+=1
#print(point2)
cv2.imshow('out',dst_color)

#cv2.imwrite('persp.jpg',dst)
