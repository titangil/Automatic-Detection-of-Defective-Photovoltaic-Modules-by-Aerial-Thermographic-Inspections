import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import math




img = cv2.imread("thermal.jpg")
cv2.namedWindow("out", cv2.WINDOW_NORMAL)

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

        


M = cv2.getPerspectiveTransform(np.array(psort), new_pt)


dst = cv2.warpPerspective(img, M, (cols, rows))
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

kernel = np.ones((3,3),np.uint8)
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
zero =  cv2.dilate(zero,kernel,iterations = 2)
zero = np.uint8(zero)
ret,th1 = cv2.threshold(zero,127,255,cv2.THRESH_BINARY)
cnst2 = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
points = []    
#cnst2=sorted(cnst2,key=cv2.contourArea,reverse=False)[:3]
cnst2 = imutils.grab_contours(cnst2) 
t = 1

print(len(cnst2))
row = len(cnst2)/2
col = 2
new_w = (80)*row
new_h = (120)*col
black_canvas = np.zeros((int(new_h),int(new_w)))
black_canvas = np.uint8(black_canvas)
black_canvas =  cv2.merge([black_canvas,black_canvas,black_canvas])
print(black_canvas.shape)

for c in cnst2:     
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)  
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
        points = sorted(points,key=lambda data:(data[0]),reverse = False)
        t+=1
      

point1 = []
point2 = []
for c in points:

    if c[1] >= dst.shape[0]/2-5:
        point2.append((c[0],c[1],c[2],c[3]))
    else:
        point1.append((c[0],c[1],c[2],c[3]))
t = 0    
row1 = 0
row2 = 0  
print(point1)
for c in point1:
    cv2.circle(dst_color,(c[0],c[1]),4,(255,0,0),-1)
    cv2.putText(dst_color,str(t),(c[0],c[1]+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1 )
    cv2.rectangle(dst_color, (c[0],c[1]),(c[0]+c[2],c[1]+c[3]), (255,0,255), 2)
    final1 = dst_color[ c[1]:c[1]+c[3],  c[0]:c[0]+c[2] ]
    final1 =  cv2.resize(final1, (80, 120))
    
    '''if t < 10:
        #print("write")
        cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(00,0'+str(t)+').jpg', final1 )
    else:
        #print("write2")
        cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(00,'+str(t)+').jpg', final1 )'''
        
    black_canvas[0:0+final1.shape[0], 80*(row1):80*(row1)+final1.shape[1]] = final1
    t+=1
    row1 +=1

for c in point2:
    cv2.circle(dst_color,(c[0],c[1]),4,(255,0,0),-1)
    cv2.putText(dst_color,str(t),(c[0],c[1]+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1 )
    cv2.rectangle(dst_color, (c[0],c[1]),(c[0]+c[2],c[1]+c[3]), (255,0,255), 2)
    final2 = dst_color[ c[1]:c[1]+c[3],  c[0]:c[0]+c[2] ]
    final2 =  cv2.resize(final2, (80, 120))
    '''if row2 < 10:
        cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(01,0'+str(row2)+').jpg', final2 )
    else:
        cv2.imwrite('D:/Works/Project/Code/Houghline/pic/Without numbers/(01,'+str(row2)+').jpg', final2 )'''
    #cv2.imwrite('D:/Works/Project/Code/Houghline/pic/With numbers/(01,'+str(row2+1)+').jpg', final2 )  #str(int((t)-len(cnst2)/2))
    
    
    black_canvas[120:120+final2.shape[0], 80*(row2):80*(row2)+final2.shape[1]] = final2
    row2 +=1
    #cv2.imshow('canvas',final2)
    t+=1

