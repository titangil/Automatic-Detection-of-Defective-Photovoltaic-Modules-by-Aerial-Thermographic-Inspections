# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 00:10:52 2021

@author: titan
"""

import cv2 as cv
import numpy as np
import imutils 
import utils
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from matplotlib.pyplot import figure
import os


'''vidcap = cv.VideoCapture('Compton.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1'''
  
cap= cv.VideoCapture('Images/flight.mp4')
i=0
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    crop_img = frame[:, :]
    cv.imwrite('kang'+str(i)+'.jpg',crop_img)
    
    #cv2.imshow('crop',crop_img)
    
    i+=1
    count += 20
    cap.set(cv.CAP_PROP_POS_FRAMES,count)
    
 
cap.release()
cv.destroyAllWindows()

for x in range(1):
    
    img1_org = cv.imread('kang1.jpg')
    #cv.imshow('1',img1_org)
    img1 = cv.cvtColor(img1_org,cv.COLOR_BGR2GRAY)
    
    img2_org = cv.imread('kang0.jpg')
    #cv.imshow('2',img2_org)
    img2 = cv.cvtColor(img2_org,cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #cv.imshow('keypoint',cv.drawKeypoints(img1, kp1, None))
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    match = cv.FlannBasedMatcher(index_params,search_params)
    #match = cv.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good.append(m)
    
    
    draw_param = dict(matchColor= (0,255,0),singlePointColor = None,flags = 2)
    
    img_matches = cv.drawMatches(img1_org,kp1,img2_org,kp2,good,None,**draw_param)
    #cv.namedWindow('matches', cv.WINDOW_NORMAL)
    #cv.imshow('matches',img_matches)
    
    MIN_MATCH_COUNT = 5
    if len(good) > MIN_MATCH_COUNT:
        src_pts =  np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts =  np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        M,mask = cv.findHomography(src_pts, dst_pts,cv.RANSAC,5.0)
        
        h,w = img1.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, M)
        imgmatch = cv.polylines(img2,[np.int32(dst)], True, 255,3,cv.LINE_AA)
        #cv.imshow('match1',imgmatch)
    else:
        print('not enough matches')
        
    dst = cv.warpPerspective(img1_org,M, (img2_org.shape[1] + img1_org.shape[1],img2_org.shape[0]))
    dst[0:img2_org.shape[0],0:img2_org.shape[1]] = img2_org
    #cv.namedWindow('final', cv.WINDOW_KEEPRATIO)
    #cv.imshow('final', dst)
    
    def trim(frame):
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop top
        if not np.sum(frame[:,0]):
            return trim(frame[:,1:])
        #crop top
        if not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])
        return frame
    #cv.namedWindow('crop', cv.WINDOW_NORMAL)
    cv.imwrite('kang.jpg',trim(dst))
    
    
for x in range(i-1):
    print(i)
    print('kang'+str(x+2)+'.jpg')
    if x+2 == i:
        break
    else:
        img1_org = cv.imread('kang'+str(x+2)+'.jpg')
        #cv.imshow('1',img1_org)
        img1 = cv.cvtColor(img1_org,cv.COLOR_BGR2GRAY)
        
        img2_org = cv.imread('kang.jpg')
        #cv.imshow('2',img2_org)
        img2 = cv.cvtColor(img2_org,cv.COLOR_BGR2GRAY)
        
        sift = cv.xfeatures2d.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        #cv.imshow('keypoint',cv.drawKeypoints(img1, kp1, None))
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        match = cv.FlannBasedMatcher(index_params,search_params)
        #match = cv.BFMatcher()
        matches = match.knnMatch(des1, des2, k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good.append(m)
        
        
        draw_param = dict(matchColor= (0,255,0),singlePointColor = None,flags = 2)
        
        img_matches = cv.drawMatches(img1_org,kp1,img2_org,kp2,good,None,**draw_param)
        #cv.namedWindow('matches', cv.WINDOW_NORMAL)
        #cv.imshow('matches',img_matches)
        
        MIN_MATCH_COUNT = 5
        if len(good) > MIN_MATCH_COUNT:
            src_pts =  np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts =  np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            
            M,mask = cv.findHomography(src_pts, dst_pts,cv.RANSAC,5.0)
            
            h,w = img1.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)
            imgmatch = cv.polylines(img2,[np.int32(dst)], True, 255,3,cv.LINE_AA)
            #cv.imshow('match1',imgmatch)
        else:
            print('not enough matches')
            
        dst = cv.warpPerspective(img1_org,M, (img2_org.shape[1] + img1_org.shape[1],img2_org.shape[0]))
        dst[0:img2_org.shape[0],0:img2_org.shape[1]] = img2_org
        #cv.namedWindow('final', cv.WINDOW_KEEPRATIO)
        #cv.imshow('final', dst)
        
        def trim(frame):
            #crop top
            if not np.sum(frame[0]):
                return trim(frame[1:])
            #crop top
            if not np.sum(frame[-1]):
                return trim(frame[:-2])
            #crop top
            if not np.sum(frame[:,0]):
                return trim(frame[:,1:])
            #crop top
            if not np.sum(frame[:,-1]):
                return trim(frame[:,:-2])
            return frame
        #cv.namedWindow('crop', cv.WINDOW_NORMAL)
        cv.imwrite('kang.jpg',trim(dst))

key = cv.waitKey(13)
if key == 13:
    cv.destroyAllWindows()
