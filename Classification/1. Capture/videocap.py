'''import cv2
import numpy as np
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

cap = cv2.VideoCapture('Images/flight.mp4')
ret, frame = cap.read()
count = 0
while(cap.isOpened()):
    if count == 0:
        prev_frame=frame[:]
    
    ret, frame = cap.read()
    if ret:
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #detect key feature points
        sift = cv2.xfeatures2d.SIFT_create()
        #kp, des = sift.detectAndCompute(gray1, None)
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        #some magic with prev_frame
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        match = cv2.FlannBasedMatcher(index_params,search_params)
        #match = cv.BFMatcher()
        matches = match.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good.append(m)
        draw_param = dict(matchColor= (0,255,0),singlePointColor = None,flags = 2)

        img_matches = cv2.drawMatches(frame,kp1,prev_frame,kp2,good,None,**draw_param)
        #cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
        #cv2.imshow('matches',img_matches)
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts =  np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts =  np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
            M,mask = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC,5.0)
    
            h,w = img1.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            imgmatch = cv2.polylines(img2,[np.int32(dst)], True, 255,3,cv2.LINE_AA)
            #cv2.imshow('match1',imgmatch)
        else:
            print('not enough matches')

        #draw key points detected
        #img=cv2.drawKeypoints(img1,kp1,img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        dst = cv2.warpPerspective(frame,M, (prev_frame.shape[1] + frame.shape[1],prev_frame.shape[0]))
        dst[0:prev_frame.shape[0],0:prev_frame.shape[1]] = prev_frame
        cv2.namedWindow('final', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('final', dst)
        cv2.imshow('1',img1)
        cv2.imshow('2',img2)
        
        prev_frame = dst
        
        cv2.namedWindow('crop', cv2.WINDOW_NORMAL)
        cv2.imshow("crop", trim(dst))
        count+=1
        print(count)
        
    else:
        print('Could not read frame')

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

#cap.release()
#cv2.destroyAllWindows()'''
import cv2
cap= cv2.VideoCapture('Images/flight.mp4')
i=0
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    crop_img = frame[:, 0:400]
    cv2.imwrite('kang'+str(i)+'.jpg',crop_img)
    
    #cv2.imshow('crop',crop_img)
    
    i+=1
    count += 10
    print(i)
    cap.set(cv2.CAP_PROP_POS_FRAMES,count)
    
 
cap.release()
cv2.destroyAllWindows()