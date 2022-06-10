
import cv2
cap= cv2.VideoCapture("")
i=0
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break
    crop_img = frame[:, 0:400]
    
    cv2.imwrite("frame_"+str(i)+".jpg",crop_img)
     
    i+=1
    count += 10
    #print(i)
    cap.set(cv2.CAP_PROP_POS_FRAMES,count)
    
 
cap.release()
cv2.destroyAllWindows()
