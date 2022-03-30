
from distutils.log import error
from logging import exception
from scipy.misc import derivative
from sensor_msgs.msg import Image
import imutils
import torch
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
from plotter import Plotter
from simple_pid import PID
import numpy as np
import cv2
from mss import mss
from PIL import Image
from datetime import datetime
import sys
#sys.path.insert(1,'/home/titan/naive-scrcpy-client')
from NaiveScrcpyClient import *

def smooth_func(spd):
    global spd_prev
    if spd == 0:spd_smth = (spd * 0.015)+(spd_prev*0.985)
    else:spd_smth = (spd * 0.005)+(spd_prev*0.995)
    spd_prev = spd_smth
    return spd_smth

def run_client(_config):
    client = NaiveScrcpyClient(_config)
    ret = client.start_loop()
    if ret:
        return ret
    while True:
        try:
            img = client.get_screen_frame()
            if img is not None:
                cv2.imshow("img", img)
            c = cv2.waitKey(10)
            if c in [27, 13]:
                break
        except KeyboardInterrupt:
            break
    client.stop_loop()
    cv2.destroyAllWindows()
    return 0

def send_local_velocity( vx, vy, vz,yaw):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000011111000111, # Following ( 1 is mask OUT): yaw vel, yaw pos, force not accell, az,ay,ax, vz,vy,vx, pz,py,px 
        0, 0, 0,
        vx, vy, vz, # speed forward, right, down...
        0, 0, 0,
        0,float(yaw))  # yaw rate in radians/s...  
    vehicle.send_mavlink(msg)
    vehicle.flush()

config = {
        "max_size": 1280,
        "bit_rate": 2 ** 30,
        "crop": "-",
        "adb_path": "adb",
        "adb_port": 61550,
        "lib_path": "lib",
        "buff_size": 0x10000,
        "deque_length": 5
    }


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_v6.pt') 
model.to('cuda')                                                                                    # load yolov5
classes = model.names

vehicle = connect('udp:0.0.0.0:14550')                                                            # connect vehicle
print('connected')
start_time      = time.time()
display_time    = 1e-10
fc              = 0
FPS             = 0

point       = 0
center      = 0
dist        = 0
cX          = 0
cY          = 0
speed       = 0
size        = 0
resume      = 0 
nSecond     = 0
spd_prev    = 0
spd_smth    = 0 
velo        = 0



max_speed = 0.3 #m/s
max_speed = max_speed*0.01


pid         = PID(0.0005, 0.000000, 0.000000, setpoint=0)        #PID(0.0019, 0.000002, 0.000001, setpoint=0)                                        # pid horizonal allignment


total_confirm_time  = 0
confirm_time        = 0
confirm_interval    = 0
return_confirmation = 0
text = 'Idle'

p = Plotter(1280, 200)                                                           # set up graph
p_spd = Plotter(1280, 200)     

plot_canvas = 255*np.ones((200, 1280,3) ,np.uint8)
cv2.line(plot_canvas,(0,100),(1280,100),(0,255,0),1,cv2.LINE_AA)

velo_canvas = 255*np.ones((200, 1280,3) ,np.uint8)
cv2.line(velo_canvas,(0,100),(1280,100),(0,255,0),1,cv2.LINE_AA)

velo_canvas


cv2.namedWindow ("Show", cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Show', 670, 1111) 

#mon = {'left': 75, 'top': 85, 'width': 795, 'height': 475}                                          # set up screen cap
startTime = datetime.now()
client = NaiveScrcpyClient(config)
ret = client.start_loop()

while True:
    try:
        
        #screenShot = sct.grab(mon)                                                      
        #img = Image.frombytes('RGB', (screenShot.width, screenShot.height), screenShot.rgb, )       # get screen cap
        img = client.get_screen_frame()
        
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("Show", gray)

            frame = np.array(img)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_result = [gray]
            results = model(frame_result)
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]                            # process screen cap     
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]

        
            h,w,l = frame.shape
            #
            bw_copy =  np.zeros((h,w,3))                                                                #create b&w copy

            regeion_up   = bw_copy[0:int(h/5),0:w]
            regeion_down = bw_copy[int(4/5*h):h,0:w]                                                    #assign region
            regeion_area = int((h/5)*w)


            fc+=1
            TIME = time.time() - start_time
            if (TIME) >= display_time :
                FPS = fc/(TIME)                                                                         # fps fnction
                fc = 0
                start_time = time.time()
            fps_disp = "FPS: "+str(FPS)[:5]
            


            
            for i in range(n):
                row = cord[i]
                confidence = (row[4].cpu().numpy())
                if row[4] >= 0.65:
                    #if classes[int(labels[i])] == 'person':
                        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                        size  = (x2-x1) * (y2-y1)
                        if size == None: size = 0 
                        if size >= 25000:
                            bgr = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                            cv2.rectangle(bw_copy,(x1, y1), (x2, y2), (255,255,255), -1)
                            cv2.putText(frame, str(np.round(confidence,2)), (x1, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)             
                            cv2.putText(frame, classes[int(labels[i])], (x1, y1+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                            cv2.putText(frame, str(size), (x1, y1+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)              # setup boundary box
                    
                    

            bw_copy = np.uint8(bw_copy)
            gray = cv2.cvtColor(bw_copy, cv2.COLOR_BGR2GRAY)
            im_bw = cv2.threshold(gray ,128, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(im_bw, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            C = []

            for c in cnts:

                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #print(cX)
                C.append(cX)
                
                cv2.circle(bw_copy, (cX, cY), 5, (0,0, 255), -1)
                cv2.putText(bw_copy, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,0), 1)                                         # setup contour

            
            try:
                cX =   min(C, key=lambda x:abs(x-int(w/2)))                                         # get lowest horizontal distance
            except ValueError:
                pass

                    
            white_px_up     = round(np.sum(regeion_up == 255)/regeion_area,2)                       # extracting only white pixels 
            white_px_down   = round(np.sum(regeion_down == 255)/regeion_area,2)                     # extracting only black pixels 

            if      white_px_up >= 0.17 and white_px_down <= 0.05:  point, position = 1, "start"   
            elif    white_px_up >= 0.17 and white_px_down >= 0.17:  point, position = 2, "middle"           # set point
            elif    white_px_up <= 0.50 and white_px_down >= 0.17:  point, position = 3, "end"
            else:                                                   point, position = 0, "unknown"

            dist = ( (w/2)-cX )            # horizontal distance from the middle

            cv2.rectangle   (bw_copy,   (0,0),              (w,int(h/5)),       (0,0,255),          2)
            cv2.rectangle   (bw_copy,   (0,int(4/5*h)),     (w,h),             (0,0,255),          2)                   
            
            cv2.line        (bw_copy,   (int(w/2),0),       (int(w/2),h),(0,255,0),2)


            cv2.rectangle(frame,   (0,0),   (1280,110),             (255,255,255),    -1)      
            cv2.putText(frame,     fps_disp,                       (10, 40),   cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 7, 0),    2,cv2.LINE_AA)
            cv2.putText(frame, "Time: " + str(nSecond),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,7,0), 2,cv2.LINE_AA)
            cv2.putText(frame, "Battery: " + str(vehicle.battery.level),(250,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,7,0), 2,cv2.LINE_AA)
            cv2.putText(frame, "Voltage: " + str(vehicle.battery.voltage),(250,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,7,0), 2,cv2.LINE_AA)
            cv2.putText(frame, 
            "Position: " + str(np.round(vehicle.location.local_frame.north,2))+" " +  str(np.round(vehicle.location.local_frame.east,2))+ " "+str(np.round(vehicle.location.local_frame.down,2))
            ,(550,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,7,0), 2,cv2.LINE_AA)
            cv2.putText(frame, 
            "Velocity: " + str(round(vehicle.velocity[0],2))+" " +  str(round(vehicle.velocity[1],2))+ " "+str(round(vehicle.velocity[2],2))
            ,(550,90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,7,0), 2,cv2.LINE_AA)



            timeElapsed = (datetime.now() - startTime).total_seconds()
    #            print 'timeElapsed: {}'.format(timeElapsed)

            if timeElapsed >= 1:
                nSecond += 1
                #X += vehicle.velocity[2]
                #print(vehicle.velocity[0])
       
                print(vehicle.location.local_frame)

                
    #                print 'nthSec:{}'.format(nSecond)
                timeElapsed = 0
                startTime = datetime.now()

           
            if position != "unknown": 
                cv2.line        (bw_copy,   (cX,cY),            (int(w/2),cY),(0,255,0),2)
                if dist >= 100:dist = 100
                elif dist <= -100:dist = -100
                p.plot(dist)
                
                plot_canvas = np.uint8(p.plot_canvas)
            elif position == "unknown": 
                dist = "unknown"
                text = "idle"
            

            p_spd.plot(velo/2)
            velo_canvas = np.uint8(p_spd.plot_canvas)


           # print(vx)
            #if dist != None:
            
            #pp.plot(vehicle.velocity[0])
            

            #lot_canvas_1 = np.uint8(pp.plot_canvas)
            #print(vehicle.velocity[0])
            final = cv2.vconcat([frame,plot_canvas,bw_copy,velo_canvas])

            #print( vehicle.airspeed)

            cv2.rectangle   (final, (0,0),  (w,h),       (255,0,0), 5)                          # visual 
            cv2.rectangle   (final, (0,h),(w,h+200),       (0,157,0), 5) 
            cv2.rectangle   (final, (0,h+200),(1280,+200 +800),      (0,0,255), 5)

            #cv2.putText    (final,str(timeElapsed),               (200,523),      cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),        2,cv2.LINE_AA)
            cv2.putText     (final, 'Camera',               (1080,h-20),      cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,137,137),  2,cv2.LINE_AA)
            cv2.putText     (final, 'Graph (allignment)',   (820,h+180),  cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,157,0),      2,cv2.LINE_AA)
            cv2.putText     (final, 'Visualization',        (970,h+180+800),     cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),      2,cv2.LINE_AA)
            cv2.putText     (final, 'Position: '+(position),(30,h+180),       cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),        2,cv2.LINE_AA)
            cv2.putText     (final, 'y = ' + str(dist),     (30,840),       cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),        2,cv2.LINE_AA)
            cv2.putText     (final, 'y = ' + str(dist),     (30,840),       cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),        2,cv2.LINE_AA)
            
            
            if resume == 1:
                try:
                    if position == "unknown":
                        text = "unknown"
                        confirm_interval = 0
                        confirm_time = 0
                        
                    # if point == 0:
                    #     dist = 0 
                    #     speed = pid(dist)
                    #     send_local_velocity( 0,0,0,0)                                                  # flying function
                    # else:            
                    #     speed = pid(dist)
                    #     #print(speed)
                    #     send_local_velocity( 0,0, 0,speed)
                    if return_confirmation == 0:
                        if point == 1:

                            confirm_interval = 0
                            confirm_time = 0

                            speed = pid(dist)
                            if dist > 10 or dist < -10 :
                                
                                text = ("[trip 1] alligning")
                                velo = smooth_func(100)
                                send_local_velocity(velo*max_speed, speed, 0,0)
                            elif dist >= -10 and dist <= 10:  
                                text = ("[trip 1] start")
                                velo = smooth_func(100)
                                send_local_velocity(velo*max_speed,speed, 0,0)
            
                
                        elif point == 2 :    

                            confirm_interval = 0
                            confirm_time = 0

                            speed = pid(dist)
                            text = ("[trip 1] following")
                            velo = smooth_func(100)
                            send_local_velocity(velo*max_speed,speed, 0,0)
                        elif point == 3 :     
                            #send_local_velocity(0,0, 0,0)
                            
                            if confirm_interval >= 10:
                                text = ('Confirmed')
                                return_confirmation =1
                                confirm_interval = 0
                                confirm_time = 0
                                #total_confirm_time = 0 
                            elif confirm_time == 0:
                                send_local_velocity(0,0, 0,0)
                                #total_confirm_time +=1
                                confirm_time = nSecond
                                #time.sleep(1)
                                text = str("[trip 1] confirming: " + str(confirm_interval))
                            else:
                                confirm_interval = nSecond-confirm_time
                                text = str("[trip 1] confirming: " + str(confirm_interval))



                    #--------------trip 2-------------------#

                    elif return_confirmation == 1:
                        #print('returning...')
                        if point == 3 :

                            confirm_interval = 0
                            confirm_time = 0

                            speed = pid(dist)
                            if dist > 10 or dist < -10 :
                                
                                text = ("[trip 2] alligning")
                        
                                send_local_velocity(0, speed, 0,0)
                            elif dist >= -10 and dist <= 10:  
                                text = ("[trip 2] starting")
                                send_local_velocity(-0.11,speed, 0,0)
                        elif point == 2 :       

                            confirm_interval = 0
                            confirm_time = 0

                            speed = pid(dist)
                            text =  ("[trip 2] following")
                            send_local_velocity(-0.3,speed, 0,0)
                        elif point == 1 :     
                            send_local_velocity(0,0, 0,0)
                            if confirm_interval >= 10:
                                text = ('Confirmed')
                                return_confirmation =2
                            elif confirm_time  == 0:
                                send_local_velocity(0,0, 0,0)
                                #total_confirm_time +=1
                                confirm_time = nSecond
                                #time.sleep(1)
                                text = str("[trip 2] confirming: " + str(confirm_interval))
                            else:
                                confirm_interval = nSecond-confirm_time
                                text = str("[trip 2] confirming: " + str(confirm_interval))
                            # if total_confirm_time >= 10:
                            #     text = ('Confirmed Goodbye')
                            #     return_confirmation =2
                            # else:
                            #     send_local_velocity(0,0, 0,0)
                            #     total_confirm_time +=1
                            #     time.sleep(1)
                            #     text = str("<trip 2> confirming: " + str(11-total_confirm_time))
                    elif return_confirmation ==2:
                        print('SUCESS!')
                        break
                    
                    status = "Active"
                except ValueError:pass
            else: 
                status = "Pause"
                spd = 0
                velo = smooth_func(0)
                if velo <= 20:pass
                else:send_local_velocity(velo*max_speed,speed, 0,0)
            if status == "Pause":
                cv2.rectangle(final,   (1060,2),   (1275,110),             (0,255,0),    -1) 
                cv2.putText  (final, status,((1075,80)),       cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0),        3,cv2.LINE_AA)
            elif status == "Active":
                cv2.rectangle(final,   (1060,2),   (1275,110),             (0,0,255),    -1)     
                cv2.putText     (final, status,((1080,80)),       cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),        3,cv2.LINE_AA)
            cv2.rectangle(final,   (1060,2),   (1275,109),             (0,0,0),    2) 
            cv2.putText    (final,'Mode: ' + text,               (400,840),      cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),        2,cv2.LINE_AA)
            
            #cv2.imshow('',plot_canvas)
            cv2.imshow("Show", final)
            if cv2.waitKey(1) & 0xFF == 112:
                if resume == 0:
                    resume = 1
                    return_confirmation = 0
                    print('resume')
                elif resume ==1:
                    
                    resume = 0
                    
                    #send_local_velocity(0,0, 0,0)
                    print('pause')
                    text = "idle"
            
            elif cv2.waitKey(1) & 0xFF == 81 :
                break
            
        elif img is None:
            frame = 255*np.ones((1280,800,3))
    except KeyboardInterrupt: 
        send_local_velocity(0,0, 0,0)
        break


print("GOODBYE")
print('time --> ' + str(nSecond) + ' seconds')
cv2.destroyAllWindows()