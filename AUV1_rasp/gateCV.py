from __future__ import print_function
import numpy as np
import cv2
import math
import cv2.cv as cv
import time
import urllib
import socket

boxes = []
def makeConnection()
	server = socket.socket()
	server.bind(('0.0.0.0',5000))
	server.listen(2)
	conn,addr = server.accept()
	print ('user connected from ',addr)
	stream=urllib.urlopen('http://192.168.1.101:7000/?action=stream')

bytes=''

# Read a feed from PiCam.
def feedread():
    global stream
    global bytes    
    while True:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv.CV_LOAD_IMAGE_COLOR)
            return i
            
            
# Define the movemet of bot wrt the position and dimensions of the image of the gate in a feed.
def movement(a,b,p_x,p_y,q_x,q_y,r_x,r_y,s_x,s_y,cent_frame_x,cent_frame_y):
        t=0
        n=1
        l,r,t,b0,h,v=0,0,0,0,0,0
        b1 = (r_x - q_x + s_x - p_x)/2.0
        a1 = (p_y - q_y + s_y - r_y)/2.0
        cent_door_x = (p_x + q_x + r_x + s_x)/4
        cent_door_y = (p_y + q_y + r_y + s_y)/4
        rot_h = cent_frame_x - cent_door_x
        rot_v = cent_frame_y - cent_door_y
        rot_sign_h = 0
        rot_sign_v = 0
        if rot_h>=0:
                rot_sign_h=1
        else:
                rot_sign_h = -1

        if rot_v>=0:
                rot_sign_v=1
        else:
                rot_sign_v = -1

        if abs(rot_v)>abs(rot_h):
                if rot_sign_v == 1:
                        print("rotate UP" + str(rot_v),end='    ')
                if rot_sign_v == -1:
                        print("rotate Down" + str(rot_v),end='    ')
        else:
                if rot_sign_h== 1:
                        print("rotate Left " + str(rot_h),end='    ')
                if rot_sign_h == -1:
                        print("rotate Right " + str(rot_h),end='    ')
        if(b1*1.0/a1*1.0)>(b*1.0/a*1.0):
                print("Move up/down "+str((b1*1.0/a1*1.0)/(b*1.0/a*1.0)))
        else:
                print("Move Right/Left "+str((b1*1.0/a1*1.0)/(b*1.0/a*1.0)))



# Store mean ad standard deviation of the color value of the gate to be detected

def on_mouse(event, x, y, flags, params):        
        if event == cv2.EVENT_LBUTTONDOWN:
                print
                'Start Mouse Position: ' + str(x) + ', ' + str(y)
                sbox = [x, y]
                boxes.append(sbox)
                
        elif event == cv2.EVENT_LBUTTONUP:
                'End Mouse Position: ' + str(x) + ', ' + str(y)
                ebox = [x, y]
                boxes.append(ebox)
                print(boxes)
                crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                cv2.imshow('crop', crop)
                key = cv2.waitKey(0) & 0xFF
                print (key)
                if ord('r') == key:
                        a = cv2.mean(crop)
                        print(a)
                        mean_color = [0, 0, 0]
                        mean_color[0] = (a[0])
                        mean_color[1] = (a[1])
                        mean_color[2] = (a[2])
                        std_color = [0, 0, 0]
                        std_color[0] = int(crop[:, :, 0].std() + 5)
                        std_color[1] = int(crop[:, :, 1].std() + 10)
                        std_color[2] = int(crop[:, :, 2].std() + 10)
                        np.save('mean_save', mean_color)
                        np.save("std_save", std_color)

def setGateParams():
	image_size = feedread().shape[:2]
	feed_center_x = int(image_size[1]/2)
	feed_center_y = int(image_size[0]/2)

	boxes = []
	count = 0
	while 1:
	        count += 1
	        img = feedread()
	        img = cv2.resize(img, None, fx=1, fy=1,interpolation=cv2.INTER_AREA)

	        cv2.namedWindow('real image')
	        cv.SetMouseCallback('real image', on_mouse, 0)
	        cv2.imshow('real image', img)
	        k = cv2.waitKey(1) & 0xFF
	        print(k)
	        if k==27:
	                cv2.destroyAllWindows()
	                break
	        elif count >= 2:
	                if cv2.waitKey(0)& 0xFF == 27:
	                        cv2.destroyAllWindows()
	                        break
	                count = 0


def startCV():
	mean_color = np.load("mean_save.npy")
	std_color = np.load("std_save.npy")

	print(mean_color)
	print(std_color)

	minLineLength = 120
	maxLineGap = 1

	minThresh = 200
	maxThresh = 220

	kernal_morph1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	kernal_morph11 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	kernal_morph2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
	kernal_morph22 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

	base_speed_min =30
	base_speed_max =30
	base_cons = 30

	motor1_speed = 0
	motor2_speed = 0
	error = 0
	last_error = 0
	integral = 0


	kp1 = 30.0
	kp_min = 1.0
	kp_max = 100.0
	kp_cons = 50.0
	kd = 0
	ki = 0

	flag=0

	speed_error = 0

	while(1):
	        ts=time.time()
	        im = feedread()
	        te=time.time()
	        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	        a = mean_color - 2*std_color
	        b = mean_color + 2*std_color
	        mask = cv2.inRange(hsv, a, b)
	        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_morph1)
	        mask_open_close = cv2.dilate(mask_open, kernal_morph2, iterations=1)
	        mask_open_1 = cv2.morphologyEx(mask_open_close, cv2.MORPH_OPEN, kernal_morph11)
	        mask_open_close_1 = cv2.dilate(mask_open_1, kernal_morph22, iterations=1)
	        res = cv2.bitwise_and(im, im, mask=mask_open_close_1)


	        cnts,_ = cv2.findContours(mask_open_close_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	        all_cnts_area=[]
	        for c in cnts:
	                a=cv2.contourArea(c)
	                all_cnts_area.append(a)
	        all_cnts_area=np.array(all_cnts_area)
	        if (all_cnts_area.size==0):
	                
	                continue
	        maxArea_index=np.nonzero(all_cnts_area==max(all_cnts_area))[0][0]
	        our_contour=cnts[maxArea_index]
	        epsilon= 0.07*cv2.arcLength(our_contour, True)
	        final_contour=cv2.approxPolyDP(our_contour, epsilon , True)
	        final_contour=cv2.convexHull(final_contour)
	        cv2.drawContours(res, our_contour, -1, (255,0,0), 3)
	        cv2.drawContours(res, final_contour, -1, (255,255,0), 3)
	        cv2.imshow('res', res)
	        key_cv2 = cv2.waitKey(1) & 0xFF
	        if key_cv2 == ord('q'):
	                cv2.destroyAllWindows()
	                # feed.close()
	                break
	        
	        if len(final_contour)!=4:
	                if flag!=0:
	                        flag=2
	                else:
	                        continue
	        else:
	                flag=1
	        if flag==1:
	                cv2.drawContours(im, final_contour, -1, (255,0,0), 3)

	                cent_frame_y , cent_frame_x=img.shape[:2][0]/2 , img.shape[:2][1]/2
	                cent_cont_y=(final_contour[0][0][0]+final_contour[1][0][0]+final_contour[2][0][0]+final_contour[3][0][0])/4
	                cent_cont_x=(final_contour[0][0][1]+final_contour[1][0][1]+final_contour[2][0][1]+final_contour[3][0][1])/4

	                box=np.array([[final_contour[0][0][0],final_contour[0][0][1]],[final_contour[1][0][0],final_contour[1][0][1]],[final_contour[2][0]              [0],final_contour[2][0][1]],[final_contour[3][0][0],final_contour[3][0][1]]])

	                left_ind=np.nonzero(box[:,1]<cent_cont_x)
	                right_ind=np.nonzero(box[:,1]>cent_cont_x)
	                        
	                left_pts = box[left_ind]
	                right_pts = box[right_ind]

	                tl_ind = np.nonzero(left_pts[:,0]==min(left_pts[:,0]))[0][0]
	                tr_ind = np.nonzero(right_pts[:,0]==min(right_pts[:,0]))[0][0]
	                bl_ind = np.nonzero(left_pts[:,0]==max(left_pts[:,0]))[0][0]
	                br_ind = np.nonzero(right_pts[:,0]==max(right_pts[:,0]))[0][0]

	                tl_pts = tuple(left_pts[tl_ind])
	                bl_pts = tuple(left_pts[bl_ind])
	                tr_pts = tuple(right_pts[tr_ind])
	                br_pts = tuple(right_pts[br_ind])

	                tr_pts, bl_pts=bl_pts, tr_pts
	                font = cv2.FONT_HERSHEY_SIMPLEX
	                cv2.putText(im,'TL',tl_pts, font, 1,(250,250,250),2)
	                cv2.putText(im,'TR',tr_pts, font, 1,(250,250,250),2)
	                cv2.putText(im,'BL',bl_pts, font,1,(250,250,250),2)
	                cv2.putText(im,'BR',br_pts, font, 1,(250,250,250),2)

	                p_x, p_y, q_x, q_y, r_x, r_y, s_x, s_y = bl_pts[0], bl_pts[1], tl_pts[0], tl_pts[1], tr_pts[0], tr_pts[1], br_pts[0], br_pts[1]
	                gate_area = cv2.contourArea(final_contour)
	                b,a=32,51
	                movement(a, b, p_x, p_y, q_x, q_y, r_x, r_y, s_x, s_y, cent_frame_x, cent_frame_y)
	                cent_door_x = (p_x + q_x + r_x + s_x) / 4
	                cent_door_y = (p_y + q_y + r_y + s_y) / 4
	                error = cent_frame_x - cent_door_x
	                error = error * 2.0 / image_size[0]
	                
	                                
	        cent_frame_x, cent_frame_y=im.shape[:2][1]/2,im.shape[:2][0]/2
	        frame_area = image_size[0]*image_size[1]
	        if gate_area!=0:
	            kp=(frame_area/gate_area)*kp_cons
	        else:
	            kp=kp_max
	        if kp>kp_min and kp<kp_max and flag==1:
	            speed_error = kp1 * error + ki * integral - kd * (error - last_error)
	        elif flag==1:
	            kp=kp_max
	            speed_error = kp1 * error + ki * integral - kd * (error - last_error)
	        elif flag==2:
	            kp=kp_min
	            speed_error = kp1 * error + ki * integral - kd * (error - last_error)

	        print ("speed_error = " + str(speed_error))
	        integral += error
	        last_error = error

	        if gate_area!=0:
	            base_speed = (frame_area/gate_area)*base_cons
	        else:
	            base_speed = base_speed_max
	        if base_speed > base_speed_min and base_speed<base_speed_max and flag==1:
	            motor1_speed = base_speed - speed_error
	            motor2_speed = base_speed + speed_error
	        elif flag==1:
	            base_speed = base_speed_max
	            motor1_speed = base_speed - speed_error
	            motor2_speed = base_speed + speed_error
	        elif flag==2:
	            base_speed  = base_speed_min
	            motor1_speed = base_speed - speed_error
	            motor2_speed = base_speed + speed_error
	        print ("left " + str(motor1_speed) + "right " + str(motor2_speed))
	        
	        conn.send(str([motor1_speed,motor2_speed,speed_error]))
	        
	        print ("fps: "+str(1.0/(te-ts)))
	        print("\n")
	        np.save('im',im)
	   		yield motor1_speed, motor2_speed, speed_error

	cv2.destroyAllWindows()

if __name__ == "__main__":
	makeConnection()
	setGateParams()
	startCV()