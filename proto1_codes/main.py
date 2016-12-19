import numpy as np
import cv2
import math
# import RPi.GPIO as GPIO
from time import sleep
# from motor_movement import *

feed = cv2.VideoCapture(0)
image_size = feed.read()[1].shape[:2]
feed_center_x = int(image_size[1]/2)
feed_center_y = int(image_size[0]/2)
print image_size
'''
lower_color = np.array([0,0,0])
mean_color = np.array([110, 70, 23])
std_color = np.array([4 + 10, 13 + 30, 12 + 50])
upper_color = np.array([20, 20, 20])
'''

########################################################################################
import cv2.cv as cv

from time import time
boxes = []

def on_mouse(event, x, y, flags, params):
    # global img
    t = time()

    if event == cv.CV_EVENT_LBUTTONDOWN:
        print
        'Start Mouse Position: ' + str(x) + ', ' + str(y)
        sbox = [x, y]
        boxes.append(sbox)
        # print count
        # print sbox

    elif event == cv.CV_EVENT_LBUTTONUP:
        print
        'End Mouse Position: ' + str(x) + ', ' + str(y)
        ebox = [x, y]
        boxes.append(ebox)
        print boxes
        crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cv2.imshow('crop', crop)
        k = cv2.waitKey(0)
        if ord('r') == k:
            a = cv2.mean(crop)
            print a
            mean_color = [0, 0, 0]
            mean_color[0] = (a[0])
            mean_color[1] = (a[1])
            mean_color[2] = (a[2])
            std_color = [0, 0, 0]
            std_color[0] = int(crop[:, :, 0].std())
            std_color[1] = int(crop[:, :, 1].std())
            std_color[2] = int(crop[:, :, 2].std())
            np.save('mean_save', mean_color)
            np.save("std_save", std_color)

count = 0
while 1:
    count += 1
    img = feed.read()[1]
    # img = cv2.blur(img, (3,3))
    img = cv2.resize(img, None, fx=1, fy=1)

    cv2.namedWindow('real image')
    cv.SetMouseCallback('real image', on_mouse, 0)
    cv2.imshow('real image', img)

    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        break
    elif count >= 5:
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
        count = 0

#################################################################################################3

mean_color = np.load("mean_save.npy")
std_color = np.load("std_save.npy")

print mean_color
print std_color


from cam_func_lib import *
from detectmoments import small, sort


#GPIO.setmode(GPIO.BCM)
Motor1 = 02
Motor2 = 03
Motor1P = 27
Motor1N = 22
Motor2P = 23
Motor2N = 24

base_speed = 50

motor1_speed = 0
motor2_speed = 0
error = 0
last_error = 0
integral = 0

max_speed = 100
min_speed = -20

W = 29

kp = 50.0
kd = 0
ki = 0

speed_error = 0
base_speed1 = base_speed
base_speed2 = base_speed
'''
activate_pins(Motor1, Motor2, Motor1P, Motor1N, Motor2P, Motor2N)
pwm1, pwm2 = start_pwm(base_speed, Motor1, Motor2)
'''
while 1:
    _, img = feed.read()
    im = img.copy()
    gray = color_crop(im, mean_color, std_color)
    CX, CY = gray.shape
    CX /= 2
    gray = cv2.GaussianBlur(gray, (3, 3), 3)
    edged = cv2.Canny(gray, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = cv2.convexHull(cnts[0])
    else:
        continue

    cv2.drawContours(img, [cnts], -1, (0, 0, 0), 10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = color_crop(im, mean_color, std_color)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh", thresh)
    edged = cv2.Canny(thresh, 10, 250)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    im2 = motion_bw(feed)
    closing1 = color_crop(im, mean_color, std_color)
    # closing1 = closing + im2
    cv2.imshow("closing1", closing1)
    (cnts, _) = cv2.findContours(closing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print len(approx)
        if len(approx) == 4:
            cv2.drawContours(im, approx, -1, (0, 255, 0), 2)
            center = ((approx[0][0][0] + approx[1][0][0] + approx[2][0][0] + approx[3][0][0]) / 4, (approx[0][0][1] +
                                                                                                    approx[1][0][1] +
                                                                                                    approx[2][0][1] +
                                                                                                    approx[3][0][
                                                                                                        1]) / 4)
            cv2.circle(im, center, 10, (255, 0, 0), 20)
            # if np.abs(approx[1][0][0] - approx[0][0][0]) < 5 and np.abs(approx[2][0][0] - approx[3][0][0]) < 5 \
            # and np.abs(approx[2][0][1] - approx[1][0][1]) < 5 and np.abs(approx[3][0][1] - approx[0][0][1]) < 5:
            points = sort(approx, center)
            h = ((points[0][0][1] - points[1][0][1]) + (points[3][0][1] - points[2][0][1])) / 2.0
            w = ((points[2][0][0] - points[1][0][0]) + (points[3][0][0] - points[0][0][0])) / 2.0

            print h, w, W, points[0][0], points[1][0], points[2][0], points[3][0]
            if w <= W:
                theta2 = np.arccos(w / W) * 180 / np.pi
            else:
                theta2 = 0
            cv2.putText(im, str(theta2), (int(CX / 4), int(3 * CY / 4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("IM", im)
    '''
    error = camera_func(feed, mean_color, std_color, feed_center_x, feed_center_y)
    if error == 0x9557705979 or error == 0x9456484690:
        #stop(Motor1P, Motor1N, Motor2P, Motor2N)
        print 'stop'
        #sleep(0.5)
        continue
    error = error*2.0/image_size[1]
    print error
    speed_error = kp * error + ki * integral - kd * (error - last_error)
    print "speed_error = " + str(speed_error)
    integral += error
    last_error = error
    motor1_speed = base_speed1 - speed_error
    motor2_speed = base_speed2 + speed_error

    print "left " + str(motor1_speed) + "right " + str(motor2_speed)
    if motor1_speed>max_speed:
        motor1_speed = max_speed
    elif motor1_speed<min_speed:
        motor1_speed = min_speed
    if motor2_speed>max_speed:
        motor2_speed = max_speed
    elif motor2_speed<min_speed:
        motor2_speed = min_speed
    '''
    '''
    if motor1_speed > 0 and motor2_speed > 0:
        forward(Motor1P, Motor1N, Motor2P, Motor2N)
    elif motor1_speed > 0 and motor2_speed < 0:
        motor2_speed = -motor2_speed
        right(Motor1P, Motor1N, Motor2P, Motor2N)
    elif motor1_speed < 0 and motor2_speed > 0:
        motor1_speed = -motor1_speed
        left(Motor1P, Motor1N, Motor2P, Motor2N)

    set_motor_speed(pwm1, pwm2, motor1_speed, motor2_speed)
    '''
    stop_key = cv2.waitKey(10)
    if stop_key == ord('x'):
        cv2.destroyAllWindows()
        break