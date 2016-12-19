import numpy as np
import cv2
import math
# import RPi.GPIO as GPIO
from time import sleep
from matplotlib import pyplot as plt
#from motor_movement import *
feed = cv2.VideoCapture(0)
image_size = feed.read()[1].shape
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
            std_color[0] = int(crop[:, :, 0].std() + 5)
            std_color[1] = int(crop[:, :, 1].std() + 10)
            std_color[2] = int(crop[:, :, 2].std() + 10)
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
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
        count = 0
#################################################################################################3
mean_color = np.load("mean_save.npy")
std_color = np.load("std_save.npy")
print mean_color
print std_color

b, c = np.zeros(image_size,dtype='float'), np.zeros(image_size[:2],dtype='uint8')
while 1:
    a = feed.read()[1]
    cv2.imshow("a", a)
    b[..., 0] = abs(1-abs(cv2.cvtColor(a,cv2.COLOR_BGR2HSV)[...,0]-mean_color[0])/90)
    b[..., 1] = abs(1 - abs(cv2.cvtColor(a, cv2.COLOR_BGR2HSV)[..., 1] - mean_color[1]) / 128)
    b[..., 2] = abs(1 - abs(cv2.cvtColor(a, cv2.COLOR_BGR2HSV)[..., 2] - mean_color[2]) / 128)
    #c = cv2.cvtColor(b, cv2.COLOR_HSV2BGR)
    c = (b[...,0]*95
         + b[...,1]*75
         + b[...,2]*85
         ).astype('uint8')
    cv2.imshow("c", c)
    hist = cv2.calcHist([c], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.show()
    k = cv2.waitKey(1)
    if k==ord('q'):
        cv2.destroyAllWindows()
        break
feed.release()
