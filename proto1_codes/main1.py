# update kp, speed_error
# make a flag

# Import necessary packages and methods
from __future__ import print_function
from time import time
import cv2
import cv2.cv as cv
import numpy as np

# Define feed that takes input from video camera and get its max size
feed = cv2.VideoCapture(0)
image_size = feed.read()[1].shape[:2]
boxes = []
fourcc = cv2.cv.CV_FOURCC('i','Y','U','V')
out = cv2.VideoWriter('myVideo1.avi',fourcc,20.0,(640,480))

def movement(a, b, p_x, p_y, q_x, q_y, r_x, r_y, s_x, s_y, cent_frame_x, cent_frame_y):
    """"
     This function takes Bottom Left (P), Top Left (Q), Top Right (R) and Bottom Right (S) points along with the
     actual width and height of the gate in pixels and center of the image frame.
     It finds the width and height in pixels of the image and finds the center of the gate using P, Q, R and S.
     The distance gap of center of frame and center of gate gives extent of required vertical and horizontal motion.
     The comparison of aspect ratios of original gate and the image of the gate gives extent of required vertical and
     horizontal rotation.
     :param a: gate height
     :param b: gate width
     :param p_x: Bottom Left's X
     :param p_y: Bottom Left's Y
     :param q_x: Top Left's X
     :param q_y: Top Left's Y
     :param r_x: Top Right's X
     :param r_y: Top Right's Y
     :param s_x: Bottom Right's X
     :param s_y: Bottom Right's Y
     :param cent_frame_x: Frame center's X
     :param cent_frame_y: Frame center's Y
     :return: None
    """
    t = 0
    n = 1
    l, r, t, b0, h, v = 0, 0, 0, 0, 0, 0
    b1 = (r_x - q_x + s_x - p_x)/2.0
    a1 = (p_y - q_y + s_y - r_y)/2.0
    cent_door_x = (p_x + q_x + r_x + s_x)/4
    cent_door_y = (p_y + q_y + r_y + s_y)/4
    rot_h = cent_frame_x - cent_door_x
    rot_v = cent_frame_y - cent_door_y
    rot_sign_h = 0
    rot_sign_v = 0
    if rot_h >= 0:
        rot_sign_h = 1
    else:
        rot_sign_h = -1

    if rot_v >= 0:
        rot_sign_v = 1
    else:
        rot_sign_v = -1

    # vertical shift
    if abs(rot_v) > abs(rot_h):
        if rot_sign_v == 1:
            print("rotate UP" + str(rot_v),end='    ')
        if rot_sign_v == -1:
            print("rotate Down" + str(rot_v),end='    ')
    else:
        if rot_sign_h == 1:
            print("rotate Left " + str(rot_h),end='    ')
        if rot_sign_h == -1:
            print("rotate Right " + str(rot_h),end='    ')
    if(b1*1.0/a1*1.0) > (b*1.0/a*1.0):
        print("Move up/down "+str((b1*1.0/a1*1.0)/(b*1.0/a*1.0)))
    else:
        print("Move Right/Left "+str((b1*1.0/a1*1.0)/(b*1.0/a*1.0)))


def on_mouse(event, x, y, flags, params):
    """
    This function is called on mouse click operation and takes a crop of initial image from a person to save
    the mean color and standard devation of the color from the mean
    :param event: Mouse click event
    :param x: X-coordinate of click
    :param y: Y-coordinate of click
    :param flags:
    :param params:
    :return: None
    """
    t = time()
    
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        sbox = [x, y]
        boxes.append(sbox)

    elif event == cv.CV_EVENT_LBUTTONUP:
        print('End Mouse Position: ' + str(x) + ', ' + str(y))
        ebox = [x, y]
        boxes.append(ebox)
        print(boxes)
        crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cv2.imshow('crop', crop)
        k = cv2.waitKey(0)
        if ord('r') == k:
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

# Take mean color and standard deviation of the color of the gate
count = 0
while 1:
    count += 1
    img = feed.read()[1]
    img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)  # Zooming out of image

    cv2.namedWindow('real image')
    cv.SetMouseCallback('real image', on_mouse, 0)
    cv2.imshow('real image', img)

    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        break
    elif count >= 50:  # Max open windows = 50
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
        count = 0

#################################################################################################3

# Load mean color and std color
mean_color = np.load("mean_save.npy")
std_color = np.load("std_save.npy")

print(mean_color)
print(std_color)

# Set parameters for Canny Edge detection
minThresh = 200
maxThresh = 220

kernal_morph1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kernel for first image opening
kernal_morph11 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Kernel for first image closing

kernal_morph2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))  # Kernel for second image opening
kernal_morph22 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))  # Kernel for second image closing


# Motor1 = 02
# Motor2 = 03
# Motor1P = 27
# Motor1N = 22
# Motor2P = 23
# Motor2N = 24

base_speed_min = -20
base_speed_max = 100
base_cons = 50

motor1_speed = 0
motor2_speed = 0

error = 0
last_error = 0
integral = 0

kp_min = 1.0
kp_max = 100.0
kp_cons = 50

kd = 0
ki = 0

flag = 0  # flag can take 3 values: 0, 1, 2. 0 means 'gate never detected',
            # 1 means 'gate has been detected in current frame',
            # 2 means 'gate was earlier detected but couldn't be detected in current image'

speed_error = 0

# activate_pins(Motor1, Motor2, Motor1P, Motor1N, Motor2P, Motor2N)
# pwm1, pwm2 = start_pwm(base_speed, Motor1, Motor2)

while True:
    # Read image
    im = feed.read()[1]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    a = mean_color - 3*std_color
    b = mean_color + 3*std_color

    # Mask the object of color of gate : OBJECT DETECTION
    mask = cv2.inRange(hsv, a, b)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_morph1)
    mask_open_close = cv2.dilate(mask_open, kernal_morph2, iterations=1)
    mask_open_1 = cv2.morphologyEx(mask_open_close, cv2.MORPH_OPEN, kernal_morph11)
    mask_open_close_1 = cv2.dilate(mask_open_1, kernal_morph22, iterations=3)
    res = cv2.bitwise_and(im, im, mask=mask_open_close_1)

    # Find all contours in image and use contour with maximum area to remove noise and extract gate
    cnts, _ = cv2.findContours(mask_open_close_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts_area = []
    for c in cnts:
        a=cv2.contourArea(c)
        all_cnts_area.append(a)
    all_cnts_area = np.array(all_cnts_area)
    if all_cnts_area.size == 0:
        continue
    maxArea_index=np.nonzero(all_cnts_area == max(all_cnts_area))[0][0]
    our_contour=cnts[maxArea_index]

    # Use Approximation and find Convex Hull to get four points of bounding Quadrilateral
    epsilon= 0.07*cv2.arcLength(our_contour, True)
    final_contour=cv2.approxPolyDP(our_contour, epsilon , True)
    final_contour=cv2.convexHull(final_contour)
    cv2.drawContours(res, our_contour, -1, (255,0,0), 3)
    cv2.drawContours(res, final_contour, -1, (255,255,0), 3)
    cv2.imshow('res', res)

    # Ends the program
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == ord('q'):
        cv2.destroyAllWindows()
        break

    # Take another frame image if 4 points not found due to noise
    if len(final_contour) != 4:
        if flag != 0:
            flag = 2
        else:
            continue
    else:
        flag = 1

    if flag == 1:
        cv2.drawContours(im, final_contour, -1, (255,0,0), 3)

        # Set center of frame and gate
        cent_frame_y, cent_frame_x=img.shape[:2][0]/2 , img.shape[:2][1]/2
        cent_cont_y = (final_contour[0][0][0]+final_contour[1][0][0]+final_contour[2][0][0]+final_contour[3][0][0])/4
        cent_cont_x = (final_contour[0][0][1]+final_contour[1][0][1]+final_contour[2][0][1]+final_contour[3][0][1])/4

        # Find Top left(Q), Top right(R), Bottom Left(P) and Bottom Right(S) points
        box = np.array([[final_contour[0][0][0], final_contour[0][0][1]], [final_contour[1][0][0], final_contour[1][0][1]],
                        [final_contour[2][0][0], final_contour[2][0][1]], [final_contour[3][0][0],final_contour[3][0][1]]])

        left_ind = np.nonzero(box[:, 1] < cent_cont_x)
        right_ind = np.nonzero(box[:, 1] > cent_cont_x)
    
        left_pts = box[left_ind]
        right_pts = box[right_ind]

        tl_ind = np.nonzero(left_pts[:, 0] == min(left_pts[:, 0]))[0][0]
        tr_ind = np.nonzero(right_pts[:, 0] == min(right_pts[:, 0]))[0][0]
        bl_ind = np.nonzero(left_pts[:, 0] == max(left_pts[:,0]))[0][0]
        br_ind = np.nonzero(right_pts[:, 0] == max(right_pts[:, 0]))[0][0]

        tl_pts = tuple(left_pts[tl_ind])
        bl_pts = tuple(left_pts[bl_ind])
        tr_pts = tuple(right_pts[tr_ind])
        br_pts = tuple(right_pts[br_ind])
        tr_pts, bl_pts = bl_pts, tr_pts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, 'TL', tl_pts, font, 1, (250, 250, 250), 2)
        cv2.putText(im, 'TR', tr_pts, font, 1, (250, 250, 250), 2)
        cv2.putText(im, 'BL', bl_pts, font, 1, (250, 250, 250), 2)
        cv2.putText(im, 'BR', br_pts, font, 1, (250, 250, 250), 2)
        p_x, p_y, q_x, q_y, r_x, r_y, s_x, s_y = bl_pts[0], bl_pts[1], tl_pts[0], tl_pts[1], tr_pts[0], tr_pts[1], br_pts[0], br_pts[1]

        # Gate Area
        gate_area = cv2.contourArea(final_contour)

        # Set actual width and height of image
        b, a = 32, 51

        # Define movement of robot according to the configuration of the image of gate and and actual gate
        movement(a, b, p_x, p_y, q_x, q_y, r_x, r_y, s_x, s_y, cent_frame_x, cent_frame_y)

        # Find distance gap between image center and frame center
        cent_door_x = (p_x + q_x + r_x + s_x) / 4
        cent_door_y = (p_y + q_y + r_y + s_y) / 4
        error = cent_frame_x - cent_door_x

        error = error * 2.0 / image_size[1]


    cent_frame_x, cent_frame_y = im.shape[:2][1]/2, im.shape[:2][0]/2

    # Set frame area
    frame_area = image_size[0] * image_size[1]

    # Set kp and find speed error
    if gate_area != 0:
        kp = (frame_area / gate_area) * kp_cons
    else:
        kp = kp_max
    if kp > kp_min and kp < kp_max and flag == 1:
        speed_error = kp * error + ki * integral - kd * (error - last_error)
    elif flag == 1:
        kp = kp_max
        speed_error = kp * error + ki * integral - kd * (error - last_error)
    elif flag == 2:
        kp = kp_min
        speed_error = kp * error + ki * integral - kd * (error - last_error)

    integral += error
    last_error = error

    # Set base speed and find motor speed
    base_speed = (frame_area / gate_area) * base_cons
    if base_speed > base_speed_min and base_speed < base_speed_max and flag == 1:
        motor1_speed = base_speed - speed_error
        motor2_speed = base_speed + speed_error
    elif flag == 1:
        base_speed = base_speed_max
        motor1_speed = base_speed - speed_error
        motor2_speed = base_speed + speed_error
    elif flag == 2:
        base_speed = base_speed_min
        motor1_speed = base_speed - speed_error
        motor2_speed = base_speed + speed_error
    '''
    if motor1_speed > max_speed:
        motor1_speed = max_speed
    elif motor1_speed < min_speed:
        motor1_speed = min_speed
    if motor2_speed > max_speed:
        motor2_speed = max_speed
    elif motor2_speed < min_speed:
        motor2_speed = min_speed

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
    cv2.line(im, (cent_frame_x, 0), (cent_frame_x, 2*cent_frame_y), (0, 0, 255), 1)
    cv2.line(im, (0, cent_frame_y), (2*cent_frame_x, cent_frame_y), (0, 0, 255), 1)
    cv2.line(im, (cent_door_x, cent_door_y), (cent_door_x, cent_frame_y), (255, 0, 0), 1)
    cv2.line(im, (cent_door_x, cent_door_y), (cent_frame_x, cent_door_y), (255, 0, 0), 1)
    cv2.line(im, (p_x, p_y), (q_x, q_y), (0, 255, 0), 2)
    cv2.line(im, (q_x, q_y), (r_x, r_y), (0, 255, 0), 2)
    cv2.line(im, (r_x, r_y), (s_x, s_y), (0, 255, 0), 2)
    cv2.imshow("img", im)
    out.write(im)

feed.release()
out.release()
cv2.destroyAllWindows()
