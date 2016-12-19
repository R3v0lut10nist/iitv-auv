import cv2
import numpy as np
#from back_analysis import *

def small(a, b, cent):
    if a[0] - cent[0] >= 0 > b[0] - cent[0]:
        return True
    if a[0] - cent[0] < 0 <= b[0] - cent[0]:
        return False
    if a[0] - cent[0] == 0 and b[0] - cent[0] == 0:
        if a[1] - cent[1] >= 0 or b[1] - cent[1] >= 0:
            return a[1] > b[1]
        return b[1] > a[1]

    det = (a[0] - cent[0]) * (b[1] - cent[1]) - (b[0] - cent[0]) * (a[1] - cent[1])
    if det < 0:
        return True
    if det > 0:
        return False

    d1 = (a[0] - cent[0]) * (a[0] - cent[0]) + (a[1] - cent[1]) * (a[1] - cent[1])
    d2 = (b[0] - cent[0]) * (b[0] - cent[0]) + (b[1] - cent[1]) * (b[1] - cent[1])
    return d1 > d2

# .........................................................................


def sort(l, c):
    for i in range(len(l)-1):
        j = i+1
        while j < len(l):
            if small(l[i][0], l[j][0], c):
                t = tuple(l[j][0])
                l[j][0] = l[i][0]
                l[i][0] = list(t)
            j += 1
    return l

# cap = cv2.VideoCapture(0)
# kernel = np.ones((3, 3), np.uint8)
# fourcc = cv2.cv.CV_FOURCC('i','Y', 'U', 'V')
# out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
# f = 3.92
# # w = 7
# ar = 12.4/7.0
#
# while True:
#     _, img = cap.read()
#     im = img.copy()
#     gray = img[:, :, 1]
#     CX, CY = gray.shape
#     CX /= 2
#
#     '''
#     gray = cv2.GaussianBlur(gray, (3, 3), 3)
#     edged = cv2.Canny(gray, 10, 250)
#     cv2.imshow("edge", gray)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#     (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if cnts:
#         cnts = cv2.convexHull(cnts[0])
#     else:
#         continue
#
#     cv2.drawContours(img, [cnts], -1, (100, 100, 100))
#     '''
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow("thresh", thresh)
#     edged = cv2.Canny(thresh, 10, 250)
#     closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#     #closing1 = im2 + closing
#     (cnts, _) = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         print len(approx)
#         if len(approx) == 4:
#             cv2.drawContours(im, approx, -1, (0, 255, 0), 2)
#             center = ((approx[0][0][0]+approx[1][0][0]+approx[2][0][0]+approx[3][0][0])/4, (approx[0][0][1] +
#                                                                                             approx[1][0][1] +
#                                                                                             approx[2][0][1] +
#                                                                                             approx[3][0][1])/4)
#             cv2.circle(im, center, 0, (255, 0, 0), 2)
#             # if np.abs(approx[1][0][0] - approx[0][0][0]) < 5 and np.abs(approx[2][0][0] - approx[3][0][0]) < 5 \
#             #         and np.abs(approx[2][0][1] - approx[1][0][1]) < 5 and np.abs(approx[3][0][1] - approx[0][0][1]) < 5:
#             points = sort(approx, center)
#             h = ((points[0][0][1] - points[1][0][1]) + (points[3][0][1] - points[2][0][1])) / 2.0
#             w = ((points[2][0][0] - points[1][0][0]) + (points[3][0][0] - points[0][0][0])) / 2.0
#             W = int(h/ar)
#             print h, w, W, points[0][0], points[1][0], points[2][0], points[3][0]
#             if w <= W:
#                 theta2 = np.arccos(w/W)*180/np.pi
#             else:
#                 theta2 = 0
#             cv2.putText(im, str(theta2), (int(CX / 4), int(3 * CY / 4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#
#     # #
#     # # thresh = cv2.dilate(thresh, kernel, iterations=12)
#     # # # cv2.imshow("kjbb", thresh)
#     # # contours, _ = cv2.findContours(thresh, 1, 2)
#     # # contours = cv2.convexHull(contours[0])
#     # # M = cv2.moments(contours)
#     # # cx = int(M['m10']/M['m00'])
#     # # cy = int(M['m01']/M['m00'])
#     # # cv2.drawContours(img, contours, -1, (0,0,0), 2)
#     # # cv2.circle(img, (cx, cy), 2, (255, 0, 0), 2)
#     # # x, y, w, h = cv2.boundingRect(contours)
#     # # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     # # theta1 = (abs(CX - cx)/100.0)*np.pi/180
#     # # W = h/ar
#     # # theta2 = np.arccos(w/W)
#     # # print "center:", (cx, cy), h/(1.0*w)
#     # # print "Theta 1:", theta1
#     # # print "Theta 2:", theta2
#     # # cv2.putText(img, "("+str(cx)+","+str(cy)+"), "+str(theta1)+", "+str(theta2), (int(CX/4), int(3*CY/4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#     cv2.imshow("Img", im)
#     cv2.imshow("Edges", closing)
#     # out.write(img)
#
#     print "........................................."
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         cv2.destroyAllWindows()
#         break
#
#
# cap.release()