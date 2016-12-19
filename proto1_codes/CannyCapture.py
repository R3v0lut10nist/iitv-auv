#def CannyCapture(feed,fsize):
import numpy as np
import cv2

feed = cv2.VideoCapture(0)
lower_color = np.array([0,0,0])
upper_color = np.array([20,20,20])

minLineLength = 200
maxLineGap = 1

minThresh = 130
maxThresh = 150
if 1==1:

    #f1size=tuple(np.append(fsize,5))
    #im5=np.array((f1size,5))
    
    a = feed.read()[1];
    b = feed.read()[1];
    c = feed.read()[1];
    d = feed.read()[1];
    e = feed.read()[1];

    edge_canny1 = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),minThresh,maxThresh,apertureSize = 3)
    edge_canny2 = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),minThresh,maxThresh,apertureSize = 3)
    edge_canny3 = cv2.Canny(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY),minThresh,maxThresh,apertureSize = 3)
    edge_canny4 = cv2.Canny(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY),minThresh,maxThresh,apertureSize = 3)
    edge_canny5 = cv2.Canny(cv2.cvtColor(e, cv2.COLOR_BGR2GRAY),minThresh,maxThresh,apertureSize = 3)

    #cv2.imshow("hfrfg",edge_canny2)
    #cv2.waitKey(5000);
    
    sum1 = edge_canny1 + edge_canny2+ edge_canny3+edge_canny4+edge_canny5;
    
    cv2.imshow("sss",sum1)
    cv2.waitKey(10000)
