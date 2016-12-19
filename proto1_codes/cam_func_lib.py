if not('cv2' in globals() or 'cv2' in locals()):
    import cv2


def motion_bw(feed):
    a1 = feed.read()[1].astype('uint8')
    # time.sleep(0.4)
    a2 = feed.read()[1].astype('uint8')
    a3 = feed.read()[1].astype('uint8')
    a11 = cv2.GaussianBlur(a1, (3, 3), 3)
    a21 = cv2.GaussianBlur(a2, (3, 3), 3)
    # cv2.imshow('a1', a1)
    # cv2.imshow('a2', a2)
    # cv2.imshow('a3', a3)
    # print(type(abs(a - b).astype('uint8')))
    # c = abs(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)).astype('uint8')
    # c = abs(a1.astype(int)-a2.astype(int)).astype('uint8')
    # d = cv2.cvtColor(abs(cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(int)-cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(int)).astype('uint8'), cv2.COLOR_HSV2BGR)
    # d = abs(a1.astype(int)-2*a2.astype(int)+a3.astype(int)).astype('uint8')
    # c = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)
    # cv2.imshow('c', c)
    # cv2.imshow('d', d)

    # apera = cv2.getTrackbarPos('apera', 'canny')
    # t1 = cv2.getTrackbarPos('t1', "canny")
    # t2 = cv2.getTrackbarPos('t2', "canny")
    t1 = 80
    t2 = 120
    # e = cv2.Canny(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), t1, t2)
    c1 = abs(a11.astype(int) - a21.astype(int)).astype('uint8') + abs(a21.astype(int) - a11.astype(int)).astype('uint8')
    e1 = cv2.Canny(cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY), t1, t2)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    f1 = cv2.dilate(e1, kernal, iterations=1)
    a4 = a1/2 + a2/2
    res = cv2.bitwise_and(a4, a4, mask=f1)
    # cv2.imshow('canny', e)
    # cv2.imshow("canny1", e1)
    '''
    if cv2.waitKey(1) == ord('q'):
        feed.release()
        cv2.destroyAllWindows()
        break
    '''
    return res


def color_crop(im, mean_color, std_color):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    a = mean_color - std_color
    b = mean_color + std_color
    # mask = cv2.inRange(hsv, cv2.subtract(mean_color, 1*std_color), cv2.add(mean_color, 1*std_color))
    return cv2.inRange(hsv, a, b)


def camera_func(feed, mean_color, std_color, feed_center_x, feed_center_y):
    minLineLength = 120
    maxLineGap = 1

    minThresh = 200
    maxThresh = 220

    kernal_morph1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernal_morph11 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    kernal_morph2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernal_morph22 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

    _, im = feed.read()
    mask = color_crop(im, mean_color, std_color)

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_morph1)
    mask_open_close = cv2.dilate(mask_open, kernal_morph2, iterations=1)
    mask_open_1 = cv2.morphologyEx(mask_open_close, cv2.MORPH_OPEN, kernal_morph11)
    mask_open_close_1 = cv2.dilate(mask_open_1, kernal_morph22, iterations=1)
    res = cv2.bitwise_and(im, im, mask=mask_open_close_1)

    # print cv2.cvtColor(res, cv2.COLOR_BGR2GRAY).dtype
    cv2.imshow('open', mask_open)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
    cv2.imshow('close', mask_open_close)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
    cv2.imshow('open1', mask_open_1)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
    cv2.imshow('close1', mask_open_close_1)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()

    # edge_canny = cv2.Canny(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), minThresh, maxThresh, apertureSize=3)
    # edge_canny = cv2.morphologyEx(edge_canny,cv2.MORPH_OPEN,kernal_morph)

    cv2.imshow('res', res)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()

    # lines = cv2.HoughLines(edge_canny,1,np.pi/180,120)
    # for rho,theta in lines[0]:
       # a = np.cos(theta)
       # b = np.sin(theta)
        # x0 = a*rho
        # y0 = b*rho
        # x1 = int(x0 + 1000*(-b))
       # y1 = int(y0 + 1000*(a))
       # x2 = int(x0 - 1000*(-b))
       # y2 = int(y0 - 1000*(a))
       # cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

    '''
    lines = cv2.HoughLinesP(edge_canny, 1, np.pi/180, 100, minLineLength, maxLineGap)
    if lines != None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('asd',im)
    key_cv2 = cv2.waitKey(100)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break
    '''

    contours_raw, heirarcy = cv2.findContours(mask_open_close_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    # print 'raw ' + str(len(contours_raw))
    # fp = open('asd.txt', mode='w')
    # fp.write('raw ' + str(heirarcy))
    # fp.close()
    if contours_raw == []:
        return 0x9557705979
    for i in range(heirarcy.shape[1]):
        if heirarcy[0][i][3] == -1:
            contours.append(contours_raw[i])

    # print 'final ' + str(len(contours))
    # fp = open('asdfas.txt', mode='w')
    # fp.write('final ' + str(contours))
    # fp.close()
    if contours == []:
        return 0x9456484690
    # cv2.drawContours(im, contours, -1, (0, 255, 0, 2))
    # print contours

    perimeter = []
    area = []
    x_array = []
    y_array = []
    w_array = []
    h_array = []
    for i in contours:
        area.append(cv2.contourArea(i))
        perimeter.append(cv2.arcLength(i, True))
    max_area=max(area)
    i=area.index(max_area)
    # max(area)
    x, y, w, h = cv2.boundingRect(contours[i])

    target_x = int(x+w/2)
    target_y = int(y+h/2)

    # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.circle(im, (target_x, target_y), 2, (255, 0, 0), 2)
    cv2.line(im, (feed_center_x, 0), (feed_center_x, 2*feed_center_y), (0, 0, 255), 1)
    cv2.line(im, (target_x, target_y), (feed_center_x, target_y), (125, 0, 125), 2)
    '''
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
    '''
    area_peri_ratio = []
    for i in range(0, area.__len__()-1, 1):
        area_peri_ratio.append(area[i]/perimeter[i])
    # print str(area_peri_ratio)
    cv2.waitKey(1)

    cv2.imshow('contours', im)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == ord('p'):
        print contours
        print 'area = ' + str(area)
        print 'peri = ' + str(perimeter)
        cv2.waitKey(50)
    elif key_cv2 == 13:
        cv2.destroyAllWindows()

    dist = feed_center_x - target_x
    return dist