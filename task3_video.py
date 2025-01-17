import numpy as np
import cv2
import math

def distance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

cap = cv2.VideoCapture('task4_level1.mov')

while True:
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    frame = cv2.resize(frame, (w//2, h//2))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([100,40,40])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s = [cv2.contourArea(x) for x in contours]
    contour = contours[s.index(max(s))]

    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),closed=True)
    if len(approx) >= 5 :
        center,(Ma,ma),e_angle = cv2.fitEllipse(approx)
    else:
        center,(Ma,ma),e_angle = cv2.fitEllipse(contour)
    center = tuple(map(int,center))

    app_tuple = [tuple(x[0]) for x in approx]
    dis_list = [distance(x,center) for x in app_tuple]

    dis_max = max(dis_list)
    top = app_tuple[dis_list.index(dis_max)]
    cv2.circle(frame, center, 5, (255, 255, 255), -1)
    cv2.circle(frame, top, 5, (255, 255, 255), -1)
    cv2.line(frame, center, top, (255, 255, 255), 3)
    
    basic_vertix = (center[0] - top[0], center[1] - top[1])
    # nomal basic_vertix should be (0,x)
    angle = basic_vertix[1] / ((basic_vertix[0]**2 + basic_vertix[1]**2)**0.5)
    angle = math.acos(angle)
    rotated_vertix = (basic_vertix[0]*math.cos(angle) - basic_vertix[1]*math.sin(angle), basic_vertix[0]*math.sin(angle) + basic_vertix[1]*math.cos(angle)) 
    if rotated_vertix[0] > 0.1:
        angle = 2 * math.pi - angle
        angle = angle*180/math.pi
        
    else:
        angle = angle*180/math.pi

    (h, w) = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w//2,h//2), angle, 1) 
    result = cv2.warpAffine(frame, m, (w, h))
    
    cv2.imshow('frame', frame)
    cv2.imshow('result', result)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break