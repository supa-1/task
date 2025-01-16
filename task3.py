import numpy as np
import cv2
import math
def distance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
# 待优化
def find_top(dots):
    top = dots[0]
    far = distance(dots[0],dots[0]) + distance(dots[0],dots[1]) 
    + distance(dots[0],dots[2]) + distance(dots[0],dots[3]) + distance(dots[0],dots[4])
    for dot in dots:
        far_new = distance(dot,dots[0]) + distance(dot,dots[1]) 
        + distance(dot,dots[2]) + distance(dot,dots[3]) + distance(dot,dots[4])
        if far < far_new:
            top = dot
    return top
  

frame = cv2.imread('pic2.jpg')
(h, w) = frame.shape[:2]

frame = cv2.resize(frame, (w//2, h//2))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100,40,40])
upper_blue = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_and(gray, gray, mask=mask)
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

contours,hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),closed=True)
        frame = cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
        center = (cx, cy)

app_tuple = [tuple(x[0]) for x in approx]
top = find_top(app_tuple)
# cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
# cv2.circle(frame, top, 5, (255, 255, 255), -1)
# cv2.line(frame, (cx, cy), top, (255, 255, 255), 3)
basic_vertix = (cx - top[0], cy - top[1])
# nomal basic_vertix should be (0,x)
angle = basic_vertix[1] / ((basic_vertix[0]**2 + basic_vertix[1]**2)**0.5)
angle = math.acos(angle)
(h, w) = frame.shape[:2]
m = cv2.getRotationMatrix2D((w//2,h//2), angle*180/math.pi, 1) 
result = cv2.warpAffine(frame, m, (w, h))
print(frame.shape)
cv2.imshow('frame',frame)
cv2.imshow('result',result)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()