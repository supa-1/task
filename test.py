import numpy as np
import cv2

frame = cv2.imread('pic1.jpg')
(h, w) = frame.shape[:2]

frame = cv2.resize(frame, (w//4, h//4))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,40,40])
upper_blue = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

contours,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)

cv2.imshow('frame',frame)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()