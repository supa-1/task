import numpy as np
import cv2

frame = cv2.imread('pic4.png')
(h, w) = frame.shape[:2]
if h > 1000 or w > 1000:
    frame = cv2.resize(frame, (w//4, h//4))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,40,40])
upper_blue = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

blue_region = cv2.bitwise_and(frame, frame, mask=mask)

gray = cv2.cvtColor(blue_region, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

thresh = cv2.drawContours(thresh, contours, -1, 255, cv2.FILLED)
num_mask = cv2.absdiff(thresh,mask)
num_mask = cv2.GaussianBlur(num_mask, (5,5), 0)
num_mask = cv2.morphologyEx(num_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
num = cv2.bitwise_and(frame, frame, mask=num_mask)

cv2.imshow('num_mask',num_mask)
cv2.imshow('num',num)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()