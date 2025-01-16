import numpy as np
import cv2
cap = cv2.VideoCapture('task4_level2.mp4')
while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([110,50,60])
    upper_blue = np.array([130,255,255])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask = mask1
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours,h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pass
        
        
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('mask',mask)
    if(cv2.waitKey(5) & 0xFF == ord('q')):
        break
    
    
    
