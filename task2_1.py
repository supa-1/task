import numpy as np
import cv2

cap = cv2.VideoCapture('task4_level1.mov')


while(True):
    
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # lower_red1 = np.array([0,43,46])
    # upper_red1 = np.array([10,255,255])
    # lower_red2 = np.array([156,43,46])
    # upper_red2 = np.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask2 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask3 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1
    # mask = mask1 + mask2 + mask3
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # ret,mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    similar = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(res, (cX, cY), 5, (255, 255, 255), -1)
            similar.append(contour)
            # num = cv2.mean(frame, mask=mask)
            # cv2.putText(res, f'{num}', (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    res = cv2.drawContours(res, similar, -1, (0,255,0), 3)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if(cv2.waitKey(5) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()