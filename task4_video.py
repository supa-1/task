import numpy as np
import cv2

cap = cv2.VideoCapture('task4_level1.mov')
# cap = cv2.VideoCapture('task4_level2.mp4')
# cap = cv2.VideoCapture('task4_level3.mp4')

lower_blue = np.array([100,50,50])
upper_blue = np.array([140,255,255])

# 标靶列表
par_approx = []

while True:
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    if h > 1000 or w > 1000:
        frame = cv2.resize(frame, (w//2, h//2))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_region = cv2.bitwise_and(frame, frame, mask=mask)
    
    gray = cv2.cvtColor(blue_region, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours,h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 不存在标靶则跳过这一帧
    if h is None:
        continue 
    # 这一帧的标靶位置列表     
    temp_approx = []
    temp = []
    num_cnt = []
    
    thresh = cv2.drawContours(thresh, contours, -1, 255, cv2.FILLED)
    num_mask = cv2.absdiff(thresh,mask)
    num_mask = cv2.GaussianBlur(num_mask, (5,5), 0)
    num_mask = cv2.morphologyEx(num_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    num = cv2.bitwise_and(frame, frame, mask=num_mask)
    
    # 用num_mask,thresh来判断mask中的有数字的标靶
    for contour in contours:    
        black = np.zeros(frame.shape[:2], np.uint8)
        contour_mask = cv2.drawContours(black, [contour], -1, 255, cv2.FILLED)
        temp_mask = cv2.bitwise_and(thresh, num_mask, mask=contour_mask)
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),closed=True)
        # level1版本
        if cv2.countNonZero(temp_mask) > 50 and len(approx) in range(5,12):
            temp.append(contour)
            temp_approx.append(approx)
        # level2,level3版本
        # if cv2.countNonZero(temp_mask) > 50 and len(approx) in range(5,12) and cv2.contourArea(contour) in range(0,2000):
        #     temp.append(contour)
        #     temp_approx.append(approx)
        
    black = np.zeros(frame.shape[:2], np.uint8)
    temp_pic = cv2.drawContours(black, temp, -1, 255, cv2.FILLED) # 显示存储的轮廓
    
    cv2.imshow('num',num)
    cv2.imshow('frame',frame)
    cv2.imshow('temp_pic',temp_pic)
    par_approx.append(temp_approx)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()