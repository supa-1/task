import numpy as np
import cv2

cap = cv2.VideoCapture('task4_level1.mov')
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)

lower_blue = np.array([100,40,40])
upper_blue = np.array([130,255,255])
# 存在子轮廓的轮廓列表
par_approx = []

while True:
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    frame = cv2.resize(frame, (w//2, h//2))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    num_cnt = []
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    contours,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 不存在标靶则跳过这一帧
    if h is None:
        continue
    
    for i in range(len(h[0])):
        n = h[0][i][2]
        m = i
        approx = cv2.approxPolyDP(contours[m],0.01*cv2.arcLength(contours[m],True),closed=True)
        
        num_cnt.append(contours[n])
        par_approx.append(approx)
        
    
    black = np.zeros(frame.shape[:2], np.uint8)

    num_mask = cv2.drawContours(black, num_cnt, 0, (255,255,255), cv2.FILLED) 
    frame = cv2.bitwise_and(frame, frame, mask=num_mask)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()