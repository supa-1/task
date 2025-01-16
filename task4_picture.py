import numpy as np
import cv2

# 图片版识别并提取数字部分
frame = cv2.imread('pic1.jpg')
(h, w) = frame.shape[:2]

frame = cv2.resize(frame, (w//4, h//4))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,40,40])
upper_blue = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

contours,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(h)
# 数字区域轮廓列表
num_cnt = []
# 标靶轮廓列表
par_cnt = []
# 标靶轮廓近似的五个顶点列表
par_approx = []
# 查找存在子轮廓的轮廓而得到子轮廓位置
for i in range(len(h[0])):
    if h[0][i][2] > -1: # 存在子轮廓
        n = h[0][i][2]
        m = i
        num_cnt.append(contours[n])
        par_cnt.append(contours[m])
        par_approx.append(cv2.approxPolyDP(contours[m],0.01*cv2.arcLength(contours[m],True),closed=True))
        
black = np.zeros(frame.shape[:2], np.uint8)
num_mask = cv2.drawContours(black, num_cnt, 0, (255,255,255), cv2.FILLED) 
frame = cv2.bitwise_and(frame, frame, mask=num_mask)
print(f'par_approx:{par_approx}')
cv2.imshow('frame',frame)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()