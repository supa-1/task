import numpy as np
import cv2
import math

def distance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

picture = 'pic4.png'
frame = cv2.imread(picture)
(h, w) = frame.shape[:2]
if h > 1000 or w > 1000:
    frame = cv2.resize(frame, (w//2, h//2))

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

num_mask = cv2.absdiff(thresh,mask)
num_mask = cv2.GaussianBlur(num_mask, (5,5), 0)
num_mask = cv2.morphologyEx(num_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
num = cv2.bitwise_and(frame, frame, mask=num_mask)

exist_mun_cnt = []

for contour in contours:
    M = cv2.moments(contour)
    black = np.zeros(frame.shape[:2], np.uint8)
    contour_mask = cv2.drawContours(black, [contour], -1, 255, cv2.FILLED)
    temp_mask = cv2.bitwise_and(thresh, num_mask, mask=contour_mask)
    
    if cv2.countNonZero(temp_mask) != 0:
        # cnt = contour
        exist_mun_cnt.append(contour)

s = [cv2.contourArea(x) for x in exist_mun_cnt]
cnt = exist_mun_cnt[s.index(max(s))]

approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),closed=True)
center,(Ma,ma),e_angle = cv2.fitEllipse(approx)
center = tuple(map(int,center))

app_tuple = [tuple(x[0]) for x in approx]
dis_list = [distance(x,center) for x in app_tuple]

dis_max = max(dis_list)
top = app_tuple[dis_list.index(dis_max)]

basic_vertix = (center[0] - top[0], center[1] - top[1])
# nomal basic_vertix should be (0,x)
angle = basic_vertix[1] / ((basic_vertix[0]**2 + basic_vertix[1]**2)**0.5)
rotated_vertix = (basic_vertix[0]*math.cos(angle) - basic_vertix[1]*math.sin(angle), basic_vertix[0]*math.sin(angle) + basic_vertix[1]*math.cos(angle)) 
angle = math.acos(angle)
if basic_vertix[0] > 0:
    angle = 2 * math.pi - angle
    angle = angle*180/math.pi
else:
    angle = angle*180/math.pi
    
(h, w) = frame.shape[:2]
m = cv2.getRotationMatrix2D((w//2,h//2), angle, 1) 
result = cv2.warpAffine(num, m, (w, h))
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

rectangle,_= cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(rectangle[0])
rotated_num = result[y:y+h,x:x+w]

# cv2.imshow('num',num)
cv2.imshow('rotated_num',rotated_num)
cv2.imwrite(f'custom_images/rotated_{picture}',rotated_num)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()