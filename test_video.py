import numpy as np
import cv2
import math

def distance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

cap = cv2.VideoCapture('task4_level3.mp4')
template = cv2.imread('gray_template.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

lower_blue = np.array([100,50,50])
upper_blue = np.array([140,255,255])
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    (h, w) = frame.shape[:2]
    if h > 1000 or w > 1000:
        frame = cv2.resize(frame, (w//2, h//2))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    temp_image = []
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_region = cv2.bitwise_and(frame, frame, mask=mask)
    
    gray = cv2.cvtColor(blue_region, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours,h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    thresh = cv2.drawContours(thresh, contours, -1, 255, cv2.FILLED)
    num_mask = cv2.absdiff(thresh,mask)
    num_mask = cv2.GaussianBlur(num_mask, (5,5), 0)
    num_mask = cv2.morphologyEx(num_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    num = cv2.bitwise_and(frame, frame, mask=num_mask)
    
    if h is None:
        continue 
    
    for contour in contours:    
        black = np.zeros(frame.shape[:2], np.uint8)
        contour_mask = cv2.drawContours(black, [contour], -1, 255, cv2.FILLED)
        temp_mask = cv2.bitwise_and(thresh, num_mask, mask=contour_mask)
        
        s = cv2.contourArea(contour)
        if s < 50:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        temp_img = contour_mask[y:y+h,x:x+w]
        similarity = cv2.matchShapes(temp_img, template, 1, 0.0)
        
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),closed=True)
        
        if cv2.countNonZero(temp_mask) > 25 and similarity < 0.00001 and len(approx) in range(3,12):
            # temp.append(contour)
            x,y,w,h = cv2.boundingRect(contour)
            # temp_approx.append([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            temp_num_imgae = frame[y:y+h,x:x+w]
            
            temp_image.append(frame[y:y+h,x:x+w])
            
    for frame in temp_image:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        s = [cv2.contourArea(x) for x in contours]
        cnt = contours[s.index(max(s))]

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
        cv2.imwrite('test\\num'+str(i)+'.png',rotated_num)
        i += 1
        
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()