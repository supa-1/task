import numpy as np
import cv2

# 摄像头版
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
# 视频版
# cap = cv2.VideoCapture('task4_level2.mp4')
# fps = cap.get(cv2.CAP_PROP_FPS)
# new_fps = 15
# frame_interval = int(fps / new_fps)
# frame_count = 0

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (320, 320))

    # if frame_count % frame_interval == 0:
    #     cv2.imshow('Frame', frame)
    #     if cv2.waitKey(int(1000 / new_fps)) & 0xFF == ord('q'):
    #         break
    # frame_count += 1
          
    
    cv2.imshow('frame', gray)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
