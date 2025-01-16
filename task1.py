import numpy as np
import cv2
import time

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('task4_level2.mp4')
prev_time = time.time()

while(True):
    ret, frame = cap.read()
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (320, 320))

    # 在帧上叠加帧率信息
    cv2.putText(gray, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
