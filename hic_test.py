import numpy as np
import cv2

roi = cv2.imread('temp.png')
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

frame = cv2.imread('pic2.jpg')
(h, w) = frame.shape[:2]
frame = cv2.resize(frame, (w//4, h//4))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(frame, thresh)
# res = np.vstack([frame, thresh, res])

cv2.imshow('res', res)

if(cv2.waitKey(0) & 0xFF == ord('q')):
    cv2.destroyAllWindows()

