import numpy as np
import cv2

img = np.zeros((3, 500, 500))
poly = np.array(([[-0.3412, -0.6326], [ 0.1746, -0.0614], [ 0.4627,  0.8090], [-0.0531,  0.2377]]))
poly = (poly * 500).astype(int)
print(cv2)
cv2.fillPoly(img, poly, 1)
cv2.imshow('huh', img)
cv2.waitKey(0)