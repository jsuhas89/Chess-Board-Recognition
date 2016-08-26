__author__ = 'raghuveer'


import cv2
import numpy as np
from PIL import Image
#from matplotlib import pyplot as plt

kernel = np.ones((5,5),np.uint8)
img = cv2.imread('test1.jpg',0)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

ret,thresh1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
edges = cv2.Canny(thresh1,100,200)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("canny.jpg", edges)
minL = np.array(100000000)
maxL = np.array(8)
lines = cv2.HoughLinesP(edges,1,np.pi/180, 85, minL, maxL)


for x1,y1,x2,y2 in lines[0]:
    cv2.line(edges,(x1,y1),(x2,y2),(0,255,255),2)
cv2.imwrite("thersh.jpg",edges)


ret,thresh2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
edges1 = cv2.Canny(thresh1,100,200)
lines1 = cv2.HoughLines(edges1,1,np.pi/180,85)

#cv2.imwrite('houghlines123.jpg',lines)

for rho,theta in lines1[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(edges1,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imwrite('houghlines3.jpg',edges1)

mask_l = np.zeros(img.shape[:2])
mask_r = np.zeros(img.shape[:2])
mask_t = np.zeros(img.shape[:2])
mask_b = np.zeros(img.shape[:2])

dif = abs(img - edges)
cv2.imwrite('houghlines4.jpg',dif)

width, height = lines.shape[:2]

im_x, im_y = img.shape


for x1,y1,x2,y2 in lines[0]:
    cv2.line(edges,(x1,y1),(x2,y2),(255,255,255),2)
    if (x2-x1) == 0:
        continue
    else:
        m = (y2 - y1)/(x2 - x1)
        if(m > 0):
            for im_count in range(im_x):
                yp = round(m * (im_count - x1)) + y1
                if yp >= 1 and yp <= im_y:
                    for temp1 in range(int(im_count),int(im_x)):
                        mask_r[int(yp)][int(temp1)] = 1
                    for temp2 in range(int(im_count)):
                        mask_l[int(yp)][int(temp2)] = 1
        else:
            for im_count in range(im_x):
                yp = round(m * (im_count - x1)) + y1
                if yp >= 1 and yp <= im_y:
                    for temp1 in range(int(yp), int(im_y)):
                        mask_b[int(im_count)][int(temp1)] = 1
                    for temp2 in range(int(yp)):
                        mask_t[int(im_count)][int(temp2)] = 1

cv2.imwrite('new.jpg', edges)

#cv2.imshow('right',mask_r)
#cv2.waitKey(0)

#cv2.imshow('left',mask_l)
#cv2.waitKey(0)

#cv2.imshow('bottom',mask_b)
#cv2.waitKey(0)

#cv2.imshow('top',mask_t)
#cv2.waitKey(0)


temp_mask1 = cv2.bitwise_and(mask_l, mask_r, mask=None)
temp_mask2 = cv2.bitwise_and(mask_t, mask_b, mask=None)
final_mask = cv2.bitwise_and(temp_mask1, temp_mask2)


final_mask[final_mask == 1] = 255

final_mask = cv2.morphologyEx((final_mask * 1.0).astype(np.float32), cv2.MORPH_CLOSE, kernel=None)

cv2.imshow('final',final_mask)
cv2.waitKey(0)

cv2.imwrite("fin.jpg",final_mask)

cv2.destroyAllWindows()

x,y = img.shape
print x
print y
ret, orig_mask = cv2.threshold(final_mask, 10, 255, cv2.THRESH_BINARY)
a,b = final_mask.shape
print a
print b


imeg = cv2.imread('test1.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)

ret, orig_mask1 = cv2.threshold(imeg, 10, 255, cv2.THRESH_BINARY)
(thresh, im_bw) = cv2.threshold(imeg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#images1 = cv2.bitwise_and(img,img, mask=orig_mask)
#cv2.imwrite("123.jpg",images1)

#cv2.imwrite("123.jpg",images1)

