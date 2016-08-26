__author__ = 'raghuveer'


import cv2
import numpy as np
from matplotlib import pyplot as plt

kernel = np.ones((5,5),np.uint8)
img = cv2.imread('test1.jpg',0)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

ret,thresh1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
edges = cv2.Canny(thresh1,100,200)
#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("canny.jpg", edges)
lines = cv2.HoughLines(edges,1,np.pi/180,85)

width, height = lines.shape[:2]
#print width
#print height

im_x, im_y = img.shape
#print im_x
#print im_y

mask_l = np.zeros(img.shape[:2])
mask_r = np.zeros(img.shape[:2])
mask_t = np.zeros(img.shape[:2])
mask_b = np.zeros(img.shape[:2])

#print mask_r[1][4]
#print lines


#for p in range(height):
    #print lines[p]


for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(edges,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imwrite('houghlines3.jpg',edges)



for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))


    x = (x0, theta)
    y = (y0, rho)
    #print x
    #print y
    m = ((y[1] - y[0]) / (x[1] - x[0]))
    #print m

    if(m > 0):
        for im_count in range(im_x):
            yp = round(m * (im_x - x[0])) + y[0]
            if yp >= 1 and yp <= im_y:
                for temp1 in range(int(im_count),int(im_x)):
                    mask_r[int(yp)][int(temp1)] = 1
                for temp2 in range(int(im_count)):
                    mask_l[int(yp)][int(temp2)] = 1
    else:
        for im_count in range(im_x):
            yp = round(m * (im_x - x[0])) + y[0]
            if yp >= 1 and yp <= im_y:
                for temp1 in range(int(yp), int(im_y)):
                    mask_b[int(im_count)][int(temp1)] = 1
                for temp2 in range(int(yp)):
                    mask_t[int(im_count)][int(temp2)] = 1
    #cv2.line(edges,(x1,y1),(x2,y2),(255,255,255),2)



temp_mask1 = cv2.bitwise_and(mask_l, mask_r, mask=None)
temp_mask2 = cv2.bitwise_and(mask_t, mask_b, mask=None)
final_mask = cv2.bitwise_and(temp_mask1, temp_mask2)
#final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
#final_mask = np.array(final_mask)

final_mask = cv2.morphologyEx((final_mask * 1.0).astype(np.float32), cv2.MORPH_CLOSE, kernel)

#mask = cv2.erode((final_mask * 1.0).astype(np.float32), kernel, iterations=1)

#final_mask = mask_r.dot(mask_l).dot(mask_b).dot(mask_t)

#print 'a'
#print mask_r

#print 'b'
#print mask_l

#print 'c'
#print mask_t

#print 'd'
#print mask_b
e,f = final_mask.shape
print e
print f

j,k = img.shape
print j
print k  

cv2.imwrite("1.jpg",img)
cv2.imwrite("2.jpg",final_mask)
#chessboard = cv2.bitwise_and(final_mask,img)
#print closing
#print final_mask
#cv2.imwrite('masked.jpg',mask)

cv2.imwrite("binary.jpg",thresh1)

#05/03/2016
#chessBoard = imerode(mask,strel('disk',8)).*im2double(gray_im);
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
chessBoard = cv2.erode(cv2.bitwise_and(final_mask,cv2.normalize(gray_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)),kernel)

#figure,imshow(chessBoard); title('Segmented Chess Board');
cv2.imshow('Segmented Chess Board',chessBoard) 

ret1,thresh2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
edges1 = cv2.Canny(thresh2,100,200)
cv2.imwrite("canny1.jpg", edges1)
lines1 = cv2.HoughLines(edges1,1,np.pi/180,85)

#out = zeros(size(thrIm));
out = np.zeros(edges1.shape[:1],edges1.shape[:2])

#n = size(out,2);
n = out[0];

#figure,imshow(out);hold on;
cv2.imwrite("out.jpg", out)

for rho,theta in lines1[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    x = (x0, theta)
    y = (y0, rho)

	line2 = lambda z: ((y[1] - y[0]) / (x[1] - x[0])) * (z - x[0]) + y[0]

corner = cv2.normalize(cv2.cvtColor(line2, cv2.COLOR_BGR2GRAY).astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#refine the results
cor = cv2.bitwise_and((1-corner), cv2.erode(final_mask, kernel))
cor = cv2.erode(cor, kernel)
squares_BW = cv2.threshold(cor,60,255,cv2.THRESH_BINARY)
squares_BW = cv2.morphologyEx(squares_BW, cv2.MORPH_OPEN, kernel)

#label the squares
CC = cv2.connectedComponentsWithStats(squares_BW)
#CC = label(squares_BW)
L = CC[1]
RGB = cv2.cvtColor(L, cv2.COLOR_BGR2RGB)
cv2.imshow('Labelled squares',RGB) 