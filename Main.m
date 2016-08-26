clc; close all; clear;

ip_im = imread('test2.png');

mexopencvDir = 'C:\Users\asr\Downloads\mexopencv-master';
addpath(mexopencvDir);

figure,imshow(ip_im);title('Input Image');

gimg = rgb2gray(ip_im);

%Process the chess board and segments it out
%by perfroming Otsu's binarization followed by Canny edge detection and Hough transform
t = graythresh(gimg);
segment = im2bw(gimg,t);
tIm = edge(segment,'canny');
figure,imshow(tIm);title('Result after Otsu binarization and Canny edge detection on the input image');

%Hough line detection
hline = cv.HoughLinesP(tIm,'MaxLineGap',500);

[im_ht,im_wid] =size(gimg);
mask_l = zeros(im_ht,im_wid);
mask_r = zeros(im_ht,im_wid);
mask_t = zeros(im_ht,im_wid);
mask_b = zeros(im_ht,im_wid);
for p=1:size(hline,2)
    x = double([hline{p}(1,1) hline{p}(1,3)]);
    y = double([hline{p}(1,2) hline{p}(1,4)]);
    m = (y(2) - y(1)) / (x(2) - x(1));
    if(m>0)
        for im_x=1:im_wid
            yp = ceil(m*(im_x-x(1))+y(1));
            if(yp >= 1 && yp <= im_ht)
                mask_r(yp,im_x:im_wid) = 1;
                mask_l(yp,1:im_x) = 1;
            end
        end
    else
        for im_x=1:im_wid
            yp = ceil(m*(im_x-x(1))+y(1));
            if(yp >= 1 && yp <= im_ht)
                mask_b(yp:im_ht,im_x) = 1;
                mask_t(1:yp,im_x) = 1;
            end
        end
    end
end

final_mask = mask_l.*mask_r.*mask_t.*mask_b;
final_mask = imclose(final_mask,strel('disk',10));
%end of extract

CB = imerode(final_mask,strel('disk',8)).*im2double(gimg);
figure,imshow(CB); title('Segmented Chess Board');

%start extracting chess corner by
%performing Otsu's tesholding, canny edge detection and Hough
t = graythresh(CB);
segment = im2bw(CB,t);
t1 = edge(segment,'canny');
figure,imshow(t1);title('Image again');
hline = cv.HoughLinesP(t1,'MaxLineGap',500);
output = zeros(size(t1));

n = size(output,2);
figure,imshow(output);hold on;
for p=1:size(hline,2)
    x = double([hline{p}(1,1) hline{p}(1,3)]);
    y = double([hline{p}(1,2) hline{p}(1,4)]);
    line1 = @(z) ((y(2) - y(1)) / (x(2) - x(1))) * (z - x(1)) + y(1);
    plot([1 n],line1([1 n]),'Color','w');
end

hold off;
gf = getframe;
[X, Map] = frame2im(gf);

cor = imclose(X,strel('disk',10));%refine the result
figure,imshow(cor)

cor1 = im2double(rgb2gray(cor));
%end chess cor1

%refine the results
cor = (1-cor1).*imerode(final_mask,strel('disk',8));
cor = imerode(cor,strel('disk',3));
sqB = im2bw(cor, graythresh(cor));
sqB = imopen(sqB,strel('disk',5));

%label the squares
c1 = bwconncomp(sqB);
L = labelmatrix(c1);
R = label2rgb(L);
figure, imshow(R); title('squares labelled');

%extract the 3 color channels
flt_im = im2double(ip_im);
R = flt_im(:,:,1);
G = flt_im(:,:,2);
B = flt_im(:,:,3);

%final_mask
m=((1-abs(R-B)));
m = im2bw(m,0.93);
mx=imopen(m,strel('square',3));
m = mx.*sqB;
figure,imshow(m);title('Binary image');

all = ((repmat(mx,[1 1 3]).*flt_im));

pp = (rgb2gray(repmat(m,[1 1 3]).*flt_im));
figure,imshow(pp);

[im_ht,im_wid] = size(gimg);
occ = zeros(size(gimg));
ocolor = zeros(size(gimg));
for seg=1:max(L(:))
    
    tmp_final_mask = zeros(size(gimg));
    area_occ = numel(find(m(find(L==seg)) == 1));
    area_reg = numel(find(L==seg));
    area_frac(1,seg) = area_occ/area_reg;
    
    if(area_frac(1,seg) > 0.4)        
        occ(find(L==seg)) = 1;
        tmp_final_mask(find(L==seg)) = 1;
        
        if(max(max(tmp_final_mask.*pp)) > 0.5)
            ocolor(find(L==seg)) = 0.75;
            disp('Detected white piece');
        else
            ocolor(find(L==seg)) = 0.25;
            disp('Detected black piece');
        end
    else
        ocolor(find(L==seg)) = 1;
    end
    
end
figure,imshow(sqB - (occ.*0.5),[]);title('Squares with pieces')

occ_c1 = bwconncomp(occ);
occ_cent = regionprops(occ_c1,'Centroid');
occ_L = labelmatrix(occ_c1);

ext_wid = 22;
ext_ht = 15;

p_ht = [];

figure,imshow(all);title('All the pawns in the image');
