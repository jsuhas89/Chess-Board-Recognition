#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <math.h>      


using namespace cv;
using namespace std;

double  mask_l[553][1268], mask_r[553][1268],  mask_t[553][1268], mask_b[553][1268] = {{0}};
double mask[553][1268];

int erosion_elem = 0;
int erosion_size = 0;

void extractChessBoard(Mat image_gray) {
 	Mat thr, segIm, thrIm, linesMat, color_thrIm;
 	cv::threshold(image_gray, thr, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
 	cv::Canny(thr, thrIm, 50, 200, 3);
	imwrite( "Canny.jpg", thr);
    cvtColor( thrIm, color_thrIm, CV_GRAY2BGR );

    vector<Vec4i> lines;
    HoughLinesP( thrIm, lines, 1, CV_PI/180, 50, 20, 300 );

    int im_ht  = image_gray.rows;
    int im_wid = image_gray.cols;
 
    imwrite( "Hough.jpg", thrIm);
 
    for( size_t i = 0; i < lines.size(); i++ ) {
    	float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( color_thrIm, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 3 );
        signed int m = ( pt2.y - pt1.y) / (pt2.x - pt1.x);
    }  
    imwrite( "HoughLines.jpg", color_thrIm);
    cvtColor( color_thrIm, color_thrIm, CV_BGR2GRAY );


    Mat mask = color_thrIm - thrIm;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4,4)));

    Mat dst;
    int erosion_type;

	imwrite( "Result_mask.jpg", mask);

    int erosion_size = 1;  
    Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size) );
    int rows1 = element.rows;
    int cols1 = element.cols;
	
	imwrite( "Result_before.jpg", thrIm);
    erode(mask,dst,element);  
	imwrite( "Result.jpg", dst);
    namedWindow( "Result window", CV_WINDOW_AUTOSIZE );   
    //imshow( "Result window", dst );

}


int main( int argc, char** argv ) {
    char* imageName = argv[1];

    Mat image, mask;
    image = imread( imageName, 1 );

    if( argc != 2 || !image.data ) {
      printf( " No image data \n " );
      return -1;
    }
  

    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    
    extractChessBoard(gray_image);

    waitKey(0);

    return 0;
}