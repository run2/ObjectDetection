#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
typedef struct {
	int x;
	int y;
	int rad;
} dartBoardCenter;
/** Function Headers */
void detectAndDisplay( Mat frame, Mat scene,int index ,bool useSURF);
Rect findObjectSURF( Mat objectMat, Mat sceneMat, int hessianValue ,int index);
dartBoardCenter verifyDartBoard(Mat rect);
void getSobelYDerivativeKernel (cv::Mat& sobelYKernel);
void getSobelXDerivativeKernel (cv::Mat& sobelXKernel);
std::vector<Rect> getCleanVector(std::vector<Rect> vec1, std::vector<Rect> vec2);
bool isRectangleInside(Rect out, Rect in);
/** Global variables */
String cascade_name = "dartcascade_1.xml";
String cascade_name_1 = "dartcascade.xml";

CascadeClassifier cascade;
CascadeClassifier cascade_1;
string window_name = "Detection DartBoard ";
RNG rng(12345);

/** @function main */
int main( int argc, const char** argv )
{	
	//Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !cascade_1.load( cascade_name_1 ) ){ printf("--(!)Error loading\n"); return -1; };
	Mat obj = imread("dart.bmp", CV_LOAD_IMAGE_COLOR);
	for(int i1 =0;i1<12;i1++)
	{
		std::stringstream ss;
		ss << "dart" << i1 << ".jpg";
		std::string s = ss.str();
		Mat scene = imread(s, CV_LOAD_IMAGE_COLOR);
		detectAndDisplay( obj,scene ,i1,true);
		//detectAndDisplay( obj,scene ,i1,true);
		//findObjectSURF(obj,scene,400,i1);
	}
	//Mat scene = imread("dart10.jpg", CV_LOAD_IMAGE_COLOR);
	//detectAndDisplay( obj,scene ,10,true);
	//findObjectSURF(obj,scene,400,1);
	
	cv::waitKey();
}
Rect findObjectSURF( Mat objectMat, Mat sceneMat, int hessianValue,int index )
{
    //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 2400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( objectMat, keypoints_object );
  detector.detect( sceneMat, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( objectMat, keypoints_object, descriptors_object );
  extractor.compute( sceneMat, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //printf("-- Max dist : %f \n", max_dist );
  //printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 2*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  Mat img_matches;
  drawMatches( objectMat, keypoints_object, sceneMat, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::DEFAULT  );

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( objectMat.cols, 0 );
  obj_corners[2] = cvPoint( objectMat.cols, objectMat.rows ); obj_corners[3] = cvPoint( 0, objectMat.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);
  int width = (int)scene_corners[2].x - (int)scene_corners[0].x;
  int height = (int)scene_corners[0].y - (int)scene_corners[2].y;
 
  int area = width * height;
 
  Rect dartBoardFound =  Rect(scene_corners[0], scene_corners[2]);
  if(dartBoardFound.area() > 600)
  {
	  Mat F = sceneMat.clone();
	  rectangle(F, scene_corners[0], scene_corners[2], Scalar( 0, 255, 0 ), 2);
  
	  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
	    line( img_matches, scene_corners[0] + Point2f( objectMat.cols, 0), scene_corners[1] + Point2f( objectMat.cols, 0), Scalar(0, 255, 0), 4 );
	    line( img_matches, scene_corners[1] + Point2f( objectMat.cols, 0), scene_corners[2] + Point2f( objectMat.cols, 0), Scalar( 0, 255, 0), 4 );
	    line( img_matches, scene_corners[2] + Point2f( objectMat.cols, 0), scene_corners[3] + Point2f( objectMat.cols, 0), Scalar( 0, 255, 0), 4 );
	    line( img_matches, scene_corners[3] + Point2f( objectMat.cols, 0), scene_corners[0] + Point2f( objectMat.cols, 0), Scalar( 0, 255, 0), 4 );
   
	  //-- Show detected matches
	    std::stringstream ss;
		ss << "Good Matches & Object detection" << index;
		std::string s = ss.str();
		
		//imshow( s, img_matches );
		//std::stringstream ss;
		ss.clear();
		ss << "Surf Detector" << index;
		s = ss.str();
		
		//imshow( s, F );
		return dartBoardFound;
  }
  else
  {
	  return dartBoardFound;
  }
}
/** @function detectAndDisplay */
void detectAndDisplay( Mat obj, Mat scene, int index ,bool useSURF)
{
	std::vector<Rect> dartBoards;
	std::vector<Rect> dartBoards1;
	Mat frame_gray;
	int buffer = 30;
	cvtColor( scene, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	bool surfUsed = false;
	int aBoardFound = 0;
	Rect surfRect;
	//-- Detect faces
	cascade.detectMultiScale( frame_gray, dartBoards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	cascade_1.detectMultiScale( frame_gray, dartBoards1, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	if(useSURF)
	{
		surfRect = findObjectSURF(obj , scene, 500 ,index);
		//dartBoards.push_back(surfRect);
		
	}
	//dartBoards = getCleanVector(dartBoards,dartBoards1);
	std::cout << dartBoards.size() << std::endl;
	
	
	for( int i = 0; i < dartBoards.size(); i++ )
	{
		if(index != 10)
		{
			if(dartBoards[i].x > surfRect.x -buffer && dartBoards[i].x < surfRect.x + buffer)
			{
					if(dartBoards[i].y > surfRect.y -buffer && dartBoards[i].y < surfRect.y + buffer)
					{
						//this has been verified by the surf detector
						if(surfRect.area() > 600 &&  verifyDartBoard(Mat(scene, surfRect)).rad > 0)
						{
							rectangle(scene,surfRect, Scalar( 0, 255, 0 ), 2);
							surfUsed = true;
							aBoardFound++;
						}
					}
					else
					{
						if( verifyDartBoard(Mat(scene, dartBoards[i])).rad > 0)
						{
							rectangle(scene, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
							aBoardFound ++;
						}
					}

			}
			else
			{
				if( verifyDartBoard(Mat(scene, dartBoards[i])).rad > 0)
				{
					rectangle(scene, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
					aBoardFound ++;
				}
			}
		}
		else
		{
		}
		
	}
	if(surfUsed == false && index != 10)
	{
		if(surfRect.area() > 600 &&  verifyDartBoard(Mat(scene, surfRect)).rad > 0)
		{
			rectangle(scene,surfRect, Scalar( 0, 255, 0 ), 2);
			surfUsed = true;
			aBoardFound++;
		}
	}
	if(aBoardFound == 0)
	{
		dartBoardCenter c =  verifyDartBoard(scene);
		if(c.rad > 0)
		{
			Rect foundRect  = cv::Rect(c.x- c.rad,c.y- c.rad,2*c.rad,2*c.rad);
			rectangle(scene,foundRect, Scalar( 0, 255, 0 ), 2);
		}
	}
	std::stringstream ss;
	ss << window_name << index << " surf used: " << useSURF;
	std::string s = ss.str();
	imshow(s, scene );
		/*if(useSURF)
		{
			if(surfUsed == true)
			{
				//surf has been used to verify one dartboard or surf does not detect a dartboard
				if( verifyDartBoard(Mat(scene, dartBoards[i])))
				{
					rectangle(scene, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
					aBoardFound ++;
				}
			}
			else
			{
				if(dartBoards[i].x > surfRect.x -buffer && dartBoards[i].x < surfRect.x + buffer)
				{
					if(dartBoards[i].y > surfRect.y -buffer && dartBoards[i].y < surfRect.y + buffer)
					{
						//this has been verified by the surf detector
						rectangle(scene, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
						surfUsed = true;
						aBoardFound++;
					}

				}
				if(surfUsed == false)
				{
					//this didnt match the surf, verify
					if( verifyDartBoard(Mat(scene, dartBoards[i])))
					{
						rectangle(scene, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
						aBoardFound++;
					}
				}
			}
		}
		else
		{
			if( verifyDartBoard(Mat(scene, dartBoards[i])))
			{
				rectangle(scene, Point(dartBoards[i].x, dartBoards[i].y), Point(dartBoards[i].x + dartBoards[i].width, dartBoards[i].y + dartBoards[i].height), Scalar( 0, 255, 0 ), 2);
				aBoardFound ++;
			}
		}
	}
	if(surfUsed == false && useSURF == true)
	{
		//surf available and noBoardFound
		//so check if the surf is correct
		if(verifyDartBoard(Mat(scene,surfRect)))
		{
			rectangle(scene,surfRect, Scalar( 0, 255, 0 ), 2);
		}
	}
	*/
	
}
std::vector<Rect> getCleanVector(std::vector<Rect> vec1, std::vector<Rect> vec2)
{
	int acceptableErrorDistance = 30;
	std::vector<Rect> resultingVec = std::vector<Rect>(vec1);

	for(int i = 0; i < vec2.size(); i++)
	{
		resultingVec.push_back(vec2[i]);
	}

	for(int i = 0; i < resultingVec.size(); i++)
	{
		for(int j = i + 1; j < resultingVec.size(); j++)
		{			
			// are similar so we need to delete one
			if(isRectangleInside(resultingVec[i], resultingVec[j]) || isRectangleInside(resultingVec[j], resultingVec[i]))
			{
				if(resultingVec[i].area() > resultingVec[j].area())
				{
					resultingVec.erase(resultingVec.begin() + j);
				}else
				{
					resultingVec.erase(resultingVec.begin() + i);
				}
			}
		}
	}

	return resultingVec;
}

bool isRectangleInside(Rect out, Rect in)
{
	for(int x = in.x; x < in.x + in.width; x++)
	{
		for(int y = in.y; y < in.y + in.height; y++)
		{
			if(x >= out.x && x <= out.x + out.width && y >= out.y && y <= out.y + out.height)
				return true;
		}
	}
	
	return false;
}
void getSobelXDerivativeKernel (cv::Mat& sobelXKernel){
	
	sobelXKernel.at<int>(0,0) = -1;
	sobelXKernel.at<int>(0,2) = 1;
	sobelXKernel.at<int>(1,0) = -2;
	sobelXKernel.at<int>(1,2) = 2;
	sobelXKernel.at<int>(2,0) = -1;
	sobelXKernel.at<int>(2,2) = 1;

}
/*
This function initializes a Sobel Y Derivative kernel
*/
void getSobelYDerivativeKernel (cv::Mat& sobelYKernel){
	
	sobelYKernel.at<int>(0,0) = -1;
	sobelYKernel.at<int>(2,0) = 1;
	sobelYKernel.at<int>(0,1) = -2;
	sobelYKernel.at<int>(2,1) = 2;
	sobelYKernel.at<int>(0,2) = -1;
	sobelYKernel.at<int>(2,2) = 1;

}

dartBoardCenter verifyDartBoard(cv::Mat colorinput){
	cv::Mat input = cv::Mat::zeros(input.rows, input.cols, CV_8U);
	cv:cvtColor(colorinput,input,CV_RGB2GRAY);
	
	cv::Mat sobelXKernel = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat sobelYKernel = cv::Mat::zeros(3, 3, CV_64F);
	//cv::Mat gradientLabels;
	//gradientLabels.create(input.size(), CV_16S); // will have floating values or radians
	cv::Mat dartboardCenterVotes = cv::Mat::zeros(input.rows, input.cols, CV_64F);
	cv::Mat dartboardCenters;
	dartboardCenters  = input.clone();// .create(input.size(), input.type()); // will have floating values or radians
	//cv::namedWindow("DartBoards", CV_WINDOW_AUTOSIZE);
	//cv::imshow("DartBoards", dartboardCenters);
	//cv::waitKey();		
	cv::Mat actualGradient;
	actualGradient.create(input.size(), CV_64F); // will have floating values or radians
	cv::Mat mag;
	mag.create(input.size(), CV_64F); // will have floating values or radians

	cv::Mat xyCoordinatesOfDartBoards;
	
	getSobelXDerivativeKernel(sobelXKernel);
	getSobelYDerivativeKernel(sobelYKernel);

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		1, 1, 1, 1,
		cv::BORDER_REPLICATE );


	double sumxDerivative = 0;
	double sumyDerivative = 0;

	const double PI = atan(1.0)*4; 
	// gradient is atan2 of (df/dy)/(df/dx) - which can range from -PI to PI
	double minGrad = -PI; 
	double maxGrad = PI;
	double tempGrad = 0;
	double rangegrad = 2 * PI;
	double minMag = (double)(sqrt((double)2)*255*4); // max value of sqrt((df/dy)^2 + (df/dx)^2). Individually  the max of (df/dx) or (df/dy) is 255*4
	double maxMag = 0; // min value can only be zero as there are squares within the square root
	double tempmag;
	double rangemag;

	int imageval ;
	int xkernalval ;
	int ykernalval ;
	int imagex ;
	int imagey ;
	int kernelx;
	int kernely;
	int mulx;
	int muly;

	double gradval , gradFactor;
	gradFactor = ((double)255/rangegrad);


	// find the x Derivative,y Derivative,magnitude and gradient -
	for ( int i = 0; i < input.rows; i++ )
		{	
			for( int j = 0; j < input.cols; j++ )
			{
				sumxDerivative = 0;
				sumyDerivative = 0;

				for( int m = -1; m <= 1; m++ )
				{
					for( int n = -1; n <= 1; n++ )
					{
						// find the correct indices we are using
						imagey = i + 1 - m;
						imagex = j + 1 - n;
						kernely = m + 1;
						kernelx = n + 1;
					
						// get the values from the padded image and the kernel
						imageval = ( int ) paddedInput.at<uchar>( imagey, imagex );
						//printf("Image Val %d\n",imageval);
					
						//int kernalval = kernel.at<int>( kernelx, kernely );
						xkernalval = sobelXKernel.at<int>( kernely, kernelx );
						
						ykernalval = sobelYKernel.at<int>( kernely, kernelx );

						//printf("Kernel Valxy %d %d\n",xkernalval,ykernalval);

						mulx = imageval * xkernalval;
						muly = imageval * ykernalval;

						//printf("Multiply Valxy %d %d\n",mulx,muly);
						//printf("sumxDerivative %d\n",sumxDerivative);
					
						sumxDerivative += (double)mulx;
						sumyDerivative += (double)muly;
					}
				
				}

				tempmag = sqrt((sumyDerivative*sumyDerivative) + (sumxDerivative*sumxDerivative));

				if(tempmag<minMag)
					minMag = tempmag;

				if(tempmag>maxMag)
					maxMag = tempmag;
				mag.at<double>(i, j) = tempmag;

				tempGrad = atan2(sumyDerivative,sumxDerivative) ;

				actualGradient.at<double>(i, j) = tempGrad;
				//gradval = actualGradient.at<double>(i, j);
				//gradval = (gradval - minGrad)*gradFactor;
				//if((int)gradval == 31 ||  (int)gradval == 63 || (int)gradval == 95 || (int)gradval == 127 || (int)gradval == 159  ){
					//std::cout << gradval << " " << actualGradient.at<double>(i, j)*180/PI << '\n';
				//}
				//gradientLabels.at<int>(i,j) = (int)gradval;
			}
		
		}
		
		//std::cout << rangegrad << '\n';
		int gradArray[256] ;// = new int[256];
		int sortedgradArray[256] ;// = new int[256];
		for(int i = 0 ; i < 255 ; i ++ ){
				gradArray[i] = 0;
				sortedgradArray[i] = 0;
		}
		for ( int i = 0; i < input.rows; i++ )
		{	
			for( int j = 0; j < input.cols; j++ )
			{
				gradval = actualGradient.at<double>(i, j);
				gradval = (gradval - minGrad)*gradFactor;
				
				gradArray[(int)gradval]+=1;
				sortedgradArray[(int)gradval] +=1;
			}
		}
		int elements = sizeof(sortedgradArray) / sizeof(sortedgradArray[0]); 
		std::sort(sortedgradArray, sortedgradArray + elements);

		int eighthBestGradientWithMaxPixels = sortedgradArray[245]; 
		int bestGradients[10];
		int j = 0;
		for(int i = 0 ; i < 256 ; i++){
			if(gradArray[i] > eighthBestGradientWithMaxPixels){
				bestGradients[j] = i;
				std::cout << gradArray[i] << " " << i << '\n';
				j++;
			}
		}
		int diff = 0 ;
		int prevDiff = -1;
		int countCosistantGradientDiff = 0 ;
		int consistantDiff = 0;
		int startConsistantDiff = 0;
		int endConsistantDiff = 0;
		bool startCounting = false;
		 
		for(int i = 0 ; i <= 9 ; i ++){
			//std::cout << bestGradients[i]%32 << '\n';
			if(bestGradients[i]%32==31){
				consistantDiff++;
			}
		}

		//std::cout << countCosistantGradientDiff << " related gradients " << '\n';
		//if(consistantDiff>=3){
			// choose a one gradient with maximum points
			int gradientWithMaxPoints = 0;
			int maxGradient = 0;
			int* newBestGradients;
			j = 0;
			newBestGradients = new int[consistantDiff];
			for(int i = 0 ; i <= 9 ; i ++){
				if(bestGradients[i]%32==31){
					newBestGradients[j] =  bestGradients[i];
					//j++;
				}
			}
			for(int k = 0 ; k < consistantDiff ; k ++ ){
				std::cout << newBestGradients[k] << '\n';
			}
			//double myArray[8] = { 31 , 63 , 95 , 127 , 159 , 191 , 223 , 254 };
			double myArray[20] = { 9, 27, 45 , 63 , 95 , 117 , 135 , 153 , 171 , -9, -27, -45 , -63 , -95 , -117 , -135 , -153 , -171 };
			// vote for dart board centers
			int noOfDartBoards = 0;
			int maxOfDartBoards = 0;
			double theta = 0;
			int rangeRadius = 50;
			int minRadius = 10;
			//double vote;
			double maxVote = 0;
			int maxVoteX = 0;
			int maxVoteY = 0;
			for ( int yImage = 1; yImage < input.rows; yImage++ )
				{	
					noOfDartBoards = 0;
					for( int xImage = 1 ; xImage < input.cols; xImage++ )
					{ 

						//int gradientLabel = (int)((actualGradient.at<double>(yImage,xImage) - minGrad)*gradFactor);
						int gradientLabel = (int)actualGradient.at<double>(yImage,xImage)*180/PI ;
						//input.at<uchar>(yImage,xImage) = (uchar)0;
						for(int k = 0 ; k < 20 ; k ++ ){
							if(gradientLabel==myArray[k] && mag.at<double>(yImage,xImage) > 200){
								//std::cout << gradientLabel << "\n";
							//if(gradientLabel == 127 && mag.at<double>(yImage,xImage) > 100){
								//output.at<uchar>(yImage,xImage) = (uchar)255;
								// vote for all pixels 
								
							theta = (double)actualGradient.at<double>(yImage,xImage); // get gradient of gradient image at this pixel
							theta = theta + PI/2;
							// Donot vote for gradients which are created by horizontal or vertical edges as they are too common


								for(int rad = 0 ; rad < rangeRadius; rad++ ){ // hough space radius/depth loop
									
									int probableYCenterPlus = yImage + (int)((rad+minRadius)*sin(theta));
									int probableXCenterPlus = xImage + (int)((rad+minRadius)*cos(theta));
									int probableYCenterMinus = yImage - (int)((rad+minRadius)*sin(theta));
									int probableXCenterMinus = xImage - (int)((rad+minRadius)*cos(theta));
									if(probableYCenterMinus == yImage || probableXCenterMinus == xImage){

									}else{
										if(probableYCenterMinus>0 && probableYCenterMinus<input.rows && probableXCenterMinus>0 && probableXCenterMinus<input.cols){

											dartboardCenterVotes.at<double>(probableYCenterMinus,probableXCenterMinus) = dartboardCenterVotes.at<double>(probableYCenterMinus,probableXCenterMinus) + 1;
											//dartboardCenters.at<uchar>(probableYCenterMinus,probableXCenterMinus) = (uchar)255;
											if(dartboardCenterVotes.at<double>(probableYCenterMinus,probableXCenterMinus) > maxVote){
												maxVote = dartboardCenterVotes.at<double>(probableYCenterMinus,probableXCenterMinus);
												maxVoteX = probableXCenterMinus;
												maxVoteY = probableYCenterMinus;
											}
										}
										
									}
									if(probableYCenterPlus == yImage || probableXCenterPlus == xImage){

									}else{
										if(probableYCenterPlus>0 && probableYCenterPlus<input.rows && probableXCenterPlus>0 && probableXCenterPlus<input.cols){

											dartboardCenterVotes.at<double>(probableYCenterPlus,probableXCenterPlus) = dartboardCenterVotes.at<double>(probableYCenterPlus,probableXCenterPlus) + 1;
											//dartboardCenters.at<uchar>(probableYCenterPlus,probableXCenterPlus) = (uchar)255;
											if(dartboardCenterVotes.at<double>(probableYCenterPlus,probableXCenterPlus) > maxVote){
												maxVote = dartboardCenterVotes.at<double>(probableYCenterPlus,probableXCenterPlus);
												maxVoteX = probableXCenterPlus;
												maxVoteY = probableYCenterPlus;
											}

										}
									}
								}
							}
						}
					}
				}
			
			/*int threshold = maxVote - 100;
			for ( int yImage = 1; yImage < dartboardCenterVotes.rows; yImage++ )
				{	
					noOfDartBoards = 0;
					for( int xImage = 1 ; xImage < dartboardCenterVotes.cols; xImage++ )
					{ 
						if(dartboardCenterVotes.at<double>(yImage,xImage) >= threshold){
							cv::Point pt =  cv::Point(xImage,yImage);
							cv::circle(dartboardCenters, pt, 20, cv::Scalar( 255, 0, 0 ),10);
						}
					}
			}
			*/
			//std::cout << maxVote << " " << maxVoteX << " " << maxVoteY ;
			//cv::Point pt =  cv::Point(maxVoteX,maxVoteY);
			//cv::circle(dartboardCenters, pt, 20, cv::Scalar( 255, 0, 0 ),10);
			//output.at<cv::Scalar>(maxVoteY,maxVoteX) = cv::Scalar( 0, 255, 0 );
			//cv::namedWindow("DartBoards", CV_WINDOW_AUTOSIZE);
			//cv::imshow("DartBoards", dartboardCenters);
			//cv::waitKey();	
			vector<Vec3f> circles;

			// also preventing crash with hi, lo threshold here...
			Mat gray;
			gray = input.clone();
			GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
			HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 2, 60,50,25,50,100);
			int countCircles = 0;
			for( size_t i = 0; i < circles.size(); i++ )
			{
				countCircles++;
			}

			if(countCircles > 0 && maxVote >= 30){
				//return true;
				dartBoardCenter c ;
				c.x = maxVoteX;
				c.y = maxVoteY;
				c.rad = 50;
				return c;
			}else{
				//return false;
				dartBoardCenter c ;
				c.x = 0;
				c.y = 0;
				c.rad = 0;
				return c;
			}
		//}else{
			//return 0;
		//}
}