
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Rcpp.h>

using namespace Rcpp;


// ######################################################################################## //
// ################################### Extra code from others ############################# //
// ######################################################################################## //

// @author - Nashruddin Amin, see:http://opencv-code.com

/**
 * Code for thinning a binary image using Guo-Hall algorithm.
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningGuoHallIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

    for (int i = 1; i < im.rows; i++)
    {
        for (int j = 1; j < im.cols; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1); 
            uchar p9 = im.at<uchar>(i-1, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Code for thinning a binary image using Guo-Hall algorithm.
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinningGuoHall(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}


/**
 * Code for thinning a binary image using Zhang-Suen algorithm. (ZhangSuen)
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningZhangSuenIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Code for thinning a binary image using Zhang-Suen algorithm. (ZhangSuen)
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
 
void thinningZhangSuen(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningZhangSuenIteration(im, 0);
        thinningZhangSuenIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}


// #################################################################################### //
// ###################################### MINE CODE ################################### //
// #################################################################################### //


 
//' Aperm in C
//' 
//' Transform data ordering
//' 
//' Function \code{\link{aperm}} is generalization of transpose function (\code{\link{t}}) that works for arrays.
//' We need this function for communication between R array and cv:mat.
//' The order of data in cv:mat (on 2x2x2 input) goes like this: r1c1C1 r1c1C2 r1c2C1 ...
//' While the order of data in R (on same input): r1c1C1 r2c1C1 r1c2C1 ...
//' 
//' This function calls \code{\link{aperm}} function form C (it is not the most effective way, but it works).
//' 
//' Actually, we don't need this function in R, this it is loaded for debugging purposes.
//' 
//' @return An array with transformed data
//' @param RMat \code{numeric} array.
//' @param resize a flag indicating whether the vector should be resized as well as having its elements reordered (default \code{TRUE}).
//' @keywords internal
//' @export
// [[Rcpp::export]]
NumericVector aperm2cv(NumericVector RMat, bool resize=true){
  // The order of cv:mat (on 2x2x2 input) goes like this: r1c1C1 r1c1C2 r1c2C1 ...
  // While the order of R (on same input): r1c1C1 r2c1C1 r1c2C1 ...
  // So in order of communication, we need reorder  


  int nrow,  ncol, nchan;
  NumericVector dim;
  
  // Getting dim
  try {
    dim = RMat.attr("dim"); 
  }
  catch (...) {
    stop("The input must have `dim` attribute and `length(dim)` must be no greater then 3. ");
  }
  
  // checking dim
  if (dim.size()==1){
    nrow = dim[0];
    ncol = 1;
    nchan = 1;
  } else if (dim.size()==2){
    nrow = dim[0];
    ncol = dim[1];
    nchan = 1;
  } else if (dim.size()==3) {
    nrow = dim[0];
    ncol = dim[1];
    nchan = dim[2];
  } else 
    stop("The number of input dimensions must be between 1 and 3.");   
  
  // independently on original dim, we set dim to be of length 3 (easer in recording)
  NumericVector dumdim(3);
  dumdim[0] = nrow;
  dumdim[1] = ncol;
  dumdim[2] = nchan;
  RMat.attr("dim") = dumdim;

  // reordin
  NumericVector perm(3);
  perm[0] = 3;
  perm[1] = 2;
  perm[2] = 1;

  // calling aperm function from R (might be more effective way, I just want to avoid bugs)
  Function aperm("aperm");
  NumericVector res = aperm(RMat, Rcpp::Named("perm", perm), Rcpp::Named("resize", false));

  // fixing dimensions
  if(resize){
    NumericVector newdim(3);
    newdim[0] = dumdim[2];
    newdim[1] = dumdim[1];
    newdim[2] = dumdim[0];
    res.attr("dim") = newdim;  
  }
  // atsattom teisybe
  RMat.attr("dim")= dim; 
  
  return res;  
}


cv::Mat RMat2CvMat(const NumericVector& RMat){
  // From R matrix (Array) to opencv::Mat
  // RMat must have dim attribute (up to the length 3)

  // reordering
  NumericVector TrRMat = aperm2cv(RMat, false);
  NumericVector dim = TrRMat.attr("dim");

  // geting data
  double* p = &TrRMat[0];  
  int channels = dim[2];
  cv::Mat CvMat = cv::Mat(dim[0], dim[1], CV_64FC(channels), p); 

  return CvMat;
}


NumericVector CvMat2RMat(const cv::Mat& cvMat){
  // from cv::Mat to R matrix (Array)

  // converting to `double`
  cv::Mat cvMatRes; 
  cvMat.convertTo(cvMatRes, CV_64F); 
 
  // checking if continuous (usually the case), because we will pass data by ref.
  if (!cvMatRes.isContinuous()){
    cvMatRes = cvMatRes.clone();
  }
  
  // reshape mat to be one long row.
  cvMatRes = cvMatRes.reshape (1, 1);

  // taking data
  NumericVector data(cvMatRes.begin<double>(), cvMatRes.end<double>());
  
  // reordering to R order
  NumericVector dim(3);
  dim[2] = cvMat.rows;
  dim[1] = cvMat.cols;
  dim[0] = cvMat.channels(); 
  data.attr("dim") = dim;  
  
  // setting dim attribute
  NumericVector res = aperm2cv(data, false);
  if(cvMat.channels()>1){
    NumericVector newdim(3);
    newdim[0] = cvMat.rows;
    newdim[1] = cvMat.cols;
    newdim[2] = cvMat.channels();
    res.attr("dim") = newdim;    
  }else{
    NumericVector newdim(2);
    newdim[0] = cvMat.rows;
    newdim[1] = cvMat.cols;
    res.attr("dim") = newdim;    
  }
  
  return res;
}


// [[Rcpp::export]]
NumericVector test_cv1(NumericVector& RMat, bool show=true){
  cv::Mat cvMat = RMat2CvMat(RMat);
  if(show){
  //  std::cout << "img = " << std::endl << " " << cvMat << std::endl << std::endl;
  }
  NumericVector res = CvMat2RMat(cvMat);
  return res;
}

// [[Rcpp::export]]
NumericVector test_cv2(const NumericVector& RMat){
    cv::Mat cvMat = RMat2CvMat(RMat);
  
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    imwrite("test.png", cvMat, compression_params);

  
    NumericVector res = CvMat2RMat(cvMat);
  return res;
}


//' Thinning 
//' 
//' Thinning a binary image
//' 
//' This function performs thinning procedures for binary images. Two algorithms are implemented, namely:
//' Guo-Hall and Zhang-Suen lagorithms.
//' 
//' @author The core of code was written by Nashruddin Amin and posted in http://opencv-code.com blog.
//' @return The matrix of transformed image.
//' @param RMat a \code{matrix} of image.
//' @param method \code{numeric} 0 or 1. If \code{method==0} (the default) Guo-Hall algorithm is applied.  If \code{method==1} Zhang-Suen algorithm is applied.
//' @param threshold a threshold value that separates 0 and 1 (black and white). If \code{0} (the default) no threshold is applied (i.e. the values remains continuous) 
//' @export
//' @examples
//' if(require("png") & require("raster")){
//'   img <- img2grayscale(readPNG(system.file("pictures", "FatNote.png", package="opencv")))
//'   thinimg0 <- thinning(img, method=0)
//'   thinimg1 <- thinning(img, method=1)
//'   op = par(mfrow = c(3, 1), mar=c(2,2,2,1))
//'   plotImg(img, main="original")
//'   plotImg(thinimg0, main="thinning (method=0)")
//'   plotImg(thinimg1, main="thinning (method=1)")
//'   par(op)
//' }     
// [[Rcpp::export(thinning)]]
NumericVector thinningFromR(NumericVector RMat, int method=0, double threshold=0){
  // RMat must be single chanel


  double maxVal, minVal, alpha; 
  cv::Point minLoc, maxLoc; 
  
  cv::Mat bw;
  cv::Mat cvMat = RMat2CvMat(RMat);
  
  if (cvMat.channels()>1){
    stop("The RMat must be single channel.");
  }  
  
  cv::minMaxLoc(cvMat, &minVal, &maxVal, &minLoc, &maxLoc );
//  std::cout << "cvMat.minVal = " << minVal << std::endl;
//  std::cout << "cvMat.maxVal = " << maxVal << std::endl;
//  std::cout << "cvMat.channels() = " << cvMat.channels() << std::endl;
    
  
  
  if (maxVal<1.001){
    alpha = 255;
  }else if(maxVal>255){
    alpha = 255/maxVal;
  } else {
    alpha = 1;
  }
 
  cvMat.convertTo(bw, CV_8U, alpha=alpha); //CV_8UC1  the number of channels are the same as the input has
  
  if (threshold>0){
    cv::threshold(bw, bw, threshold, 255, CV_THRESH_BINARY);
  }
  
//  cv::minMaxLoc(bw, &minVal, &maxVal, &minLoc, &maxLoc );
//  //std::cout << "img = " << std::endl << " " << cvMat << std::endl << std::endl;
//  std::cout << "bw.minVal = " << minVal << std::endl;
//  std::cout << "bw.maxVal = " << maxVal << std::endl;
//  std::cout << "bw.channels() = " << bw.channels() << std::endl;
  
  
  if(method==0){
    thinningGuoHall(bw);
  }else if (method==1){
    thinningZhangSuen(bw);
  } else {
    stop("Unknown method.") ;
  }
  
  
  // bw = cvMat;
  bw /= alpha;
  NumericVector res = CvMat2RMat(bw);
  return res;  
}



//' matchTemplate
//' 
//' matchTemplate function from opencv library
//' 
//' This function calls matchTemplate function form opencv library. For special details see documentation of
//' opencv. 
//' 
//' The function slides through image, compares the overlapped patches against templ using 
//' the specified method and stores the comparison results in result . 
//' Here are the formula for the available comparison methods ( I denotes image, T template, R result ).
//' The method must be number from 0 to 5.  
//' 
//' \code{method=0}(SQDIFF)
//' \deqn{ R(x, y) = \sum (T(x', y') - I(x+x', y+y'))^2  }
//' 
//' \code{method=1}(SQDIFF_NORMED)
//' \deqn{ R(x, y) = \sum (T(x', y') - I(x+x', y+y'))^2 / \sqrt{\sum (T(x', y')^2 \sum I(x+x', y+y')^2 }}  
//' 
//' \code{method=2}(CCORR)
//' \deqn{ R(x, y) = \sum T(x', y')I(x+x', y+y')  }
//' 
//' \code{method=3}(CCORR_NORMED)
//' \deqn{ R(x, y) = \sum T(x', y')I(x+x', y+y') / \sqrt{\sum (T(x', y')^2 \sum I(x+x', y+y')^2 }}
//' 
//' \code{method=4}(CCOEFF)
//' \deqn{ R(x, y) = \sum T'(x', y')I'(x+x', y+y')  }
//' 
//' \code{method=5}(CCOEFF_NORMED)
//' \deqn{ R(x, y) = \sum T'(x', y')I'(x+x', y+y') / \sqrt{\sum (T'(x', y')^2 \sum I'(x+x', y+y')^2 }}
//' 
//' where 
//' \deqn{ T' = T - mean(T)} 
//' \deqn{ I'(x,y) = I(x,y) - mean(I, by=(x', y'))}
//' 
//' @seealso GFiler2D_bf
//' 
//' @return A 2D matrix with \eqn{R} values.
//' @param image \code{numeric} array (2D or 3D) of image data.
//' @param templ \code{numeric} array (2D or 3D) of image data. The size of \code{templ} must be not greater then \code{image}.
//' @param method \code{integer} value form 0 to 5. See details.
//' @export
//' @examples
//' if(require("png") & require("raster")){
//'   img <- readPNG(system.file("pictures", "minesweeper.png", package="opencv"))
//'   tm <- readPNG(system.file("pictures", "minesweeper_bomb.png", package="opencv"))
//'   op = par(mfrow = c(1, 2), mar=c(2,2,2,1))
//'   plotImg(img, main="image")
//'   plotImg(tm, main="template")
//'   par(op)
//'   filter <- matchTemplate_cv(img, tm, method=1)
//'   ids <- which(filter<=0.001)
//'   length(ids)
//'   rowids <- row(filter)[ids]
//'   colids <- col(filter)[ids]
//'   op = par(mfrow = c(1, 1), mar=c(2,2,2,1))
//'   plotImg(img, main="image")
//'   points(colids, -rowids, pch=19, col=3)
//'   par(op)
//' }
// [[Rcpp::export]]
NumericVector matchTemplate_cv(const NumericVector& image, const NumericVector& templ, int method=0){
  // bender to cv::matchTemplate function
  
  // checking
  if((method>6) or (method<0)){
    stop("method must be from 0 to 5");
  }
  
  // converting to approporite cv::mat object
  cv::Mat img = RMat2CvMat(image);
  cv::Mat tmp = RMat2CvMat(templ);  
  img.convertTo(img, CV_32F);
  tmp.convertTo(tmp, CV_32F);
  if(img.type() != tmp.type()){
    // std::cout << "img.type() = " << img.type() << std::endl; 
    // std::cout << "tmp.type() = " << tmp.type() << std::endl; 
    stop("the img matrix type do not match the tmp matrix type. It's probably the number of channels.");
  }
  
  
  
  /// Create the result matrix
  int result_cols =  img.cols - tmp.cols + 1;
  int result_rows = img.rows - tmp.rows + 1;
  if((result_cols<1) or (result_rows<1)){
    stop("Template image must be not greater then image. ");
  }   
  cv::Mat result;
  result.create( result_cols, result_rows, CV_32FC1 );
  
  /// Do the Matching 
  cv::matchTemplate(img, tmp, result, method );
  
  /// back to R
  NumericVector res = CvMat2RMat(result); 
  
  return res;
}


//' Linear 2D filter
//' 
//' Linear 2D filter from opencv
//' 
//' This function calls filter2D function from opencv. For special details see documentation of
//' opencv. 
//' 
//' The borderType must be in \code{0:4}. It is pixel extrapolation method in area of borders. The meaning of values:
//' \itemize{
//'   \item 0 BORDER_CONSTANT 	
//'   \item 1 BORDER_REPLICATE   
//'   \item 2 BORDER_REFLECT   
//'   \item 3 BORDER_WRAP   
//'   \item 4 BORDER_DEFAULT   
//' }
//' 
//' @return An array with the same dimensions as \code{image}. If you don't need borders, you have to crop it your self.
//' @param image a \code{matrix} of image (or other numeric data).
//' @param kernel a kernel \code{matrix}.
//' @param anchor anchor of the kernel that indicates the relative position of a filtered point within the kernel; the anchor should lie within the kernel; 
//' negalive values \code{c(-1,-1)} means that the anchor is at the kernel center.
//' @param delta optional value added to the filtered pixels before storing them in result matrix.
//' @param borderType pixel extrapolation method.
//' @export
//' @examples
//' if(require("png") & require("raster")){
//'   ### the meaning of borders
//'   img <- readPNG(system.file("pictures", "art.png", package="opencv"))
//'   gaus <- function(x, y, sigma=1) 1/(2*pi*sigma^2)*exp(-1/(2*sigma^2)*(x^2+y^2))
//'   fil <- outer(-30:30, -30:30, gaus, sigma=7)
//'   fil <- fil/sum(fil)
//'   plotImg(fil/max(fil), main="filter")
//'   f0 = filter2D_cv(img, kernel=fil, anchor=c(-1,-1), borderType=0)
//'   f1 = filter2D_cv(img, kernel=fil, anchor=c(-1,-1), borderType=1)
//'   f2 = filter2D_cv(img, kernel=fil, anchor=c(-1,-1), borderType=2)
//'   f3 = filter2D_cv(img, kernel=fil, anchor=c(-1,-1), borderType=3)
//'   f4 = filter2D_cv(img, kernel=fil, anchor=c(-1,-1), borderType=4)
//'   op = par(mfrow = c(3, 2), mar=c(2,2,2,1))
//'   plotImg(img, main="original")
//'   plotImg(f0, main="BORDER_CONSTANT (0)")
//'   plotImg(f1, main="BORDER_REPLICATE (1)")
//'   plotImg(f2, main="BORDER_REFLECT (2)")
//'   plotImg(f3, main="BORDER_WRAP (3)")
//'   plotImg(f4, main="BORDER_DEFAULT (4)")
//'   par(op)
//'   
//'   ### fat -> thin -> fat
//'   img <- img2grayscale(readPNG(system.file("pictures", "FatNote.png", package="opencv")))
//'   thinImg <- thinning(img)
//'   fil = matrix(c(1, 2, 1, 2, 4, 2, 1, 2, 1), 3, 3)/4
//'   plotImg(fil, main="filter")
//'   filterimg = filter2D_cv(thinImg, kernel=fil, anchor=c(-1,-1))
//'   filterimg[filterimg>1] = 1
//'   op = par(mfrow = c(3, 1), mar=c(2,2,2,1))
//'   plotImg(img, main="original")
//'   plotImg(thinImg, main="thin")
//'   plotImg(filterimg, main="back to fat")
//'   par(op)
//'} 
// [[Rcpp::export]]
NumericVector filter2D_cv(const NumericVector& image, const NumericVector& kernel, NumericVector anchor, double delta=0, int borderType=4){
  // bender
  
  cv::Mat src, dst, krnl;
  
  src = RMat2CvMat(image);
  krnl = RMat2CvMat(kernel);
  
  if(krnl.channels()>1){
    stop("kernel must be a matrix") ; 
  }
  if(anchor.size()!=2){
    stop("anchor must be of length 2.") ; 
  }  
  cv::Point anchorcv = cv::Point((int) anchor[0], (int) anchor[1]);
  
  if((borderType!=0) & (borderType!=1) & (borderType!=2) & (borderType!=3) & (borderType!=4)  ){
    stop("borderType must be in 0:4.") ;
  }

  cv::filter2D(src, dst, CV_64F , krnl, anchorcv, delta, borderType );
  
  /// back to R
  NumericVector res = CvMat2RMat(dst); 

  return res;
}



//' General form of 2D filter
//' 
//' Custom made 2D filter
//' 
//' General form filter. Returns matrix \code{res} with values \code{res[i,j] = sum(K*(A[si, sj] - D)^p)}.
//' This filtering is very not efficient (it written in brute force style). Nevertheless, it works good and it quite flexible.
//' It allows to match template using masks (see examples).
//' 
//' 
//' @seealso matchTemplate_cv
//' 
//' @return The \code{matrix}.
//' @param A big \code{matrix} that will be filtered.
//' @param D small \code{matrix} that will be used in difference.
//' @param K small \code{matrix} that will be used as kernel
//' @param p the power of differences.
//' @export
//' @examples
//' if(require("png") & require("raster")){
//'   img <-readPNG(system.file("pictures", "minesweeper.png", package="opencv"))
//'   tm <- readPNG(system.file("pictures", "minesweeper_bomb_transparent.png", package="opencv"))
//'   op = par(mfrow = c(1, 2), mar=c(2,2,2,1))
//'   plotImg(img, main="image")
//'   plotImg(tm, main="template")
//'   par(op)
//'   w <- tm[,,4]
//'   img2D <- img2grayscale(img)
//'   tm2D <- img2grayscale(tm)
//'   filter <- GFiler2D_bf(A=img2D, D=tm2D, K=w, p=2)
//'   ids <- which(filter<=0.001)
//'   length(ids)
//'   rowids <- row(filter)[ids]
//'   colids <- col(filter)[ids]
//'   op = par(mfrow = c(1, 1), mar=c(2,2,2,1))
//'   plotImg(img, main="image")
//'   # note that the `red` boob also matched. 
//'   points(colids, -rowids, pch=19, col=3) 
//'   par(op)
//' }
// [[Rcpp::export]]
NumericMatrix GFiler2D_bf(const NumericMatrix& A, const NumericMatrix& D,const NumericMatrix& K, const double p=1){
  double resval;
  int FR, FC, RR, RC;
  
  // D and K must match
  if ((D.nrow()!=K.nrow()) or (D.ncol()!=K.ncol())){
    stop("D and K dimensions mast match!") ;
  }
  FR = D.nrow();
  FC = D.ncol();

  //F.dim<A.dim
  RR = A.nrow() - FR + 1;
  RC = A.ncol() - FC + 1;
  if ((RR<1) or (RC<1)){
    stop("Dimension of result is negative! Your filter inputs(D and K) are to large for A.") ;    
  }
  
  NumericMatrix res(RR, RC);
  for(int i=0; i<RR; i++){
    for(int j=0; j<RC; j++){
      // Start filter of particular point
      resval = 0;
      for(int k=0; k<FR; k++){
        for(int l=0;l<FC;l++){
          resval += std::pow(A(i+k, j+l) - D(k, l), p)*K(k, l);
        }
      }      
      res(i, j) = resval;      
    }    
  }
  
  return res;
}
