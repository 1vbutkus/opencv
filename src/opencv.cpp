
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Rcpp.h>

using namespace Rcpp;


// ######################################################################################## //
// ################################### Extra code from others ############################# //
// ######################################################################################## //

// @author - Nashruddin Amin, see:http://opencv-code.com

/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 */


/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat& im, int iter)
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
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
 
void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}

/**
 * This is an example on how to call the thinning function above.
 */
int showmain()
{
    cv::Mat src = cv::imread("test_image.png");
    if (src.empty())
        return -1;

    cv::Mat bw;
    cv::cvtColor(src, bw, CV_BGR2GRAY);
    cv::threshold(bw, bw, 10, 255, CV_THRESH_BINARY);

    thinning(bw);

    cv::imshow("src", src);
    cv::imshow("dst", bw);
    cv::waitKey(0);

    return 0;
}




// #################################################################################### //
// ###################################### MINE CODE ################################### //
// #################################################################################### //








 
//' aaa
//' 
//' ...
//' 
//' \deqn{ v_p(f) = \sup \left\{ \sum_{i=1}^m |f(t_i) -
//' f(t_{i-1})|^p : 0=t_0<t_1<\dots<t_m=1 \right\} }{ v_p(f) =
//' sup { \sum |f(t_i) - f(t_{i-1})|^p : 0=t_0<t_1<\dots<t_m=1}
//' }
//' 
//' @return The vector of index of change points.
//' @param RMat \code{numeric} vector.
//' @param resize \code{numeric} vector.
//' @export
// [[Rcpp::export]]
NumericVector aperm2cv(NumericVector RMat, bool resize=true){
  // The order of cv:mat (on 2x2x2 input) goes like this: r1c1C1 r1c1C2 r1c2C1 ...
  // While the order of R (on same input): r1c1C1 r2c1C1 r1c2C1 ...
  // So in order of comunication, we need reorder  


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
    stop("The number of input dimesnions must be betwine 1 and 3.");   
  
  // independently on original dim, we set dim to be of length 3 (easyer in reording)
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

  // calling aperm function from R (migth be more efective way, I just want to awoid bugs)
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
 
  // checking if continuous (usaly the case), because we will pass data by ref.
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
  if(show)
    std::cout << "img = " << std::endl << " " << cvMat << std::endl << std::endl;
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

// [[Rcpp::export]]
int test_cv3(){
  std::cout << "CV_TM_SQDIFF = " << std::endl << " " << CV_TM_SQDIFF << std::endl << std::endl;
  std::cout << "CV_TM_SQDIFF_NORMED = " << std::endl << " " << CV_TM_SQDIFF_NORMED << std::endl << std::endl;
  std::cout << "CV_TM_CCORR = " << std::endl << " " << CV_TM_CCORR << std::endl << std::endl;
  std::cout << "CV_TM_CCORR_NORMED = " << std::endl << " " << CV_TM_CCORR_NORMED << std::endl << std::endl;
  std::cout << "CV_TM_CCOEFF = " << std::endl << " " << CV_TM_CCOEFF << std::endl << std::endl;
  std::cout << "CV_TM_CCOEFF_NORMED = " << std::endl << " " << CV_TM_CCOEFF_NORMED << std::endl << std::endl;
  return 0;
}




//' aaa
//' 
//' ...
//' 
//' asa
//' 
//' @return The vector of index of change points.
//' @param RMat \code{numeric} vector.
//' @param threshold \code{numeric} vector.
//' @export
// [[Rcpp::export(thinning)]]
NumericVector thinningFromR(NumericVector RMat, double threshold=10){
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
  
  cv::threshold(bw, bw, threshold, 255, CV_THRESH_BINARY);
  
  
//  cv::minMaxLoc(bw, &minVal, &maxVal, &minLoc, &maxLoc );
//  //std::cout << "img = " << std::endl << " " << cvMat << std::endl << std::endl;
//  std::cout << "bw.minVal = " << minVal << std::endl;
//  std::cout << "bw.maxVal = " << maxVal << std::endl;
//  std::cout << "bw.channels() = " << bw.channels() << std::endl;
  
  thinning(bw);
  
  // bw = cvMat;
  bw /= alpha;
  NumericVector res = CvMat2RMat(bw);
  return res;  
}






//' Linear 2d filter from opencv
//' 
//' ...
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
//' 
//' @return The vector of index of change points.
//' @param image \code{numeric} vector.
//' @param templ \code{numeric} vector.
//' @param method \code{numeric} vector.
//' @export
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


//' Linear 2d filter from opencv
//' 
//' ...
//' 
//' ...
//' 
//' @return The vector of index of change points.
//' @param image \code{numeric} vector.
//' @param kernel \code{numeric} vector.
//' @param anchor \code{numeric} vector.
//' @param delta \code{numeric} vector.
//' @export
// [[Rcpp::export]]
NumericVector filter2D_cv(const NumericVector& image, const NumericVector& kernel, NumericVector anchor, double delta=0){
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


  cv::filter2D(src, dst, CV_64F , krnl, anchorcv, delta, cv::BORDER_DEFAULT );
  
  /// back to R
  NumericVector res = CvMat2RMat(dst); 

  return res;
}



//' General form of 2d filter
//' 
//' ...
//' 
//' Gener form filter. Returns matrix \code{res} with values \code{res[i,j] = sum((A[si, sj] - D)^p)}.
//' 
//' @return The vector of index of change points.
//' @param A \code{numeric} vector.
//' @param D \code{numeric} vector.
//' @param K \code{numeric} vector.
//' @param p \code{numeric} vector.
//' @export
// [[Rcpp::export]]
NumericMatrix GFiler2D_bf(const NumericMatrix& A, const NumericMatrix& D,const NumericMatrix& K, const double p=1){
  double resval;
  int FR, FC, RR, RC;
  
  // D and K must match
  if ((D.nrow()!=K.nrow()) or (D.ncol()!=K.ncol())){
    stop("D and K dimenstion mast match!") ;
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