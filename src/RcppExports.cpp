// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// aperm2cv
NumericVector aperm2cv(NumericVector RMat, bool resize = true);
RcppExport SEXP opencv_aperm2cv(SEXP RMatSEXP, SEXP resizeSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericVector >::type RMat(RMatSEXP );
        Rcpp::traits::input_parameter< bool >::type resize(resizeSEXP );
        NumericVector __result = aperm2cv(RMat, resize);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// test_cv1
NumericVector test_cv1(NumericVector& RMat, bool show = true);
RcppExport SEXP opencv_test_cv1(SEXP RMatSEXP, SEXP showSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericVector& >::type RMat(RMatSEXP );
        Rcpp::traits::input_parameter< bool >::type show(showSEXP );
        NumericVector __result = test_cv1(RMat, show);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// test_cv2
NumericVector test_cv2(const NumericVector& RMat);
RcppExport SEXP opencv_test_cv2(SEXP RMatSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< const NumericVector& >::type RMat(RMatSEXP );
        NumericVector __result = test_cv2(RMat);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// thinningFromR
NumericVector thinningFromR(NumericVector RMat, int method = 0, double threshold = 0);
RcppExport SEXP opencv_thinningFromR(SEXP RMatSEXP, SEXP methodSEXP, SEXP thresholdSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericVector >::type RMat(RMatSEXP );
        Rcpp::traits::input_parameter< int >::type method(methodSEXP );
        Rcpp::traits::input_parameter< double >::type threshold(thresholdSEXP );
        NumericVector __result = thinningFromR(RMat, method, threshold);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// matchTemplate_cv
NumericVector matchTemplate_cv(const NumericVector& image, const NumericVector& templ, int method = 0);
RcppExport SEXP opencv_matchTemplate_cv(SEXP imageSEXP, SEXP templSEXP, SEXP methodSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< const NumericVector& >::type image(imageSEXP );
        Rcpp::traits::input_parameter< const NumericVector& >::type templ(templSEXP );
        Rcpp::traits::input_parameter< int >::type method(methodSEXP );
        NumericVector __result = matchTemplate_cv(image, templ, method);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// filter2D_cv
NumericVector filter2D_cv(const NumericVector& image, const NumericVector& kernel, NumericVector anchor, double delta = 0, int borderType = 4);
RcppExport SEXP opencv_filter2D_cv(SEXP imageSEXP, SEXP kernelSEXP, SEXP anchorSEXP, SEXP deltaSEXP, SEXP borderTypeSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< const NumericVector& >::type image(imageSEXP );
        Rcpp::traits::input_parameter< const NumericVector& >::type kernel(kernelSEXP );
        Rcpp::traits::input_parameter< NumericVector >::type anchor(anchorSEXP );
        Rcpp::traits::input_parameter< double >::type delta(deltaSEXP );
        Rcpp::traits::input_parameter< int >::type borderType(borderTypeSEXP );
        NumericVector __result = filter2D_cv(image, kernel, anchor, delta, borderType);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// GFiler2D_bf
NumericMatrix GFiler2D_bf(const NumericMatrix& A, const NumericMatrix& D, const NumericMatrix& K, const double p = 1);
RcppExport SEXP opencv_GFiler2D_bf(SEXP ASEXP, SEXP DSEXP, SEXP KSEXP, SEXP pSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< const NumericMatrix& >::type A(ASEXP );
        Rcpp::traits::input_parameter< const NumericMatrix& >::type D(DSEXP );
        Rcpp::traits::input_parameter< const NumericMatrix& >::type K(KSEXP );
        Rcpp::traits::input_parameter< const double >::type p(pSEXP );
        NumericMatrix __result = GFiler2D_bf(A, D, K, p);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
