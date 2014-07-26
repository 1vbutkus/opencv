#' opencv
#' 
#' This package is a beneder to well known Computer Vision library \code{opencv}. 
#' The primary goal of this package to have some opencv functions in R.
#' 
#' \tabular{ll}{
#' Package: \tab opencv\cr
#' Type: \tab Package\cr
#' Version: \tab 1\cr
#' Date: \tab 2014-08-30\cr
#' License: \tab GPL-2\cr
#' System requremnt: \tab opencv library, see ...\cr
#' }
#'
#' Up till now, there is only two functions from opencv that is implimented in R, namely
#' \code{\link{matchTemplate_cv}} and \code{\link{filter2D_cv}}.
#' The new functions will come in place according to needs.
#' 
#'   
#' @author Author and Maintainer: Vygantas Butkus <Vygantas.Butkus@@gmail.com>.
#' Well, I am a numby in C++, therefore, any contribution from someone who actually knows that he is doing would
#' be apriseted. If you have ambitions extend this package, maintain ir or have new functions that will fit to this package 
#' - please contact me. 
#' 
#' @seealso
#' Two main functions are \code{\link{matchTemplate_cv}} and \code{\link{filter2D_cv}}.
#' 
#' @encoding utf8 
#' @import Rcpp
#' @docType package
#' @name opencv-package
#' @useDynLib opencv
#' @aliases opencv-package
NULL 
