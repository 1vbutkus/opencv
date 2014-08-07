#' opencv
#' 
#' This package is a bender to well known computer vision library \code{opencv}. 
#' The primary goal of this package to have some \code{opencv} functions in R.
#' In addition, some extra functions (on top of \code{opencv} functions) are included.
#' 
#' \tabular{ll}{
#' Package: \tab opencv\cr
#' Type: \tab Package\cr
#' Version: \tab 1\cr
#' Date: \tab 2014-08-30\cr
#' License: \tab GPL-2\cr
#' System requirement: \tab opencv library, see http://opencv.org/\cr
#' }
#'
#' Up till now, there is only two functions from \code{opencv} that is implemented in R, namely
#' \code{\link{matchTemplate_cv}} and \code{\link{filter2D_cv}}.
#' The new functions will come in place according to needs.
#' 
#' In addition, same very useful code written by  Nashruddin Amin ( see:http://opencv-code.com )
#' was implemented. The code is for thinning a binary image using
#' Guo-Hall and Zhang-Suen algorithms, see \code{\link{thinning}}.
#' 
#'   
#' @author Author and Maintainer: Vygantas Butkus <Vygantas.Butkus@@gmail.com>.
#' I belive that full opencv package can be usefull for quite a numer of R useres.
#' If you have ambitions extend this package, maintain it or have new functions that will 
#' fit to this package - please contact me. 
#' 
#' @seealso
#' Main functions are \code{\link{matchTemplate_cv}}, \code{\link{filter2D_cv}}, \code{\link{thinning}}.
#' 
#' @encoding utf8 
#' @import Rcpp
#' @docType package
#' @name opencv-package
#' @useDynLib opencv
#' @aliases opencv-package
NULL 
