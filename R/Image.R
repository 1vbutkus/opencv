#' Rescale values
#' 
#' Rescale values in to match new range.
#' 
#' Function rescales the values to have a new range.
#'
#' @param x a data object. Can be vector, matrix array.
#' @param newrange the desired range. 
#' @export
#' @examples
#' x <- runif(100)
#' y <- rescale(x, c(-2, 2))
#' op <- par(mfrow = c(2, 1), mar=c(3,3,2,1))
#' hist(x) 
#' hist(y) 
#' par(op)
rescale <- function(x, newrange) {
  if (nargs() > 1 && is.numeric(x) && is.numeric(newrange)) {
    # if newrange has max first, reverse it
    if (newrange[1] > newrange[2]) {
      newmin <- newrange[2]
      newrange[2] <- newrange[1]
      newrange[1] <- newmin
    }
    xrange <- range(x)
    if (xrange[1] == xrange[2]) 
      stop("can't rescale a constant vector!")
    mfac <- (newrange[2] - newrange[1])/(xrange[2] - xrange[1])
    invisible(newrange[1] + (x - xrange[1]) * mfac)
  } else {
    cat("Usage: rescale(x,newrange)\n")
    cat("\twhere x is a numeric object and newrange is the min and max of the new range\n")
  }
}



#' Plot image
#' 
#' Plot image
#' 
#' Plot an image using a functions from `raster` package \code{rasterImage}.
#'
#' @param x an array of image data (can be 2D or 3D).
#' @param xlab the \code{xlab} paramiter in \code{plot} function.
#' @param ylab the \code{ylab} paramiter in \code{plot} function.
#' @param asp the \code{asp} paramiter in \code{plot} function.
#' @param \dots futher argumets pass to plot.
#' @export 
#' 
#' @examples
#' if(require("png") & require("raster")){
#'   img <- readPNG(system.file("pictures", "FatNote.png", package="opencv"))
#'   plotImg(img)
#' }
plotImg <- function(x, xlab = "", ylab = "", asp = 1, ...) {
  if(!require("raster")){
    stop("`raster` package is required for plotting")
  }
  
  if (length(dim(x)) < 2) 
    stop("x must have at least two dimensions.")
  
  if (length(dim(x)) > 3) 
    stop("x must have no more then 3 dimensions.")
  
  if (length(dim(x)) == 3) {
    if (dim(x)[3] != 3 & dim(x)[3] != 4) 
      stop("deep dimension must be 3 or 4")
  }

  # initial plot
  plot(c(0, dim(x)[2]), c(0, -dim(x)[1]), type = "n", xlab = xlab, ylab = ylab, asp = asp, ...)
  
  
  # transparency background
  if (isTRUE(dim(x)[3] == 4)) {
    bord <- matrix(grey(0.6), dim(x)[1], dim(x)[2])
    bord[(floor(0.5 + row(bord)/2) + floor(0.5 + col(bord)/2))%%2 == 1] <- grey(0.8)
    rasterImage(bord, xleft = 0, ybottom = -dim(x)[1], xright = dim(x)[2], ytop = 0, interpolate = FALSE)
  }
  
  
  # rescale
  xrange <- range(x)
  if (xrange[1] < 0 | xrange[2] > 1) {
    if (xrange[1] != xrange[2]) {
      x[] <- rescale(x, 0:1)
    } else {
      x[] <- 0
    }
  }
  
  # actual plot
  class(x) <- NULL
  rasterImage(x, xleft = 0, ybottom = -dim(x)[1], xright = dim(x)[2], ytop = 0, interpolate = FALSE)
}


#' Convert img to grayscale
#' 
#' Convert img to grayscale
#' 
#' This function ensures that the image is 2D matrix (rather then 3D).  
#'
#' @return Image matrix
#'
#' @param img image array. Can be 2D or 3D.
#' @export 
#' @examples
#' if(require("png") & require("raster")){
#'   op = par(mfrow = c(1, 2))
#'   img <- readPNG(system.file('pictures', 'cat.png', package='opencv'))
#'   plotImg(img, main='original') 
#'   plotImg(img2grayscale(img), main='grayscale')
#'   par(op)
#' }
img2grayscale <- function(img) {
  if (length(dim(img)) == 3) {
    if (dim(img)[3] >= 3) {
      res <- as.matrix(0.3 * img[, , 1] + 0.5 * img[, , 2] + 0.2 * img[, , 3])
    } else {
      res <- as.matrix(img[, , 1])
    }
  } else {
    res <- as.matrix(img)
  }  
  res
}






