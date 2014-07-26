context("C and R compatibility")


test_that("aperm2cv", {

  v = 1:8  
  dim(v) = c(2,4)
  vv = v
  dim(vv) = c(2,4,1)
  expect_that(aperm2cv(v, TRUE), equals(aperm(vv, 3:1, resize=TRUE)))
  expect_that(aperm2cv(v, FALSE), equals(aperm(vv, 3:1, resize=FALSE)))

  v = 1:8  
  dim(v) = c(2, 2, 2)
  vv = v
  expect_that(aperm2cv(v, TRUE), equals(aperm(vv, 3:1, resize=TRUE)))
  expect_that(aperm2cv(v, FALSE), equals(aperm(vv, 3:1, resize=FALSE)))
  
  expect_that(aperm2cv(1:8), throws_error())
  
})

test_that("R <-> C", {
  
  v = 1:8  
  dim(v) = c(2,4)
  expect_that(test_cv1(v, FALSE), equals(v))
  
  v = 1:8  
  dim(v) = c(2,4,1)
  expect_that(test_cv1(v, FALSE), not(equals(v)))
  expect_that(dim(test_cv1(v, FALSE)), equals(c(2,4)))  
  
})




# 
# test_that("raster vs C (round 1)", {
#   N = 5
#   n = 2
#   A = matrix(rnorm(N^2), N, N)
#   D = matrix(1, n, n)
#   K = matrix(2, n, n)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
#   D = matrix(1, n, n+2)
#   K = matrix(2, n, n+2)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
#   D = matrix(1, n, n+1)
#   K = matrix(2, n, n+1)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
#   D = matrix(1, n+1, n)
#   K = matrix(2, n+1, n)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
#   D = matrix(1, n+1, n+1)
#   K = matrix(2, n+1, n+1)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
# })
# 
# test_that("raster vs C (round 2)", {
#   N = 15
#   M = 20
#   n = 5
#   m = 4
#   A = matrix(runif(N*M), N, M)
#   D = matrix(runif(n*m), n, m)
#   K = matrix(runif(n*m), n, m)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
#   p = 2
#   expect_that(filter2dRaster(A, D, K, p), equals(filter2dC(A, D, K, p)))
#   q=1/3
#   expect_that(filter2dRaster(A, D, K, p, q), equals(filter2dC(A, D, K, p, q)))
#   
#   D = matrix(0, n, m)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
# 
#   
#   
#   N = 201
#   M = 200
#   n = 16
#   m = 21
#   A = matrix(rnorm(N*M), N, M)
#   D = matrix(rnorm(n*m), n, m)
#   K = matrix(rnorm(n*m), n, m)
#   expect_that(filter2dRaster(A, D, K), equals(filter2dC(A, D, K)))
#   
# })
# 
# test_that("matchtemplate basic", {
#   img <- img(system.file("pictures", "minesweeper.png", package="imagebrowser"))
#   tm <- img(system.file("pictures", "minesweeper_bomb.png", package="imagebrowser"))
#   tmt <- img(system.file("pictures", "minesweeper_bomb_transparent.png", package="imagebrowser"))
#   
#   expect_that(fil <- matchtemplate(img, tm, method="SQDIFF_NORMED"), not(throws_error()))
#   expect_that(fil <- matchtemplate(img, tm, method="SQDIFF"), not(throws_error()))
#   expect_that(fil <- matchtemplate(img, tm, method="CCOEFF_NORMED"), not(throws_error()))
# 
# 
#   expect_that(cor1 <- findimg(img, tm, method="SQDIFF_NORMED"), not(throws_error()))
#   expect_that(cor2 <- findimg(img, tm, method="SQDIFF"), not(throws_error()))
#   expect_that(cor3 <- findimg(img, tm, method="CCOEFF_NORMED"), not(throws_error()))
#   expect_that(cor1, equals(cor2))
#   expect_that(cor1, equals(cor3))
# 
# 
#   expect_that(cor1 <- findimg_all(img, tm, method="SQDIFF_NORMED"), not(throws_error()))
#   expect_that(cor2 <- findimg_all(img, tm, method="SQDIFF"), not(throws_error()))
#   expect_that(cor3 <- findimg_all(img, tm, method="CCOEFF_NORMED"), not(throws_error()))
#   expect_that(cor1, equals(cor2))
#   expect_that(cor1, equals(cor3))
# 
#   expect_that(cor1 <- findimg_all(img, tmt, method="SQDIFF_NORMED"), not(throws_error()))
#   expect_that(cor2 <- findimg_all(img, tmt, method="SQDIFF"), not(throws_error()))
#   expect_that(cor3 <- findimg_all(img, tmt, method="CCOEFF_NORMED"), not(throws_error()))
#   expect_that(cor1, equals(cor2))
#   expect_that(cor1, equals(cor3))
#   
# 
#   cat <- img(system.file("pictures", "cat.png", package="imagebrowser"))
#   tm <- img(system.file("pictures", "cat_template.png", package="imagebrowser"))  
# #   existimg(cat, tm, method="SQDIFF_NORMED")
# #   existimg(cat, tm, method="SQDIFF")
# #   existimg(cat, tm, method="CCOEFF_NORMED")
# 
#   
#   
# })
# 
# 
# 
