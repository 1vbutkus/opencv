context("proc")


test_that("filter2d_cv", {
  
  if(require("png")){
    
    img <- readPNG(system.file("pictures", "minesweeper.png", package="imagebrowser"))
    tm <- readPNG(system.file("pictures", "minesweeper_bomb.png", package="imagebrowser"))
    
    img = (img[,,1]+img[,,2]+img[,,3])/3
    tm = (tm[,,1]+tm[,,2]+tm[,,3])/3
    
    D = matrix(0, dim(tm)[1], dim(tm)[2])
    
    r1 = matchTemplate_cv(img, tm, method=2)
    r2 = filter2D_cv(img, tm, c(0,0))
    r2 = r2[1:nrow(r1), 1:ncol(r1)]
    r3 = GFiler2D_bf(A=img, D=D,  K=tm)
     
    expect_that(max(abs(r1-r2))<0.1^5,  is_true())
    expect_that(max(abs(r1-r3))<0.1^5,  is_true())
    expect_that(max(abs(r3-r1))<0.1^5,  is_true())
    
  }else{
    warning("package png is not found.")
  }
})
