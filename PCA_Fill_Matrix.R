eliminate_na_cols <- function(data) {
  data[,colSums(is.na(data)) < dim(data)[1]]
}

fill_mean <- function(data) {
  missing <- which(is.na(data), arr.ind = T) 
  means <- colMeans(data, na.rm = T) 
  data[missing] <- means[missing[,2]] 
  return(data)
}

reconstruct_matrix <- function(data, K) {
  pca <- prcomp(data)
  Xapp <- pca$x[, 1:K] %*% t(pca$rotation[, 1:K]) 
  return(Xapp)
}

pca_matrix_fill <- function(data, K, th = 1e-7, last_label = T) { 
  
  data <- data %>% eliminate_na_cols
    
  if (last_label) {
    data <- data %>% select(-dim(data)[2]) 
  }
  
  working_mat <- fill_mean(data)
  
  na_locs <- is.na(data)
  means <- colMeans(data, na.rm = T)
  mssold <- mean((scale(data, means, F)[!na_locs])^2) 
  mss0 <- mean(data[!na_locs]^2)
  
  iter <- 0
  err <- 1
  
  while (err > th) {
    iter <- iter + 1
    
    pca <- reconstruct_matrix(working_mat, K) 
    working_mat[na_locs] <- pca[na_locs]
    mss <- mean(((data - pca)[!na_locs])^2)
    err <- abs(mssold - mss) / mss0
    mssold <- mss
    if (iter %% 50 == 0) {
      cat("Iter:", iter, "MSS:", mss, "Rel. Err:", err, "\n")
    }
  }
  
  miss <- which(na_locs, arr.ind = T) 
  for (i in 1:dim(miss)[1]) {
    cat("\n\nMissing value", i, "index:\n\n")
    print(miss[i,])

    cat("\nMean fill value:", means[miss[i,2]],"\n\n")
    cat("PCA fill value:", working_mat[miss[i,1], miss[i,2]],"\n\n")
  }
  
  return(working_mat) 
}