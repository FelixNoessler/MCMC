library(ggplot2)
library(dplyr)


n <- 40
S <- matrix(c(100*100, 0.8*100*200,
              0.8*100*200, 200*200), 
            nrow = 2, ncol = 2, byrow = TRUE)
            
mu <- c(X = 1000, Y = 2000)
random_samples <- MASS::mvrnorm(n, mu = mu, Sigma = S)


weights <- random_samples[, 1] ^ 20 #runif(n, 1, 100)
weights_std <- weights / sum(weights)

df <- tibble(x = random_samples[, 1], y = random_samples[, 2], weights)
ggplot(df, aes(x, y, size = weights_std)) +
  geom_point(alpha = 0.3, color = "orange") +
  scale_size_continuous() +
  theme_classic()


## estimate mv normal with weights

weighted_mean <- function(X, w) colSums(X * w) / sum(w) 
weighted_cov <- function(X, w) t(X - weighted_mean(X, w)) %*% (diag(w) %*% (X - weighted_mean(X, w))) / sum(w)

weighted_cov(random_samples, weights)
weighted_mean(data[, 1], weights)

# weighted mean
data <- random_samples
weighted_mean <- colSums(data * weights_std) 
colMeans(data)
# weighted covariance
weighted_cov <- t(data - weighted_mean) %*% (diag(weights) %*% (data - weighted_mean)) / sum(weights)

# library(MASS)
# # Generate contours from fitted parameters
grid <- expand.grid(x = seq(0.5*min(data[,1]), 1.5*max(data[,1]), length.out = 50),
                    y = seq(0.5*min(data[,2]), 1.5*max(data[,2]), length.out = 50))

grid$density_values <- mapply(function(x, y) {
  mvtnorm::dmvnorm(cbind(x, y), mean = weighted_mean, sigma = weighted_cov)
}, grid$x, grid$y)


ggplot(grid, aes(x,y)) +
    geom_raster(aes(fill = density_values)) +
    theme_classic() 




# contour(matrix(density_values, nrow = 100), x = seq(min(data[,1]), max(data[,1]), length.out = 100),
#         y = seq(min(data[,2]), max(data[,2]), length.out = 100),
#         xlab = "X1", ylab = "X2", main = "Fitted Multivariate Normal Contours")

# plot
library(ggforce)
ggplot(df, aes(x, y, size = weight)) +
  geom_point(alpha = 0.3, color = "orange") +
  scale_size_continuous(range = c(1, 10)) +
  theme_classic() +
  annotate(x = weighted_mean[1], y = weighted_mean[2], color = "red", size = 5, geom = "point") +
  geom_ellipse(aes(x0 = weighted_mean[1], y0 = weighted_mean[2], 
                   a = sqrt(weighted_cov[1, 1]), b = sqrt(weighted_cov[2, 2]),
                   angle = atan(weighted_cov[1, 2] / weighted_cov[1, 1]) * 180 / pi),
               color = "red", linetype = 2)
