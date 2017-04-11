source('src/R/DeepR/DeepR.R')

input <- scale(iris[-5])
labels <- matrix(c(iris[5] == 'setosa', iris[5] == 'virginica', iris[5] == 'versicolor'), ncol = 3)

n.iter.range <- c(2, 4)
alpha.range <- c(-1, -4)
lambda.range <- c(-3, 3)
breadth.vec <- (1 * ncol(input)):(4 * ncol(input))
depth.vec <- 0:3
n.samples <- 10

sample.param <- function(range) runif(1, min = 10^range[1], max = 10^range[2])

cv.results <- data.frame()
for (i in 1:n.samples) {
    n.iter <- round(sample.param(n.iter.range))
    alpha <- sample.param(alpha.range)
    lambda <- sample.param(lambda.range)
    breadth <- sample(breadth.vec, 1)
    depth <- sample(depth.vec, 1)
    
    layers <- c(ncol(input), rep(breadth, depth), ncol(labels))
    model <- train(layers, input, labels, n.iter, alpha, lambda)
    cv <- test(input, model, labels, lambda)
    cv.results <- rbind(cv.results, c(n.iter, alpha, lambda, breadth, depth, cv$loss, cv$accuracy))
}
names(cv.results) <- c('n.iter', 'alpha', 'lambda', 'breadth', 'depth', 'loss', 'accuracy')
cv.results