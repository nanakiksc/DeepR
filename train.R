source('src/R/DeepR/DeepR.R')

input <- iris[-5]
labels <- matrix(c(iris[5] == 'setosa', iris[5] == 'virginica', iris[5] == 'versicolor'), ncol = 3)

idx <- sample(nrow(input), nrow(input) * 0.8)
train.set <- scale(input[idx, ])
test.set <- scale(input[-idx,], center = attr(train.set, 'scaled:center'), scale = attr(train.set, 'scaled:scale'))
train.labels <- labels[idx, ]
test.labels <- labels[-idx,]

n.iter.range <- c(2, 4)
alpha.range <- c(-5, -3)
mu.vec <- c(0.5, 0.9, 0.95, 0.99) # TODO: Increase over time?
lambda.range <- c(-3, 0)
breadth.range <- c(0, 5)
depth.vec <- 1:4
n.samples <- 10

sample.param <- function(range) runif(1, min = min(range), max = max(range))

cv.results <- data.frame()
for (i in 1:n.samples) {
    n.iter <- round(10^sample.param(n.iter.range))
    alpha <- 10^sample.param(alpha.range)
    mu <- sample(mu.vec, 1)
    lambda <- 10^sample.param(lambda.range)
    breadth <- round(ncol(input) * 2^sample.param(breadth.range))
    depth <- sample(depth.vec, 1)

    layers <- c(ncol(input), rep(breadth, depth), ncol(labels))
    model <- train(layers, train.set, train.labels, n.iter, alpha, mu, lambda)
    cv <- test(model, test.set, test.labels, lambda)
    cv.results <- rbind(cv.results, c(log10(n.iter), log10(alpha), mu, log10(lambda), log2(breadth / ncol(input)), depth, cv$loss, cv$accuracy))
}
names(cv.results) <- c('n.iter', 'alpha', 'mu', 'lambda', 'breadth', 'depth', 'loss', 'accuracy')

save.path <- '~/Dropbox/iris_test/'
# prefix <- ''
prefix <- 'iris_test_'
# prefix <- paste0(sub('_*$', '', prefix), '_')
save(cv.results, file = paste0(save.path, prefix, gsub('-|:| ', '', Sys.time()), '.rda'))
