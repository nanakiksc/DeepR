source('DeepR.R')

layers <- c(4, 8, 3)
input <- scale(iris[-5])
labels <- matrix(c(iris[5] == 'setosa', iris[5] == 'virginica', iris[5] == 'versicolor'), ncol = 3)
model <- train(layers, input, labels, n.iter = 1e3, alpha = 1e-3, lambda = 0)
mean(apply(round(test(input, model, labels, lambda = 0)$h) == labels, 1, all))
