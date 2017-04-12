source('src/R/DeepR/DeepR.R')

input <- scale(iris[-5])
labels <- matrix(c(iris[5] == 'setosa', iris[5] == 'virginica', iris[5] == 'versicolor'), ncol = 3)

save.path <- '~/Dropbox/iris_test/'
prefix <- 'iris_test_'

n.iter.range <- c(2, 3)
alpha.range <- c(-4, -1)
lambda.range <- c(-3, 0)
breadth.vec <- (1 * ncol(input)):(4 * ncol(input))
depth.vec <- 0:3
n.samples <- 10

sample.param <- function(range) runif(1, min = 10^min(range), max = 10^max(range))

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
save(cv.results, file = paste0(save.path, prefix, gsub('-|:| ', '', Sys.time()), '.rda'))

plot.cv.summary <- function(save.path) {
    all.results <- data.frame()
    for (file.name in list.files(save.path, full.names = TRUE)) {
        load(file.name)
        all.results <- rbind(all.results, cv.results)
    }

    exclude <- all.results$loss > 1e3
    exclude[is.na(exclude)] <- FALSE
    all.results <- all.results[!exclude, ]

    plot.results <- list()
    for (name in names(all.results)[1:5]) {
        plot.results[[name]] <- all.results[c(name, 'loss', 'accuracy')]
        plot.results[[name]] <- plot.results[[name]][order(plot.results[[name]][name]), ]
    }

    par(mfrow = c(2, 3))
    for (name in names(all.results)[1:5]) {
        d <- plot.results[[name]]
        plot(d[, name], d$loss, xlab = name, ylab = 'Loss', ylim = c(0, max(d$loss, na.rm = TRUE)), mgp = c(2, 1, 0))
        par(new = TRUE)
        plot(d[, name], d$accuracy, col = 2, xlab = NA, ylab = NA, ylim = c(0, 1), xaxt = 'n', yaxt = 'n')
        axis(4, col = 2, col.axis = 2)
        mtext('Accuracy', 4, 2, col = 2, cex = par('cex'))
    }
}

# Analyze results.
plot.cv.summary(save.path)
