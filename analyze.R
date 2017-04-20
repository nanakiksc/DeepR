read.results <- function(save.path) {
    all.results <- data.frame()
    for (file.name in list.files(save.path, full.names = TRUE)) {
        load(file.name)
        all.results <- rbind(all.results, cv.results)
    }
    all.results
}

plot.cv.summary <- function(save.path) {
    all.results <- read.results(save.path)

    exclude <- all.results$loss > quantile(all.results, probs = seq(0, 1, 0.01), na.rm = T)[98]
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

save.path <- 'Dropbox/iris_test/'
plot.cv.summary(save.path)
