read.results <- function(save.path) {
    all.results <- data.frame()
    for (file.name in list.files(save.path, full.names = TRUE)) {
        load(file.name)
        all.results <- rbind(all.results, cv.results)
    }
    all.results
}

plot.cv.summary <- function(save.path, show = 1, threshold = 1) {
    # show only that best proportion of values (loss).
    # threshold the intensity to color the resulting plot.
    stopifnot(show >= 0 && show <=1)
    results <- read.results(save.path)

    exclude <- results$loss > quantile(results, probs = seq(0, 1, 0.01), na.rm = T)[round(100 * show)]
    exclude[is.na(exclude)] <- FALSE
    results <- results[!exclude, ]
    results$loss <- log(results$loss)

    library(MASS)
    n.colors <- 256
    colors <- colorRampPalette(c('black', rep('white', threshold)))(n.colors)
    parcoord(results, var.label = TRUE, col = colors[cut(results$loss, n.colors)])
}

plot.cv.summary(save.path, 0.5, 2)
