read.results <- function(save.path) {
    all.results <- data.frame()
    for (file.name in list.files(save.path, full.names = TRUE)) {
        load(file.name)
        all.results <- rbind(all.results, cv.results)
    }
    all.results
}

plot.cv.summary <- function(save.path, name = 'loss', show = 1, threshold = 1) {
    # `show` only that best proportion of name values.
    # `threshold` the intensity to color the resulting plot.
    stopifnot(show >= 0 && show <=1)
    results <- read.results(save.path)

    exclude <- results[[name]] > quantile(results, probs = seq(0, 1, 0.01), na.rm = T)[round(100 * show)]
    exclude[is.na(exclude)] <- FALSE
    results <- results[!exclude, ]
    # results[[name]] <- log(results[[name]])

    library(MASS)
    n.colors <- 256
    colors <- colorRampPalette(c(rep('white', threshold), 'black'))(n.colors)
    parcoord(results, var.label = TRUE, col = colors[cut(results[[name]], n.colors)])
}

plot.cv.summary(save.path, 0.5, 2)

