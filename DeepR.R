# TODO: Add Nesterov momentum to Adam (Nadam).
# TODO: customize last.layer(): define it as part of the model (and therefore test()).

init.model <- function(layers, seed = NULL, neuron.type = 'ReLU', scale.method = 'He', dropout = 0.5, dropout.input = 0.8, lambda = 0) {
    if (!is.null(seed)) set.seed(seed)

    model <- list(weights = list(), biases = list())
    model <- choose.neuron(model, neuron.type)
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]
        ncol <- layers[i + 1]
        model$weights[[i]] <- matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
        model$weights[[i]] <- scale.weights(model$weights[[i]], scale.method)
        model$biases[[i]]  <- rep(0, ncol)
        model$m.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$m.biases[[i]]  <- rep(0, ncol)
        model$v.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$v.biases[[i]]  <- rep(0, ncol)
    }
    model$iteration <- 0
    model$dropout <- dropout
    model$dropout.input <- dropout.input
    model$lambda <- lambda

    model
}

train <- function(model, input, labels, n.iter = 1e3, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, batch.size = nrow(input)) {
    input <- scale(as.matrix(input))
    batch.size <- min(batch.size, nrow(input))
    attr(model, 'scaled:center') <- attr(input, 'scaled:center')
    attr(model, 'scaled:scale') <- attr(input, 'scaled:scale')

    for (i in 1:n.iter) {
        idx <- sample(nrow(input), batch.size)
        model <- forward.propagation(input[idx, ], model)
        last.deltas <- last.layer(model, labels[idx, ])$d
        model <- backpropagation(model, last.deltas)
        model <- update.model(model, alpha, beta1, beta2, epsilon)
    }
    model
}

test <- function(model, input, labels) {
    input <- as.matrix(input)
    input <- scale(as.matrix(input), center = attr(model, 'scaled:center'), scale = attr(model, 'scaled:scale'))

    model <- forward.propagation(input, model, dropout = FALSE)
    hypothesis <- last.layer(model, labels)$h
    # loss <- mean((hypothesis - labels)^2) / 2 # MSE (regression) loss.

    # Cross-entropy (logistic) loss.
    loss <- -labels * log(hypothesis) - (1 - labels) * log(1 - hypothesis)
    loss[is.nan(loss)] <- 0
    loss <- mean(loss)

    loss <- loss + model$lambda * sum(unlist(model$weights)^2) / (2 * nrow(model$neurons$z[[length(model$neurons$z)]])) # Add L2 regularization term.
    accuracy <- mean(apply(round(hypothesis) == labels, 1, all))
    list(hypothesis = hypothesis, loss = loss, accuracy = accuracy)
}

choose.neuron <- function(model, neuron.type) {
    if (neuron.type == 'ReLU') {
        model$activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.
        model$gradient   <- function(z) z > 0 # Faster than ifelse(z > 0, 1, 0).
    } else if (neuron.type == 'sigmoid') {
        model$activation <- function(z) 1 / (1 + exp(-z))
        model$gradient   <- function(z) { s <- activation(z); s * (1 - s) }
    } else if (neuron.type == 'tanh') {
        model$activation <- function(z) tanh(z)
        model$gradient   <- function(z) 1 - tanh(z)^2
    } else {
        print('Unknown activation function. Please choose ReLU, sigmoid or tanh.')
        stop()
    }

    model
}

scale.weights <- function (weights, scale.method = 'He') {
    if      (scale.method == 'He')     weights * sqrt(2 / nrow(weights)) # He et al. adjusted initialization for deep ReLU networks.
    else if (scale.method == 'Xavier') weights * sqrt(2 / sum(dim(weights))) # Xavier initializarion for deep networks.
    else if (scale.method == 'Caffe')  weights * sqrt(nrow(weights)) # Caffe version of Xavier initialization.
    else if (scale.method == 'none')   weights # Like... no scaling. Why would you do that, right?
    else { print('Unknown wheight initialization method. Please choose He, Xavier, Caffe or none.'); stop() }
}

forward.propagation <- function(input, model, dropout = TRUE) {
    model$neurons$a[[1]] <- input
    if (dropout) model$neurons$a[[1]] <- dropout.mask(model, 1)
    n.weights <- length(model$weights)
    if (n.weights > 1) {
        for (i in 2:n.weights) {
            weights <- model$weights[[i - 1]]
            if (!dropout) weights <- weights * if (i == 2) model$dropout.input else model$dropout # Input layer has different dropout prob.
            model$neurons$z[[i]] <- sweep(model$neurons$a[[i - 1]] %*% weights, 2, model$biases[[i - 1]], '+')
            model$neurons$a[[i]] <- model$activation(model$neurons$z[[i]])
            if (dropout) model$neurons$a[[i]] <- dropout.mask(model, i)
        }
    }
    # Last layer always contains all neurons.
    weights <- model$weights[[n.weights]]
    if (!dropout) weights <- weights * model$dropout
    model$neurons$z[[n.weights + 1]] <- sweep(model$neurons$a[[n.weights]] %*% weights, 2, model$biases[[n.weights ]], '+')

    model
}

dropout.mask <- function(model, layer) {
    nrow <- nrow(model$neurons$a[[layer]])
    ncol <- ncol(model$neurons$a[[layer]])
    dropout <- if (layer == 1) model$dropout.input else model$dropout
    mask <- matrix(sample(0:1, nrow * ncol, replace = TRUE, prob = c(1 - dropout, dropout)), nrow = nrow, ncol = ncol)
    model$neurons$a[[layer]] * mask
}

last.layer <- function(model, labels) {
    # Compute last layer activations (hypothesis). It must return its deltas.
    # hypothesis <- model$neurons$z[[length(model$neurons$z)]] # Linear activation.
    hypothesis <- 1 / (1 + exp(-model$neurons$z[[length(model$neurons$z)]])) # Sigmoid activation.
    # hypothesis <- tanh(model$neurons$z[[length(model$neurons$z)]]) # Hyperbolic tangent activation.
    list(h = hypothesis, d = hypothesis - labels)
}

backpropagation <- function(model, last.deltas) {
    n.layers <- length(model$weights) + 1
    model$deltas[[n.layers]] <- last.deltas
    if (n.layers == 2) return(model)
    for (i in (n.layers - 1):2) {
        model$deltas[[i]] <- tcrossprod(model$deltas[[i + 1]], as.matrix(model$weights[[i]])) * model$gradient(model$neurons$z[[i]])
    }
    model
}

update.model <- function(model, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    model$iteration <- model$iteration + 1
    for (i in 1:length(model$weights)) {
        # Compute gradients.
        model$weights.grad[[i]] <- crossprod(model$neurons$a[[i]], model$deltas[[i + 1]]) + model$lambda * model$weights[[i]] # Add L2 regularization term.
        model$biases.grad[[i]]  <- colSums(model$deltas[[i + 1]])

        # Adam update.
        model$m.weights[[i]] <- beta1 * model$m.weights[[i]] + (1 - beta1) * model$weights.grad[[i]]
        model$m.biases[[i]]  <- beta1 * model$m.biases[[i]]  + (1 - beta1) * model$biases.grad[[i]]
        model$v.weights[[i]] <- beta2 * model$v.weights[[i]] + (1 - beta2) * model$weights.grad[[i]]^2
        model$v.biases[[i]]  <- beta2 * model$v.biases[[i]]  + (1 - beta2) * model$biases.grad[[i]]^2

        alpha.t <- alpha * sqrt(1 - beta2^model$iteration) / (1 - beta1^model$iteration)
        model$weights[[i]] <- model$weights[[i]] - alpha.t * model$m.weights[[i]] / (sqrt(model$v.weights[[i]]) + epsilon)
        model$biases[[i]]  <- model$biases[[i]]  - alpha.t * model$m.biases[[i]]  / (sqrt(model$v.biases[[i]])  + epsilon)
    }
    model
}