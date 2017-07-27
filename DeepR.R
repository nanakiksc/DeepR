# TODO: Customize last.layer(): define it as part of the model (and therefore test()).
# TODO: Add Nesterov momentum to Adam (Nadam).
# TODO: Maybe substitute plain dropout by multiplicative Gaussian noise.

init.model <- function(layers, seed = NULL, neuron.type = 'ReLU', scale.method = 'He', task.type = 'sigmoid.classification', recurrent = FALSE, dropout = 0.5, dropout.input = 0.8, lambda = 0) {
    if (!is.null(seed)) set.seed(seed)

    model <- list(weights = list(), biases = list())
    model <- choose.neuron(model, neuron.type)
    model <- choose.task(model, task.type)
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]
        ncol <- layers[i + 1]
        model$weights[[i]] <- matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
        model$weights[[i]] <- scale.weights(model$weights[[i]], scale.method)
        if (recurrent && i > 1) {
            model$weights$recurrent[[i]] <- matrix(rnorm(ncol * ncol), nrow = ncol, ncol = ncol)
            model$weights$recurrent[[i]] <- scale.weights(model$weights$recurrent[[i]], scale.method)
        }
        model$biases[[i]]  <- rep(0, ncol)
        model$m.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$m.biases[[i]]  <- rep(0, ncol)
        model$v.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$v.biases[[i]]  <- rep(0, ncol)
        if (recurrent && i > 1) model$neurons$a[[i]] <- rep(0, ncol) # Siraj initializes the hidden layer activations to 0.
    }
    model$is.recurrent <- recurrent
    model$dropout <- dropout
    model$dropout.input <- dropout.input
    model$lambda <- lambda
    model$iteration <- 0

    model
}

train <- function(model, input, labels, n.iter = 1e3, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, batch.size = nrow(input)) {
    input <- scale(as.matrix(input))
    attr(model, 'scaled:center') <- attr(input, 'scaled:center')
    attr(model, 'scaled:scale') <- attr(input, 'scaled:scale')
    labels <- as.matrix(labels)
    batch.size <- min(batch.size, nrow(input))

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
    input <- scale(as.matrix(input), center = attr(model, 'scaled:center'), scale = attr(model, 'scaled:scale'))
    labels <- as.matrix(labels)

    model <- forward.propagation(input, model, dropout = FALSE)
    hypothesis <- last.layer(model, labels)$h
    # loss <- mean((hypothesis - labels)^2) / 2 # MSE (regression) loss.
    loss <- mean(Vectorize(function(l, h) if (l) -log(h) else -log(1 - h))(labels, hypothesis)) # Cross-entropy (logistic) loss.
    # loss <- mean(Vectorize(function(l, h) if (l == 1) -log((h + 1) / 2) else -log(1 - (h + 1) / 2))(labels, hypothesis)) # Cross-entropy (tanh) loss.

    loss <- loss + model$lambda * sum(unlist(model$weights)^2) / (2 * nrow(model$neurons$z[[length(model$neurons$z)]])) # Add L2 regularization term.

    accuracy <- mean(apply(round(hypothesis) == labels, 1, all))
    list(hypothesis = hypothesis, loss = loss, accuracy = accuracy)
}

choose.neuron <- function(model, neuron.type) {
    if (compare.words('ReLU', neuron.type)) {
        model$activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.
        model$gradient   <- function(z) z > 0 # Faster than ifelse(z > 0, 1, 0).
    } else if (compare.words('sigmoid', neuron.type)) {
        model$activation <- function(z) 1 / (1 + exp(-z))
        model$gradient   <- function(z) { s <- activation(z); s * (1 - s) }
    } else if (compare.words('tanh', neuron.type)) {
        model$activation <- function(z) tanh(z)
        model$gradient   <- function(z) 1 - tanh(z)^2
    } else {
        print('Unknown activation function. Please choose ReLU, sigmoid or tanh.')
        stop()
    }

    model
}

choose.task <- function(model, task.type) {
    if (compare.words('regression', task.type)) {
        model$hypothesis <- function(model) model$neurons$z[[length(model$neurons$z)]] # Linear activation.
        model$loss       <- function(labels, hypothesis) mean((hypothesis - labels)^2) / 2 # MSE (regression) loss.
    } else if (compare.words('sigmoid.classification', task.type)) {
        model$hypothesis <- function(model) 1 / (1 + exp(-model$neurons$z[[length(model$neurons$z)]])) # Sigmoid activation.
        model$loss       <- function(labels, hypothesis) mean(Vectorize(function(l, h) if (l) -log(h) else -log(1 - h))(labels, hypothesis)) # Cross-entropy (logistic) loss.
    } else if (compare.words('tanh.classification', task.type)) {
        model$hypothesis <- function(model) tanh(model$neurons$z[[length(model$neurons$z)]]) # Hyperbolic tangent activation.
        model$loss       <- function(labels, hypothesis) mean(Vectorize(function(l, h) if (l == 1) -log((h + 1) / 2) else -log(1 - (h + 1) / 2))(labels, hypothesis)) # Cross-entropy (tanh) loss.
    } else {
        print('Unknown loss function. Please choose regression, sigmoid.classification, or tanh.classification.')
        stop()
    }

    model
}

compare.words <- function(x, y) { w <- tolower(c(x, y)); substr(w[1], 1, nchar(w[2])) == w[2] } # Compare the beginning.

scale.weights <- function (weights, scale.method = 'He') {
    if      (compare.words('He',     scale.method)) weights * sqrt(2 / nrow(weights)) # He et al. adjusted initialization for deep ReLU networks.
    else if (compare.words('Xavier', scale.method)) weights * sqrt(2 / sum(dim(weights))) # Xavier initializarion for deep networks.
    else if (compare.words('Caffe',  scale.method)) weights * sqrt(nrow(weights)) # Caffe version of Xavier initialization.
    else if (compare.words('none',   scale.method)) weights # Like... no scaling. Why would you do that, right?
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
            if (model$is.recurrent) model$neurons$z[[i]] <- model$neurons$z[[i]] # + model$weights[[h-h]]
            model$neurons$a[[i]] <- model$activation(model$neurons$z[[i]])
            if (dropout) model$neurons$a[[i]] <- dropout.mask(model, i)
        }
    }
    # Last layer always contains all neurons.
    weights <- model$weights[[n.weights]]
    if (!dropout) weights <- weights * model$dropout
    model$neurons$z[[n.weights + 1]] <- sweep(model$neurons$a[[n.weights]] %*% weights, 2, model$biases[[n.weights ]], '+')
    model$neurons$a[[n.weights + 1]] <- model$activation(model$neurons$z[[n.weights + 1]])

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