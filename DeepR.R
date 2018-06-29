# TODO: Try AdaMax instead of Adam. Removes epsilon and the need for bias correction in the beta2 term (and the max operation looks similar to AMSGrad).
# TODO: Study how AMSGrad could go with Nesterov momentum.
# TODO: Maybe substitute plain dropout by multiplicative Gaussian noise.
# TODO: Test dropout-corrected weight initialization (scaling by the numer of "effective" neurons).
# TODO: Make Nelder-Mead wrapper for hyperparameter tuning. Default Nelder-Mead parameters should work.

init.model <- function(layers, seed = NULL, neuron.type = 'ReLU', scale.method = 'He', task.type = 'sigmoid.classification', dropout = 0.5, dropout.input = 0.8, dropout.scaling = TRUE, lambda = 0) {
    if (!is.null(seed)) set.seed(seed)
    if ( compare.words('ReLU', neuron.type) & !compare.words('He', scale.method))     print('Maybe you should consider He initialization when using ReLU neurons.')
    if (!compare.words('ReLU', neuron.type) & !compare.words('Xavier', scale.method)) print('Maybe you should consider Xavier initialization when using non-ReLU neurons.')

    model <- list(weights = list(), biases = list())
    model <- choose.neuron(model, neuron.type)
    model <- choose.task(model, task.type)
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]
        ncol <- layers[i + 1]
        dropout.factor <- if (dropout.scaling) { if (i == 1) c(dropout.input, dropout) else dropout } else 1
        model$weights[[i]] <- matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
        model$weights[[i]] <- scale.weights(model$weights[[i]], scale.method, dropout.factor)
        model$biases[[i]]  <- rep(0, ncol)
        model$m.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$m.biases[[i]]  <- rep(0, ncol)
        model$v.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$v.biases[[i]]  <- rep(0, ncol)
    }
    model$dropout <- dropout
    model$dropout.input <- dropout.input
    model$lambda <- lambda
    model$iteration <- 0

    model
}

train <- function(model, input, labels, n.iter = 1e3, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, batch.size = nrow(as.matrix(input))) {
    input <- scale(as.matrix(input))
    attr(model, 'scaled:center') <- attr(input, 'scaled:center')
    attr(model, 'scaled:scale') <- attr(input, 'scaled:scale')
    labels <- as.matrix(labels)
    batch.size <- min(batch.size, nrow(input))

    for (i in 1:n.iter) {
        idx <- sample(nrow(input), batch.size)
        model <- forward.propagation(model, input[idx, , drop = FALSE])
        model <- backpropagation(model, labels[idx, , drop = FALSE])
        model <- update.model(model, alpha, beta1, beta2, epsilon)
    }
    model
}

test <- function(model, input, labels, use.lambda = TRUE) {
    input <- scale(as.matrix(input), center = attr(model, 'scaled:center'), scale = attr(model, 'scaled:scale'))
    labels <- as.matrix(labels)

    model <- forward.propagation(model, input, use.dropout = FALSE)
    hypothesis <- model$neurons$a[[length(model$neurons$a)]] # Already computed in forward propagation.
    loss <- model$loss(hypothesis, labels)
    if (use.lambda) { # Optionally omit L2 regularization for plotting loss.
       loss <- loss + model$lambda * sum(unlist(model$weights)^2) / (2 * nrow(model$neurons$z[[length(model$neurons$z)]])) # Add L2 regularization term.
    }

    accuracy <- NA
    if (model$task.type != 'regression') accuracy <- mean(apply(round(hypothesis) == labels, 1, all))
    list(hypothesis = hypothesis, loss = loss, accuracy = accuracy)
}

predict <- function(model, input) {
    input <- scale(as.matrix(input), center = attr(model, 'scaled:center'), scale = attr(model, 'scaled:scale'))

    model <- forward.propagation(model, input, use.dropout = FALSE)
    model$neurons$a[[length(model$neurons$a)]] # Already computed in forward propagation.
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
    } else if (compare.words('softsign', neuron.type)) {
        model$activation <- function(z) z / (1 + abs(z))
        model$gradient   <- function(z) 1 / (1 + abs(z))^2
    } else {
        print('Unknown activation function. Please choose ReLU, sigmoid, tanh or softsign.')
        stop()
    }

    model
}

choose.task <- function(model, task.type) {
    # The hypothesis function is the last layer activation,
    # while the loss function is only used when testing the predictions (not for computing the error during backpropagation).
    # Since both cross-entropy and quadratic loss functions have the same derivative, it is hardcoded in backpropagation().
    # TODO: Allow for different (composite) last-layer activations.
    if (compare.words('regression', task.type)) {
        model$task.type  <- 'regression'
        model$hypothesis <- function(model) model$neurons$z[[length(model$neurons$z)]] # Linear activation.
        model$loss       <- function(h, l) mean((h - l)^2) / 2 # MSE (regression) loss.
    } else if (compare.words('softmax', task.type)) {
        model$task.type  <- 'softmax'
        model$hypothesis <- function(model) { e <- exp(model$neurons$z[[length(model$neurons$z)]]); e / sum(e) } # Softmax activation.
        model$loss       <- function(h, l) mean(Vectorize(function(h, l) if (l) -log(h) else -log(1 - h))(h, l)) # Cross-entropy loss. TODO: test.
    } else if (compare.words('sigmoid.classification', task.type)) {
        model$task.type  <- 'sigmoid.classification'
        model$hypothesis <- function(model) 1 / (1 + exp(-model$neurons$z[[length(model$neurons$z)]])) # Sigmoid activation.
        model$loss       <- function(h, l) mean(Vectorize(function(h, l) if (l) -log(h) else -log(1 - h))(h, l)) # Cross-entropy loss.
    } else if (compare.words('tanh.classification', task.type)) {
        model$task.type  <- 'tanh.classification'
        model$hypothesis <- function(model) tanh(model$neurons$z[[length(model$neurons$z)]]) # Hyperbolic tangent activation.
        model$loss       <- function(h, l) mean(Vectorize(function(h, l) if (l == 1) -log((h + 1) / 2) else -log(1 - (h + 1) / 2))(h, l)) # Cross-entropy (tanh) loss (just remap tanh domain to logistic domain before calculating the cross-entropy).
    } else if (compare.words('none', task.type)) {
        print('Remember to specify a hypothesis (activation) function for the last layer and a loss function.')
    } else {
        print('Unknown loss function. Please choose regression, softmax, sigmoid.classification, tanh.classification or none.')
        stop()
    }

    model
}

compare.words <- function(x, y) { w <- tolower(c(x, y)); substr(w[1], 1, nchar(w[2])) == w[2] } # Compare the beginning.

scale.weights <- function (weights, scale.method = 'He', dropout.factor = 1) { # TODO: test dropout factor corrections.
    # Note that this version of Xavier initialization adjustes the variance of a normal distribution, it doesn't sample from a uniform.
    if      (compare.words('He',     scale.method)) weights * sqrt(2 / nrow(weights) / dropout.factor[1]) # He et al. adjusted initialization for deep ReLU networks.
    else if (compare.words('Xavier', scale.method)) weights * sqrt(2 / sum(dim(weights) * dropout.factor)) # Xavier initializarion for deep networks.
    else if (compare.words('Caffe',  scale.method)) weights * sqrt(nrow(weights) * dropout.factor[1]) # Caffe version of Xavier initialization.
    else if (compare.words('none',   scale.method)) weights # Like... no scaling. Why would you do that, right?
    else { print('Unknown weight initialization method. Please choose He, Xavier, Caffe or none.'); stop() }
}

forward.propagation <- function(model, input, use.dropout = TRUE) {
    model$neurons$a[[1]] <- input
    if (use.dropout) model$neurons$a[[1]] <- dropout.mask(model, 1)
    else model$neurons$a[[1]] <- model$neurons$a[[1]] * model$dropout.input
    n.weights <- length(model$weights)
    if (n.weights > 1) {
        for (i in 2:n.weights) {
            model$neurons$z[[i]] <- sweep(model$neurons$a[[i - 1]] %*% model$weights[[i - 1]], 2, model$biases[[i - 1]], '+')
            model$neurons$a[[i]] <- model$activation(model$neurons$z[[i]])
            if (use.dropout) model$neurons$a[[i]] <- dropout.mask(model, i)
            else model$neurons$a[[i]] <- model$neurons$a[[i]] * model$dropout
        }
    }
    # Last layer always contains all neurons (no dropout).
    model$neurons$z[[n.weights + 1]] <- sweep(model$neurons$a[[n.weights]] %*% model$weights[[n.weights]], 2, model$biases[[n.weights ]], '+')
    model$neurons$a[[n.weights + 1]] <- model$hypothesis(model)

    model
}

dropout.mask <- function(model, layer) {
    nrow <- nrow(model$neurons$a[[layer]])
    ncol <- ncol(model$neurons$a[[layer]])
    dropout <- if (layer == 1) model$dropout.input else model$dropout
    mask <- matrix(sample(0:1, nrow * ncol, replace = TRUE, prob = c(1 - dropout, dropout)), nrow = nrow, ncol = ncol)
    model$neurons$a[[layer]] * mask
}

backpropagation <- function(model, labels) {
    n.layers <- length(model$weights) + 1
    model$deltas[[n.layers]] <- model$neurons$a[[n.layers]] - labels # Compute last deltas from last activations (hypothesis).
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
        model$weights.grad[[i]] <- crossprod(model$neurons$a[[i]], model$deltas[[i + 1]])
        if (model$lambda) model$weights.grad[[i]] <- model$weights.grad[[i]] + model$lambda * model$weights[[i]] # Add L2 regularization term.
        model$biases.grad[[i]] <- colSums(model$deltas[[i + 1]])

        # AMSGrad update.
        beta1.t <- beta1# * (1 - 1e-8)^(model$iteration - 1) # First moment running average coefficient decay.
        model$m.weights[[i]] <- beta1.t * model$m.weights[[i]] + (1 - beta1.t) * model$weights.grad[[i]]
        model$m.biases[[i]]  <- beta1.t * model$m.biases[[i]]  + (1 - beta1.t) * model$biases.grad[[i]]
        model$v.weights[[i]] <- pmax(model$v.weights[[i]], beta2 * model$v.weights[[i]] + (1 - beta2) * model$weights.grad[[i]]^2)
        model$v.biases[[i]]  <- pmax(model$v.biases[[i]],  beta2 * model$v.biases[[i]]  + (1 - beta2) * model$biases.grad[[i]]^2)

        alpha.t <- alpha #/ sqrt(model$iteration) # Stepsize annealing schedule. TODO: implement Vaswani et al. (2017) schedule.
        bc <- sqrt(1 - beta2^model$iteration) / (1 - beta1^model$iteration) # Initialization bias correction factor.
        model$weights[[i]] <- model$weights[[i]] - alpha.t * bc * model$m.weights[[i]] / (sqrt(model$v.weights[[i]]) + epsilon)
        model$biases[[i]]  <- model$biases[[i]]  - alpha.t * bc * model$m.biases[[i]]  / (sqrt(model$v.biases[[i]])  + epsilon)
    }
    model
}
