# TODO: try dropout (hidden units are set to 0 with probability p, at test time, out weights are multiplied by p).
# TODO: implement Adam. Add Nesterov momentum (Nadam)?
# TODO: customize last.layer() (and maybe test()).

init.model <- function(layers, seed = NULL, neuron.type = 'ReLU', scale.method = 'He') {
    if (!is.null(seed)) set.seed(seed)

    model <- list(weights = list(), biases = list())
    model <- choose.neuron(model, neuron.type)
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]
        ncol <- layers[i + 1]
        model$weights[[i]] <- matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
        model$weights[[i]] <- scale.weights(model$weights[[i]], scale.method)
        model$biases[[i]] <- rep(0, ncol)
        model$m.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$m.biases[[i]] <- rep(0, ncol)
        model$v.weights[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$v.biases[[i]] <- rep(0, ncol)
    }
    model$iteration <- 0

    model
}

train <- function(model, input, labels, n.iter = 1e3, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lambda = 0) {
    input <- as.matrix(input)

    for (i in 1:n.iter) {
         model <- forward.propagation(input, model)
         last.deltas <- last.layer(model, labels) # Hypothesis and Deltas.
         model <- backpropagation(model, last.deltas$d)
         model <- update.model(model, alpha, mu, lambda)
      }
      model
}

test <- function(model, input, labels, lambda = 0) {
   model <- forward.propagation(input, model)
   hypothesis <- last.layer(model, labels)$h
   # loss <- mean((hypothesis - labels)^2) / 2 # MSE (regression) loss.
   loss <- mean(-labels * log(hypothesis) - (1 - labels) * log(1 - hypothesis)) # Cross-entropy (logistic) loss.
   loss <- loss + lambda * sum(unlist(model$weights)^2) / (2 * nrow(model$neurons$z[[length(model$neurons$z)]])) # Add L2 regularization term.
   accuracy <- mean(apply(round(hypothesis) == labels, 1, all))
   list(hypothesis = hypothesis, loss = loss, accuracy = accuracy)
}

choose.neuron <- function(model, neuron.type) {
    if (neuron.type == 'ReLU') {
        activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.
        gradient <- function(z) z > 0 # Faster than ifelse(z > 0, 1, 0).
    } else if (neuron.type == 'sigmoid') {
        activation <- function(z) 1 / (1 + exp(-z))
        gradient <- function(z) { s <- activation(z); s * (1 - s) }
    } else if (neuron.type == 'tanh') {
        activation <- function(z) tanh(z)
        gradient <- function(z) 1 - tanh(z)^2
    } else {
        print('Unknown activation function. Please choose ReLU, sigmoid or tanh.')
        stop()
    }

    model$activation <- activation
    model$gradient <- gradient
    model
}

scale.weights <- function (weights, scale.method = 'He') {
    if      (scale.method == 'He')     weights * sqrt(2 / nrow(weights)) # He et al. adjusted initialization for deep ReLU networks.
    else if (scale.method == 'Xavier') weights * sqrt(2 / sum(dim(weights))) # Xavier initializarion for deep networks.
    else if (scale.method == 'Caffe')  weights * sqrt(nrow(weights)) # Caffe version of Xavier initialization.
    else if (scale.method == 'none')   weights # Like... no scaling. Why would you do that, right?
    else { print('Unknown wheight initialization method. Please choose He, Xavier, Caffe or none.'); stop() }
}

forward.propagation <- function(input, model) {
    model$neurons$a[[1]] <- input
    n.weights <- length(model$weights)
    if (n.weights > 1) {
        for (i in 2:n.weights) {
            model$neurons$z[[i]] <- sweep(model$neurons$a[[i - 1]] %*% model$weights[[i - 1]], 2, model$biases[[i - 1]], '+')
            model$neurons$a[[i]] <- activation(model$neurons$z[[i]])
        }
    }
    model$neurons$z[[n.weights + 1]] <- sweep(model$neurons$a[[n.weights]] %*% model$weights[[n.weights]], 2, model$biases[[n.weights ]], '+')
    model
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
        model$deltas[[i]] <- tcrossprod(model$deltas[[i + 1]], as.matrix(model$weights[[i]])) * gradient(model$neurons$z[[i]])
    }
    model
}

update.model <- function(model, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, mu = 0, lambda = 0) {
    iteration <- iteration + 1
    for (i in 1:length(model$weights)) {
        # Adam update. Add L2 regularization term to weights.
        model$m.weights[[i]] <- beta1 * model$m.weights[[i]] + (1 - beta1) * (crossprod(model$neurons$a[[i]], model$deltas[[i + 1]]) + lambda * model$weights[[i]])
        model$m.biases[[i]] <- beta1 * model$m.biases[[i]] + (1 - beta1) * colSums(model$deltas[[i + 1]])
        model$v.weights[[i]] <- beta2 * model$v.weights[[i]] + (1 - beta2) * (crossprod(model$neurons$a[[i]], model$deltas[[i + 1]]) + lambda * model$weights[[i]])^2
        model$v.biases[[i]] <- beta2 * model$v.biases[[i]] + (1 - beta2) * colSums(model$deltas[[i + 1]])^2

        model$weights[[i]] <- model$weights[[i]] - alpha * model$m.weights[[i]] / (1 - beta1^iteration) / (sqrt(model$v.weights[[i]] / (1 - beta2^iteration)) + epsilon)
        model$biases[[i]] <- model$biases[[i]] - alpha * model$m.biases[[i]] / (1 - beta1^iteration) / (sqrt(model$v.biases[[i]] / (1 - beta2^iteration)) + epsilon)
    }
    model
}