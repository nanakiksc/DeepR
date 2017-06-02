# TODO: try dropout (hidden units are set to 0 with probability p, at test time, out weights are multiplied by p).
# TODO: implement Adam. Add Nesterov momentum (Nadam)?
# TODO: customize last layer (and maybe test function).

init.model <- function(layers, seed = NULL, method = 'He') {
    if (!is.null(seed)) set.seed(seed)

    scale.weights <- if (method == 'He')     function(weights) weights * sqrt(2 / nrow(weights)) # He et al. adjusted initialization for deep ReLU networks.
                else if (method == 'Xavier') function(weights) weights * sqrt(2 / sum(dim(weights))) # Xavier initializarion for deep networks.
                else if (method == 'Caffe')  function(weights) weights * sqrt(nrow(weights)) # Caffe version of Xavier initialization.
                else if (method == 'none')   function(weights) weights # Like... no scaling. Why would you do that, right?
                else { print('Unknown wheight initialization method. Please choose He, Xavier, Caffe or none.'); stop() }

    model <- list(weights = list(), biases = list())
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]
        ncol <- layers[i + 1]
        model$weights[[i]] <- scale.weights(matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol))
        model$biases[[i]] <- rep(0, ncol)
        model$w.velocities[[i]] <- matrix(0, nrow = nrow, ncol = ncol)
        model$b.velocities[[i]] <- rep(0, ncol)
        # TODO: deltas and neurons NOT vectors if operations are vectorized (they are matrices).
        # TODO: these guys aren't even necessary, as they will be init at the firs iteration.
        # model$deltas[[i + 1]] <- rep(NA, layers[i + 1]) # deltas[[1]] is NULL.
        # model$neurons$a[[i + 1]] <- rep(NA, layers[i + 1]) # neurons$a[[1]] are the inputs.
        # model$neurons$z[[i + 1]] <- rep(NA, layers[i + 1]) # neurons$z[[1]] is NULL.
    }

    model
}

train <- function(model, input, labels, n.iter = 1e3, alpha = 1, mu = 0, lambda = 0, neuron.type = 'ReLU') {
    input <- as.matrix(input)

    if (neuron.type == 'ReLU') {
        activation <<- function(z) (abs(z) + z) / 2
        gradient <<- function(z) z > 0
    } else if (neuron.type == 'sigmoid') {
        activation <<- function(z) 1 / (1 + exp(-z))
        gradient <<- function(z) { s <- activation(z); s * (1 - s) }
    } else if (neuron.type == 'tanh') {
        activation <<- function(z) tanh(z)
        gradient <<- function(z) 1 - tanh(z)^2
    } else {
        print('Unknown activation function. Please choose ReLU, sigmoid or tanh.')
        stop()
    }

    for (i in 1:n.iter) {
         # neurons <- forward.propagation(input, model)
         model <- forward.propagation(input, model)
         # last.deltas <- last.layer(neurons, labels) # Hypothesis and Deltas.
         last.deltas <- last.layer(model, labels) # Hypothesis and Deltas.
         # deltas <- backpropagation(neurons, model, last.deltas$d)
         model <- backpropagation(model, last.deltas$d)
         # model <- update.model(neurons, model, deltas, alpha, mu, lambda)
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

forward.propagation <- function(input, model) {
   # TODO: maybe avoid creating a list at every iteration.
   # neurons <- list(a = list(input))
   model$neurons$a[[1]] <- input
   n.weights <- length(model$weights)
   if (n.weights > 1) {
      for (i in 2:n.weights) {
         # neurons$z[[i]] <- sweep(neurons$a[[i - 1]] %*% model$weights[[i - 1]], 2, model$biases[[i - 1]], '+')
         # neurons$a[[i]] <- activation(neurons$z[[i]])
         model$neurons$z[[i]] <- sweep(model$neurons$a[[i - 1]] %*% model$weights[[i - 1]], 2, model$biases[[i - 1]], '+')
         model$neurons$a[[i]] <- activation(model$neurons$z[[i]])
      }
   }
   model$neurons$z[[n.weights + 1]] <- sweep(model$neurons$a[[n.weights]] %*% model$weights[[n.weights]], 2, model$biases[[n.weights ]], '+')
   # neurons
   model
}

activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.

gradient <- function(z) z > 0 # Faster than ifelse(z > 0, 1, 0).

# last.layer <- function(neurons, labels) {
last.layer <- function(model, labels) {
   # Compute last layer activations (hypothesis). It must return its deltas.
   # hypothesis <- model$neurons$z[[length(model$neurons$z)]] # Linear activation.
   # print(model$neurons$z[[length(model$neurons$z)]])
   hypothesis <- 1 / (1 + exp(-model$neurons$z[[length(model$neurons$z)]])) # Sigmoid activation.
   # hypothesis <- tanh(model$neurons$z[[length(model$neurons$z)]]) # Hyperbolic tangent activation.
   list(h = hypothesis, d = hypothesis - labels)
}

# backpropagation <- function(neurons, model, last.deltas) {
backpropagation <- function(model, last.deltas) {
    n.layers <- length(model$weights) + 1
    # deltas <- list() # TODO: maybe avoid creating a list at every iteration.
    # deltas[[n.layers]] <- last.deltas
    model$deltas[[n.layers]] <- last.deltas
    # if (n.layers == 2) return(deltas)
    if (n.layers == 2) return(model)
    for (i in (n.layers - 1):2) {
        # deltas[[i]] <- tcrossprod(deltas[[i + 1]], as.matrix(model$weights[[i]])) * gradient(neurons$z[[i]])
        model$deltas[[i]] <- tcrossprod(model$deltas[[i + 1]], as.matrix(model$weights[[i]])) * gradient(model$neurons$z[[i]])
    }
    # deltas
    model
}

# update.model <- function(neurons, model, deltas, alpha = 1, mu = 0, lambda = 0) {
update.model <- function(model, alpha = 1, mu = 0, lambda = 0) {
    for (i in 1:length(model$weights)) {
        # Add momentum and L2 regularization term.
        # model$w.velocities[[i]] <- mu * model$w.velocities[[i]] + alpha * (crossprod(neurons$a[[i]], deltas[[i + 1]]) + lambda * model$weights[[i]])
        # model$b.velocities[[i]] <- mu * model$b.velocities[[i]] + alpha * colSums(deltas[[i + 1]])
        # model$weights[[i]] <- model$weights[[i]] - model$w.velocities[[i]]
        # model$biases[[i]] <- model$biases[[i]] - model$b.velocities[[i]]

        model$w.velocities[[i]] <- mu * model$w.velocities[[i]] + alpha * (crossprod(model$neurons$a[[i]], model$deltas[[i + 1]]) + lambda * model$weights[[i]])
        model$b.velocities[[i]] <- mu * model$b.velocities[[i]] + alpha * colSums(model$deltas[[i + 1]])
        model$weights[[i]] <- model$weights[[i]] - model$w.velocities[[i]]
        model$biases[[i]] <- model$biases[[i]] - model$b.velocities[[i]]
    }
    model
}
