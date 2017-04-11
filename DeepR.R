##########################################
## DeepR: Deep Learning Framework for R ##
##########################################

# TODO: apply L2 regularization, do not regularize bias terms.
# TODO: implement annealing schedule.
# TODO: try dropout (hidden units are set to 0 with probability p). Already done by ReLU?. At test time, out weights are multiplied by p.
# TODO: add momentum?

init.model <- function(layers, seed = NULL) {
    if (!is.null(seed)) set.seed(seed)
    model <- list(weights = list(), biases = list())
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]
        ncol <- layers[i + 1]
        model$weights[[i]] <- matrix(rnorm(nrow * ncol), nrow = nrow, ncol = ncol)
        # model$weights[[i]] <- model$weights[[i]] * sqrt(2 / (nrow + ncol)), # Xavier initializarion for deep networks.
        # model$weights[[i]] <- model$weights[[i]] * sqrt(nrow), # Caffe version of Xavier initialization.
        model$weights[[i]] <- model$weights[[i]] * sqrt(2 / nrow) # He et al. adjusted initialization for deep ReLU networks.
        model$biases[[i]] <- rep(0, ncol)
    }
    model
}

train <- function(layers, input, labels, n.iter = 1e3, alpha = 1, lambda = 0, seed = NULL, neuron.type = 'ReLU', diagnostics = FALSE) {
   model <- init.model(layers, seed)
   
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
   
   # if (diagnostics) { # TODO: remove this. Implement a sampling schedule (train() without diagnostics + test() every now and then).
   #    curve <- rep(NA, n.iter)
   #    accuracy <- rep(NA, n.iter)
   #    for (i in 1:n.iter) {
   #       neurons <- forward.propagation(input, model)
   #       last.deltas <- last.layer.diagnostics(neurons, labels, model, lambda) # Hypothesis and Deltas.
   #       curve[i] <- last.deltas$l
   #       accuracy[i] <- mean(apply(round(last.deltas$h) == labels, 1, all))
   #       deltas <- backpropagation(neurons, model, last.deltas$d)
   #       model <- update.model(neurons, model, deltas, alpha, lambda)
   #    }
   #    list(model = model, c = curve, a = accuracy)
   # } else {
      for (i in 1:n.iter) {
         neurons <- forward.propagation(input, model)
         last.deltas <- last.layer(neurons, labels) # Hypothesis and Deltas.
         deltas <- backpropagation(neurons, model, last.deltas$d)
         model <- update.model(neurons, model, deltas, alpha, lambda)
      }
      model
   # }
}

test <- function(input, model, labels, lambda = 0) {
   neurons <- forward.propagation(input, model)
   hypothesis <- last.layer(neurons, labels)$h
   # loss <- mean((hypothesis - labels)^2) / 2 # MSE (regression) loss.
   loss <- mean(-labels * log(hypothesis) - (1 - labels) * log(1 - hypothesis)) # Cross-entropy (logistic) loss.
   loss <- loss + lambda * sum(unlist(model$weights)^2) / (2 * nrow(neurons$z[[length(neurons$z)]])) # Add L2 regularization term.
   list(h = hypothesis, l = loss)
}

forward.propagation <- function(input, model) {
   # TODO: maybe avoid creating a list at every iteration.
   neurons <- list(a = list(input))
   n.weights <- length(model$weights)
   if (n.weights > 1) {
      for (i in 2:n.weights) {
         neurons$z[[i]] <- sweep(neurons$a[[i - 1]] %*% model$weights[[i - 1]], 2, model$biases[[i - 1]], '+')
         neurons$a[[i]] <- activation(neurons$z[[i]])
      }
   }
   neurons$z[[n.weights + 1]] <- sweep(neurons$a[[n.weights]] %*% model$weights[[n.weights]], 2, model$biases[[n.weights ]], '+')
   neurons
}

activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.

gradient <- function(z) z > 0 # Faster than ifelse(z > 0, 1, 0).

last.layer <- function(neurons, labels) {
   # Compute last layer activations (hypothesis). It must return its deltas.
   # hypothesis <- neurons$z[[length(neurons$z)]] # Linear activation.
   hypothesis <- 1 / (1 + exp(-neurons$z[[length(neurons$z)]])) # Sigmoid activation.
   list(h = hypothesis, d = hypothesis - labels)
}

backpropagation <- function(neurons, model, last.deltas) {
    n.layers <- length(model$weights) + 1
    deltas <- list()
    deltas[[n.layers]] <- last.deltas
    if (n.layers == 2) return(deltas)
    for (i in (n.layers - 1):2) {
       deltas[[i]] <- deltas[[i + 1]] %*% t(as.matrix(model$weights[[i]])) * gradient(neurons$z[[i]])
    }
    deltas
}

update.model <- function(neurons, model, deltas, alpha = 1, lambda = 0) {
    for (i in 1:length(model$weights)) {
        # Update weights with a MINUS as this is a MINIMIZATION problem. Add L2 regularization term.
        # model$weights[[i]] <- model$weights[[i]] - alpha * t(neurons$a[[i]]) %*% deltas[[i + 1]]
        model$weights[[i]] <- model$weights[[i]] - alpha * (t(neurons$a[[i]]) %*% deltas[[i + 1]] + lambda * model$weights[[i]])
        model$biases[[i]] <- model$biases[[i]] - alpha * colSums(deltas[[i + 1]])
    }
    model
}

#     idx <- c()
#     # Do not include the first n.bars examples from the training set.
#     for (i in 1:ceiling(n.iter / (nrow(train.set) - n.bars))) {
#         idx <- c(idx, sample(nrow(train.set) - n.bars) + n.bars)
#     }
#     idx <- idx[1:n.iter]
#     for (i in 1:n.iter) input <- matrix(train.set[(idx[i] - n.bars + 1):idx[i], ], nrow = 1)

# play <- function(train.set, idx, neurons, n.bars) {
#     # Use tanh activation for the B/S signal and linear activation for setting the limit.
#     output <- neurons$z[[length(neurons$z)]]
#     reward <- 0
#     if (!round(tanh(output[1]))) return(reward)
#
#     entry <- train.set$Close[idx]
#     #limit <- output[2] # Use the same limit for SL and TP.
#     limit <- sd(train.set$Close[(idx - n.bars + 1):idx]) * 2
#     for (i in (idx + 1):nrow(train.set)) {
#         if (train.set$High[i] > entry + limit) {
#             reward <- limit
#             break
#         }
#         else if (train.set$Low[i] < entry - limit) {
#             reward <- -limit
#             break
#         }
#     }
#     reward * sign(tanh(output[1]))
# }