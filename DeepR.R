##########################################
## DeepR: Deep Learning Framework for R ##
##########################################

# TODO: apply L2 regularization.
# TODO: implement annealing schedule.
# DONE (TEST): maybe try zero initialization on the bias weights.
# TODO: do not regularize bias terms.
# DONE (TEST): Maybe remove bias neurons altogether and keep a separate, unregularized, biases matrix.
# TODO: apply dropout (hidden units are set to 0 with probability p). Already done by ReLU?. At test time, out weights are multiplied by p.

layers <- c(4, 8, 3)
input <- scale(iris[-5])
labels <- as.matrix(iris[5] == 'setosa')
labels <- cbind(labels, iris[5] == 'virginica')
labels <- cbind(labels, iris[5] == 'versicolor')
model <- train(layers, input, labels, n.iter = 1e3, alpha = 1e-2)

init.model <- function(layers, seed = 0) {
    set.seed(seed)
    model <- list(weights = list(), biases = list())
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i]# + 1 # +1 connection from the bias unit.
        ncol <- layers[i + 1]
        model$weights[[i]] <- matrix(rnorm(nrow * ncol) * sqrt(2 / nrow),
                                     nrow = nrow, ncol = ncol)
        model$biases[[i]] <- matrix(0, nrow = nrow)
    }
    model$biases[[length(layers)]] <- matrix(0, nrow = layers[length(layers)])
    model
}

train <- function(layers, input, labels, n.iter = 1e3, alpha = 1e0, seed = 0, neuron.type = 'ReLU', diagnostics = FALSE) {
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
   
   if (diagnostics) {
      curve <- rep(NA, n.iter)
      accuracy <- rep(NA, n.iter)
      for (i in 1:n.iter) {
         neurons <- forward.propagation(input, model)
         last.deltas <- last.layer.diagnostics(neurons, labels) # Hypothesis and Deltas.
         curve[i] <- last.deltas$l
         accuracy[i] <- mean(apply(round(last.deltas$h) == labels, 1, all))
         deltas <- backpropagation(neurons, model, last.deltas$d)
         model <- update.weights(neurons, model, deltas, alpha)
      }
      list(model = model, c = curve, a = accuracy)
   } else {
      for (i in 1:n.iter) {
         neurons <- forward.propagation(input, model)
         last.deltas <- last.layer(neurons, labels) # Hypothesis and Deltas.
         deltas <- backpropagation(neurons, model, last.deltas$d)
         model <- update.weights(neurons, model, deltas, alpha)
      }
      model
   }
}

test <- function(input, model, labels) {
   neurons <- forward.propagation(input, model)
   last.layer(neurons, labels)$h
}

forward.propagation <- function(input, model) {
   # TODO: maybe avoid creating a list at every iteration.
   neurons <- list(a = list(input))
   n.weights <- length(model$weights)
   if (n.weights > 1) {
      for (i in 1:(n.weights - 1)) {
         # neurons$z[[i + 1]] <- cbind(1, neurons$a[[i]]) %*% weights[[i]] # Add bias unit.
         neurons$z[[i + 1]] <- neurons$a[[i]] %*% model$weights[[i]] + model$biases
         neurons$a[[i + 1]] <- activation(neurons$z[[i + 1]])
      }
   }
   # neurons$z[[n.weights + 1]] <- cbind(1, neurons$a[[n.weights]]) %*% weights[[n.weights]] # Add bias unit.
   neurons$z[[n.weights + 1]] <- neurons$a[[i]] %*% model$weights[[i]] + model$biases
   neurons
}

activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.

gradient <- function(z) z > 0 # Faster than ifelse(z > 0, 1, 0).

last.layer <- function(neurons, labels) {
   # Compute both last layer activations AND loss. It must return its deltas.
   # hypothesis <- neurons$z[[length(neurons$z)]] # Linear activation.
   hypothesis <- 1 / (1 + exp(-neurons$z[[length(neurons$z)]])) # Sigmoid activation.
   list(h = hypothesis, d = hypothesis - labels)
}

last.layer.diagnostics <- function(neurons, labels) {
   # Compute both last layer activations AND loss. It must return its deltas.
   # hypothesis <- neurons$z[[length(neurons$z)]] # Linear activation.
   # loss <- mean((hypothesis - labels)^2) / 2 # Regression loss.
   hypothesis <- 1 / (1 + exp(-neurons$z[[length(neurons$z)]])) # Sigmoid activation.
   loss <- mean(-labels * log(hypothesis) - (1 - labels) * log(1 - hypothesis)) # Logistic loss.
   list(h = hypothesis, d = hypothesis - labels, l = loss)
}

backpropagation <- function(neurons, model, last.deltas) {
    # n.layers <- length(model$weights) + 1
    n.layers <- length(model$weights)
    deltas <- list()
    deltas[[n.layers]] <- last.deltas
    if (n.layers == 2) return(deltas)
    for (i in (n.layers - 1):2) {
        # deltas[[i]] <- deltas[[i + 1]] %*% t(as.matrix(weights[[i]][-1, ])) * gradient(neurons$z[[i]])
       deltas[[i]] <- deltas[[i + 1]] %*% t(as.matrix(model$weights[[i]])) * gradient(neurons$z[[i]])
    }
    deltas
}

update.weights <- function(neurons, weights, deltas, alpha = 1) {
    for (i in 1:length(weights)) {
        # Update weights with a MINUS as this is a MINIMIZATION problem. Add bias unit.
        # weights[[i]] <- weights[[i]] - alpha * t(cbind(1, neurons$a[[i]])) %*% deltas[[i + 1]]
        update <- alpha * t(neurons$a[[i]]) %*% deltas[[i + 1]]
        model$weights[[i]] <- model$weights[[i]] - update
        model$biases[[i]] <- model$biases[[i]] - update
    }
    # weights
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