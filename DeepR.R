##########################################
## DeepR: Deep Learning Framework for R ##
##########################################

# TODO: apply L2 regularization.
# TODO: set initial learning rate alpha and annealing schedule.
# TODO: maybe try zero initialization on the bias weights.
# TODO: do not regularize bias terms.
# TODO: apply dropout (hidden units are set to 0 with probability p). Already done by ReLU?. At test time, out weights are multiplied by p.

init.weights <- function(layers, seed = 0) {
    set.seed(seed)
    weights <- list()
    for (i in 1:(length(layers) - 1)) {
        nrow <- layers[i] + 1 # +1 connection from the bias unit.
        ncol <- layers[i + 1]
        weights[[i]] <- matrix(rnorm(nrow * ncol) * sqrt(2 / nrow),
                               nrow = nrow, ncol = ncol)
    }
    weights
}

# train <- function(train.set, raw.train.set, weights, n.bars, n.iter = 1000, alpha = 1) {
#     set.seed(0)
#     idx <- c()
#     # Do not include the first n.bars examples from the training set.
#     for (i in 1:ceiling(n.iter / (nrow(train.set) - n.bars))) {
#         idx <- c(idx, sample(nrow(train.set) - n.bars) + n.bars)
#     }
#     idx <- idx[1:n.iter]
#     rewards <- rep(NA, n.iter)
#     for (i in 1:n.iter) {
#         input <- matrix(train.set[(idx[i] - n.bars + 1):idx[i], ], nrow = 1)
#         neurons <- forward.propagation(input, weights)
#         # Computes last layer activations AND loss.
#         last.deltas <- last.layer(neurons, labels)
#         deltas <- backpropagation(neurons, weights, last.deltas)
#         # Compute reward from external environment.
#         #rewards[i] <- play(raw.train.set, idx[i], neurons, n.bars)
#         #deltas <- backpropagation(neurons, weights, rewards[i])
#         weights <- update.weights(neurons, weights, deltas, alpha)
#     }
#     list(w = weights, r = rewards)
# }

layers <- c(4, 8, 1)
idx <- sample(nrow(iris), 20)
input <- scale(iris[idx, -5])
labels <- iris[idx, 5] == 'setosa'

train <- function(layers, input, labels, n.iter = 1e3, alpha = 1e0, seed = 0) {
   weights <- init.weights(layers, seed)
   curve <- rep(NA, n.iter)
   accuracy <- rep(NA, n.iter)
   for (i in 1:n.iter) {
      neurons <- forward.propagation(input, weights)
      last.deltas <- last.layer(neurons, labels) # Hypothesis and Deltas.
      curve[i] <- last.deltas$l
      accuracy[i] <- mean(round(neurons$a[[length(neurons$a)]]) == labels)
      deltas <- backpropagation(neurons, weights, last.deltas$d)
      weights <- update.weights(neurons, weights, deltas, alpha)
   }
   list(w = weights, c = curve, a = accuracy)
}

test <- function(input, weights, labels) {
   neurons <- forward.propagation(input, weights)
   last.layer(neurons, labels)$h
}

forward.propagation <- function(input, weights) {
   # TODO: remove unnecessary last layer activation.
   # TODO: maybe avoid creating a list at every iteration.
    neurons <- list(a = list(input))
    for (i in 1:length(weights)) {
        neurons$z[[i + 1]] <- cbind(1, neurons$a[[i]]) %*% weights[[i]] # Add bias unit.
        neurons$a[[i + 1]] <- activation(neurons$z[[i + 1]])
    }
    neurons
}

# activation <- function(z) (abs(z) + z) / 2 # Faster than pmax(0, z) or z[z < 0] <- 0.

# gradient <- function(z) ifelse(z > 0, 1, 0)

activation <- function(z) 1 / (1 + exp(-z))

gradient <- function(z) { s <- activation(z); s * (1 - s) }

last.layer <- function(neurons, labels) {
   # Compute both last layer activations AND loss. It must return its gradient or delta.
   #hypothesis <- activation(neurons$z[[length(neurons$z)]])
   hypothesis <- neurons$a[[length(neurons$a)]]
   loss <- mean(-labels * log(hypothesis) - (1 - labels) * log(1 - hypothesis))
   list(h = hypothesis, l = loss, d = hypothesis - labels)#loss * gradient(neurons$z[[length(neurons$z)]]))
   #loss * gradient(neurons$z[[length(neurons$z)]])
}

# play <- function(train.set, idx, neurons, n.bars) {
#     # Use tanh for the B/S signal and x^2 for setting the limit.
#     output <- neurons$z[[length(neurons$z)]]
#     reward <- 0
#     if (!round(tanh(output[1]))) return(reward)
#
#     entry <- train.set$Close[idx]
#     #limit <- (output[2])^2 # Use the same limit for SL and TP.
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

backpropagation <- function(neurons, weights, last.deltas) {
    n.layers <- length(weights) + 1
    deltas <- list()
    deltas[[n.layers]] <- last.deltas
    # Hardcoded output gradients.
    #deltas[[n.layers]] <- c(reward * (1 - tanh(neurons$z[[n.layers]][1])^2))#, # tanh gradient.
    #reward * 2 * neurons$z[[n.layers]][2]) # x^2 gradient.
    if (n.layers == 2) return(deltas)
    for (i in (n.layers - 1):2) {
        deltas[[i]] <- deltas[[i + 1]] %*% t(as.matrix(weights[[i]][-1, ])) * gradient(neurons$z[[i]])
    }
    deltas
}

update.weights <- function(neurons, weights, deltas, alpha = 1) {
    for (i in 1:length(weights)) {
        # Update weights with a MINUS as this is a MINIMIZATION problem. Add bias unit.
        direction <- t(cbind(1, neurons$a[[i]])) %*% deltas[[i + 1]]
        #direction <- direction / sqrt(as.numeric(t(direction) %*% direction))
        weights[[i]] <- weights[[i]] - alpha * direction
    }
    weights
}