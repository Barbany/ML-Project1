"""Stochastic Gradient Descent"""

from utils.costs import compute_loss
from utils.helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return -1/len(y)*tx.transpose().dot(y-tx.dot(w))


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        w = w - gamma*compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
