"""Stochastic Gradient Descent"""

from utils.costs import compute_loss, compute_gradient
from utils.helpers import batch_iter


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_function='mse'):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w, loss_function)
        w = w - gamma * compute_gradient(minibatch_y, minibatch_tx, w, loss_function)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
    return losses, ws
