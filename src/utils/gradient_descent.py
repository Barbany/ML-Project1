"""Gradient Descent"""

from utils.costs import compute_loss, compute_gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_function)
        w = w - gamma * compute_gradient(y, tx, w, loss_function)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
