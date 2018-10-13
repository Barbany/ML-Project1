"""Gradient Descent"""

from src.utils.costs import compute_loss


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return -1 / len(y) * tx.transpose().dot(y - tx.dot(w))


def compute_mae_gradient(y, tx, w):
    """Compute the gradient for mae loss."""
    return -1 / len(y) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))])


def gradient_descent(y, tx, initial_w, max_iters, gamma, mse=True):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        if mse:
            w = w - gamma * compute_gradient(y, tx, w)
        else:
            w = w - gamma * compute_mae_gradient(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
