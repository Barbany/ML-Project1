"""Function used to compute the loss."""


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return 1 / len(y) * sum((y - tx.dot(w)) ** 2)
