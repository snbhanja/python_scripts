import numpy as np

def cross_entropy(Y, P):
    # two class cross entropy
    # Write a function that takes as input two lists Y, P,
    # and returns the float corresponding to their cross-entropy.
    # Y is the true labels and P is the predicted probabilities.
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

# Reference:-
# https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
