import numpy as np
import torch
from scipy.special import expit as sigmoid


def random_P(rg, T):
    P = rg.normal(size=(T, T))
    return sigmoid(0.5 * (P + P.T))
