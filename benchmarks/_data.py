"""Shared deterministic inputs; no numerical framework is initialized here."""

import numpy as np


def data(n):
    x = np.linspace(-4, 4, n)[:, None]
    return x, (np.sin(1.7 * x[:, 0]) + 0.1 * np.cos(3.1 * x[:, 0]))


def query_points(q, shift=0.0):
    return (
        np.array([[0.137]]) + shift
        if q == 1
        else np.linspace(-3.8 + shift, 3.8 + shift, q)[:, None]
    )
