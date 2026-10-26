import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hiarn_explain import predict, sufficient_bounds, sufficient_milp, explain


def brute_sufficient(P, s, fixed, c_hat):
    free = np.where(~fixed)[0]
    for vals in itertools.product([0.0, 1.0], repeat=len(free)):
        x = s.copy()
        x[free] = vals
        _, layers, Wo, b = P
        a = x
        for W in layers:
            a = ((a @ W.T) > 0).astype(float)
        z = a @ Wo.T + b
        if any(z[c] >= z[c_hat] for c in range(len(z)) if c != c_hat):
            return False
    return True


def random_network(rng, m, sizes, n_cls, density=0.6):
    layers, prev = [], m
    for h in sizes:
        W = rng.normal(size=(h, prev)) * (rng.random((h, prev)) < density)
        layers.append(W)
        prev = h
    return np.zeros(m), layers, rng.normal(size=(n_cls, prev)), rng.normal(size=n_cls) * 0.1


def test_toy_example():
    W = np.array([[2.0, 1.0, -1.0], [-1.0, 0.0, 2.0]])
    P = (np.zeros(3), [W], np.eye(2), np.array([0.0, 0.2]))
    s = np.array([1.0, 1.0, 0.0])
    c_hat, fixed = explain(P, s, sufficient_bounds)
    assert c_hat == 0
    assert fixed.tolist() == [True, False, True]


def test_bound_check_and_milp_check_are_sound():
    rng = np.random.RandomState(0)
    for trial in range(60):
        sizes = [int(rng.randint(2, 6))] if trial % 2 == 0 else [int(rng.randint(3, 6)), int(rng.randint(2, 5))]
        P = random_network(rng, 7, sizes, int(rng.randint(2, 4)))
        s = rng.randint(0, 2, 7).astype(float)
        c_hat = int(predict(P, s[None])[0])
        z = s
        for W in P[1]:
            z = ((z @ W.T) > 0).astype(float)
        scores = z @ P[2].T + P[3]
        if np.sort(scores)[-1] - np.sort(scores)[-2] < 1e-9:
            continue
        for _ in range(8):
            fixed = rng.random(7) < 0.5
            exact = brute_sufficient(P, s, fixed, c_hat)
            if sufficient_bounds(P, s, fixed, c_hat):
                assert exact
            if sufficient_milp(P, s, fixed, c_hat):
                assert exact


def test_explanations_are_sufficient_and_minimal_wrt_check():
    rng = np.random.RandomState(1)
    for trial in range(40):
        sizes = [int(rng.randint(2, 6))] if trial % 2 == 0 else [int(rng.randint(3, 6)), int(rng.randint(2, 5))]
        P = random_network(rng, 8, sizes, int(rng.randint(2, 4)))
        s = rng.randint(0, 2, 8).astype(float)
        c_hat, fixed = explain(P, s, sufficient_bounds)
        if not sufficient_bounds(P, s, fixed, c_hat):
            continue
        assert brute_sufficient(P, s, fixed, c_hat)
        for j in np.where(fixed)[0]:
            smaller = fixed.copy()
            smaller[j] = False
            assert not sufficient_bounds(P, s, smaller, c_hat)
        c_m, fixed_m = explain(P, s, sufficient_milp)
        assert brute_sufficient(P, s, fixed_m, c_m)


if __name__ == '__main__':
    test_toy_example()
    test_bound_check_and_milp_check_are_sound()
    test_explanations_are_sufficient_and_minimal_wrt_check()
    print('all tests passed')
