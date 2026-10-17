"""Exhaustive finite-vocabulary check of two heterogeneous proposal models."""
from itertools import product
import unittest

import numpy as np


def target(prefix):
    # Strictly positive and prefix-dependent; every correction branch matters.
    row = np.array([1., 2., 3., 4.])
    row = np.roll(row, sum((i + 1) * token for i, token in enumerate(prefix)) % 4)
    return row / row.sum()


class HybridExactness(unittest.TestCase):
    def test_joint_three_token_distribution(self):
        q1 = np.array([.05, .15, 0., .80])  # MTP, outside DSpine V1={0,1}.
        q2 = np.array([[.1, 0., .9, 0.], [.7, 0., .3, 0.],
                       [.4, 0., .6, 0.], [.85, 0., .15, 0.]])
        # An unrelated model proposes token 2 conditioned on the actual MTP ID.
        observed = np.zeros((4, 4, 4))

        def fill(prefix, mass):
            if len(prefix) == 3:
                observed[tuple(prefix)] += mass
            else:
                for token, probability in enumerate(target(prefix)):
                    fill(prefix + [token], mass * probability)

        def verify(proposals, position, prefix, mass):
            if position == 2:
                fill(prefix, mass)  # bonus, then any remaining ordinary draws
                return
            p = target(prefix)
            q = q1 if position == 0 else q2[proposals[0]]
            token = proposals[position]
            accept = min(1., p[token] / q[token])
            verify(proposals, position + 1, prefix + [token], mass * accept)
            residual = np.maximum(p - q, 0.)
            if residual.sum() > 0:
                residual /= residual.sum()
                for correction, probability in enumerate(residual):
                    fill(prefix + [correction], mass * (1 - accept) * probability)

        for first, second in product(range(4), repeat=2):
            mass = q1[first] * q2[first, second]
            if mass:
                verify([first, second], 0, [], mass)
        expected = np.zeros_like(observed)
        for path in product(range(4), repeat=3):
            mass = 1.
            for i, token in enumerate(path):
                mass *= target(path[:i])[token]
            expected[path] = mass
        np.testing.assert_allclose(observed, expected, atol=2e-16, rtol=0.)
        self.assertAlmostEqual(float(observed.sum()), 1., places=14)

    def test_target_mixture_retention_identity(self):
        p = target([])
        q = np.array([.0, .8, .0, .2])
        for mixing in (0., .2, .57, 1.):
            mixed = mixing * p + (1 - mixing) * q
            np.testing.assert_allclose(np.minimum(mixed / p, 1.),
                mixing + (1 - mixing) * np.minimum(q / p, 1.), atol=1e-15)


if __name__ == '__main__':
    unittest.main()
