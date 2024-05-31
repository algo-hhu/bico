import unittest

import numpy as np
from sklearn.utils._param_validation import InvalidParameterError

from bico import BICO

np.random.seed(42)


class TestBICO(unittest.TestCase):

    example_data = np.random.rand(10000, 10)

    def test_fit(self) -> None:

        bico = BICO(n_clusters=2, random_state=42, fit_coreset=True)
        bico.fit(self.example_data)

        assert isinstance(bico.cluster_centers_, np.ndarray)

    def test_n_clusters(self) -> None:
        bico = BICO(n_clusters=0)

        self.assertRaises(InvalidParameterError, bico.fit, self.example_data)


if __name__ == "__main__":
    unittest.main()
