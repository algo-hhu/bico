import unittest

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import InvalidParameterError
from ucimlrepo import fetch_ucirepo

from bico import BICO

np.random.seed(42)


def fetch_dataset(dataset_name: str) -> pd.DataFrame:
    us_census_data_1990 = fetch_ucirepo(name=dataset_name)

    return us_census_data_1990.data.features


class TestBICO(unittest.TestCase):

    example_data = np.random.rand(10000, 10)

    def test_fit(self) -> None:

        bico = BICO(n_clusters=2, random_state=42, fit_coreset=True)
        bico.fit(self.example_data)

        assert isinstance(bico.cluster_centers_, np.ndarray)
        assert isinstance(bico.coreset_points_, np.ndarray)

    def test_n_clusters(self) -> None:
        bico = BICO(n_clusters=0)

        self.assertRaises(InvalidParameterError, bico.fit, self.example_data)

    def test_datasets(self) -> None:
        for dataset_name, short_name in [
            ("US Census Data (1990)", "census"),
            ("Covertype", "covertype"),
        ]:
            data = fetch_dataset(dataset_name)
            d = data.shape[1]
            data.to_csv(f"{short_name}.txt", index=False, header=False)
            del data
            for k in [10, 20, 30]:
                for m in [k * 50, k * 100, k * 200]:
                    with self.subTest(msg=f"{short_name}_k={k}_m={m}"):
                        bico = BICO(n_clusters=k, summary_size=m, random_state=42)
                        for chunk in pd.read_csv(
                            f"{short_name}.txt",
                            delimiter=",",
                            header=None,
                            chunksize=10000,
                        ):
                            bico.partial_fit(chunk.to_numpy(copy=False))
                        bico.partial_fit()

                        py_result = pd.DataFrame(
                            data=bico.coreset_points_,
                            columns=[i for i in range(1, d + 1)],
                        )
                        py_result.insert(0, "weight", bico.coreset_weights_)

                        c_result = pd.read_csv(
                            f"tests/bico_results/{short_name}_{k}_{m}.txt",
                            header=None,
                            delimiter=" ",
                            skiprows=[0],
                        )
                        c_result.rename(
                            columns={col: str(col) for col in c_result.columns},
                            inplace=True,
                        )
                        c_result.rename(columns={"0": "weight"}, inplace=True)
                        c_result["weight"] = c_result["weight"].astype(int)

                        is_close = np.isclose(py_result, c_result)

                        assert is_close.all(), (
                            py_result.values[~is_close] - c_result.values[~is_close]
                        )


if __name__ == "__main__":

    unittest.main()
