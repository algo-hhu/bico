import ctypes
import time
from numbers import Integral
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    _fit_context,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import Interval

import bico._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(bico._core.__file__)


class BICO(BaseEstimator, ClusterMixin, ClassNamePrefixFeaturesOutMixin):
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "summary_size": [None, Interval(Integral, 1, None, closed="left")],
        "n_random_projections": [None, Interval(Integral, 1, None, closed="left")],
        "random_state": [None, Interval(Integral, 0, None, closed="left")],
        "fit_coreset": [bool],
        "coreset_estimator": [None, BaseEstimator],
    }

    _CORESET_ESTIMATOR_ERROR = (
        "BICO computes a coreset. If you want to compute labels, "
        "then you need to call another estimator on the coreset. "
        "This is done automatically by using the `fit_coreset=True` parameter during "
        "the initialization of the class. "
        "The default estimator is KMeans, but you can pass any other estimator to "
        "the `coreset_estimator` parameter."
    )

    _NOT_FITTED_CORESET_ERROR = (
        "This BICO instance is not fitted yet. Call `fit` "
        "or `partial_fit` before using its results. If you are using "
        "`partial_fit`, then you need to call `partial_fit` with no arguments to "
        "compute the coreset."
    )

    def __init__(
        self,
        n_clusters: int,
        summary_size: Optional[int] = None,
        n_random_projections: Optional[int] = None,
        random_state: Optional[int] = None,
        fit_coreset: bool = False,
        coreset_estimator: Optional[BaseEstimator] = None,
    ):
        self.n_clusters = n_clusters
        self.summary_size = (
            summary_size if summary_size is not None else n_clusters * 200
        )
        self.n_random_projections = n_random_projections
        self.random_state = random_state
        self.fit_coreset = fit_coreset
        self.coreset_estimator = coreset_estimator

    @property
    def labels_(self) -> np.ndarray:
        if not hasattr(self, "_labels"):
            raise NotFittedError(self._CORESET_ESTIMATOR_ERROR)
        return self._labels

    @property
    def inertia_(self) -> float:
        if not hasattr(self, "_inertia"):
            raise NotFittedError(self._CORESET_ESTIMATOR_ERROR)
        return self._inertia

    @property
    def cluster_centers_(self) -> np.ndarray:
        if not hasattr(self, "_cluster_centers"):
            raise NotFittedError(self._CORESET_ESTIMATOR_ERROR)
        return self._cluster_centers

    @property
    def coreset_points_(self) -> np.ndarray:
        if not hasattr(self, "_coreset_points"):
            raise NotFittedError(self._NOT_FITTED_CORESET_ERROR)
        return self._coreset_points

    @property
    def coreset_weights_(self) -> np.ndarray:
        if not hasattr(self, "_coreset_weights"):
            raise NotFittedError(self._NOT_FITTED_CORESET_ERROR)
        return self._coreset_weights

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
    ) -> "BICO":
        return self._fit(X, partial=False)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(
        self,
        X: Optional[Sequence[Sequence[float]]] = None,
        y: Any = None,
    ) -> "BICO":
        if X is None:
            self._compute_coreset()
            return self
        else:
            return self._fit(X, partial=True)

    def _fit_coreset(
        self,
    ) -> None:
        if self.coreset_estimator is None:
            from sklearn.cluster import KMeans

            self.coreset_estimator = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )

        self.coreset_estimator.fit(
            self._coreset_points, sample_weight=self._coreset_weights
        )
        self._cluster_centers: np.ndarray = self.coreset_estimator.cluster_centers_
        self._labels: np.ndarray = self.coreset_estimator.labels_
        self._inertia: float = self.coreset_estimator.inertia_

    def _compute_coreset(self, fit_coreset: bool = False) -> "BICO":
        if not hasattr(self, "bico_obj_"):
            raise NotFittedError(
                "This BICO instance is not fitted yet. " "Call `fit` or `partial_fit`."
            )

        c_coreset_weights = (ctypes.c_double * self.summary_size)()
        c_points = (ctypes.c_double * self.n_features_in_ * self.summary_size)()

        _DLL.compute.restype = ctypes.c_size_t
        n_found_points = _DLL.compute(self.bico_obj_, c_coreset_weights, c_points)

        self._coreset_weights: np.ndarray = np.ctypeslib.as_array(c_coreset_weights)[
            :n_found_points
        ]
        self._coreset_points: np.ndarray = np.ctypeslib.as_array(
            c_points, shape=(self.summary_size, self.n_features_in_)
        )[:n_found_points]

        self._n_features_out = n_found_points

        if self.fit_coreset or fit_coreset:
            self._fit_coreset()

        return self

    def _fit(
        self, X: Sequence[Sequence[float]], partial: bool, fit_coreset: bool = False
    ) -> "BICO":
        has_bico_obj = getattr(self, "bico_obj_", None)
        first_call = not (partial and has_bico_obj)

        _X = np.array(X, dtype=np.float64, order="C", copy=False)
        assert isinstance(_X, np.ndarray)

        if first_call:
            self.n_features_in_: int = _X.shape[1]
            _seed = int(time.time()) if self.random_state is None else self.random_state

            # In Melanie's thesis, p = d
            if self.n_random_projections is None:
                self.n_random_projections = self.n_features_in_

            # Declare c types
            c_d = ctypes.c_uint(self.n_features_in_)
            c_k = ctypes.c_uint(self.n_clusters)
            c_p = ctypes.c_uint(self.n_random_projections)
            c_m = ctypes.c_uint(self.summary_size)
            c_random_state = ctypes.c_size_t(_seed)

            _DLL.init.restype = ctypes.POINTER(ctypes.c_void_p)
            self.bico_obj_: Any = _DLL.init(c_d, c_k, c_p, c_m, c_random_state)

        c_array = _X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_n = ctypes.c_uint(_X.shape[0])
        _DLL.addData(self.bico_obj_, c_array, c_n)

        if not partial or fit_coreset:
            self._compute_coreset(fit_coreset)

        return self

    def __del__(self) -> None:
        if hasattr(self, "bico_obj_"):
            _DLL.freeBico(self.bico_obj_)
            del self.bico_obj_

    def fit_predict(
        self, X: Sequence[Sequence[float]], y: Any = None, **kwargs: Any
    ) -> np.ndarray:
        self._fit(X, partial=False, fit_coreset=True)
        return self.labels_

    def predict(self, X: Sequence[Sequence[float]]) -> Any:
        self._fit_coreset()

        if self.coreset_estimator is None:
            raise NotFittedError(self._CORESET_ESTIMATOR_ERROR)

        return self.coreset_estimator.predict(X)
