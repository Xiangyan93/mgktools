#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any, Union
import numpy as np
import copy
from mgktools.data import Dataset
from mgktools.kernels.base import BaseKernelConfig
from mgktools.kernels.GraphKernel import GraphKernelConfig
from mgktools.kernels.HybridKernel import HybridKernelConfig


class PreComputedKernel:
    def __init__(self, X: np.ndarray, K: np.ndarray, theta: np.ndarray):
        idx = np.argsort(X)
        self.X = X[idx]
        self.K = K[idx][:, idx]
        self.theta_ = theta
        self.hyperparameters_ = np.exp(self.theta_)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X_idx = np.searchsorted(self.X, X.ravel())
        Y_idx = np.searchsorted(self.X, Y.ravel()) if Y is not None else X_idx
        if eval_gradient:
            return self.K[X_idx][:, Y_idx], np.zeros((len(X_idx), len(Y_idx), 1))
        else:
            return self.K[X_idx][:, Y_idx]

    def diag(self, X, eval_gradient=False):
        X_idx = np.searchsorted(self.X, X).ravel()
        if eval_gradient:
            return np.diag(self.K)[X_idx], np.zeros((len(X_idx), 1))
        else:
            return np.diag(self.K)[X_idx]

    @property
    def hyperparameters(self):
        return self.hyperparameters_

    @property
    def theta(self):
        return np.log(self.hyperparameters_)

    @theta.setter
    def theta(self, value):
        self.hyperparameters_ = np.exp(value)

    @property
    def n_dims(self):
        return len(self.theta)

    @property
    def bounds(self):
        theta = self.theta.reshape(-1, 1)
        return np.c_[theta, theta]

    @property
    def requires_vector_input(self):
        return False

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(X=self.X, K=self.K, theta=self.theta_)


class PreComputedKernelConfig(BaseKernelConfig):
    def __init__(
        self,
        kernel_dict: Dict,
    ):
        X = kernel_dict["X"]
        K = kernel_dict["K"]
        theta = kernel_dict["theta"]
        self.kernel = PreComputedKernel(X, K, theta)

    def update_kernel(self):
        pass

    def get_space(self) -> Dict:
        return {}

    def update_from_space(self, space: Dict[str, Any]):
        pass

    def get_trial(self, trial) -> Dict:
        return {}

    def update_from_trial(self, trial: Dict[str, Any]):
        pass

    def update_from_theta(self):
        pass
    

def calc_precomputed_kernel_config(
    kernel_config: Union[GraphKernelConfig, HybridKernelConfig], dataset: Dataset
) -> Union[PreComputedKernelConfig, HybridKernelConfig]:
    if isinstance(kernel_config, GraphKernelConfig):
        # Single graph kernel, no feature kernel.
        assert dataset.N_features_mol == dataset.N_features_add == 0
        kernel_dict = kernel_config.get_kernel_dict(
            dataset.X_mol, dataset.X_graph_repr.ravel()
        )
        return PreComputedKernelConfig(kernel_dict=kernel_dict)
    else:
        n_MGK = sum([isinstance(kc, GraphKernelConfig) for kc in kernel_config.kernel_configs])
        if dataset.N_features_mol == dataset.N_features_add == 0:
            # multiple graph kernels, no feature kernel.
            assert n_MGK == len(kernel_config.kernel_configs)
            precomputed_kernel_configs = []
            for i in range(n_MGK):
                kc = kernel_config.kernel_configs[i]
                kernel_dict = kc.get_kernel_dict(
                    dataset.X_mol[:, i], dataset.X_graph_repr[:, i].ravel()
                )
                precomputed_kernel_configs.append(
                    PreComputedKernelConfig(kernel_dict=kernel_dict)
                )
            return HybridKernelConfig(
                kernel_configs=precomputed_kernel_configs,
                composition=kernel_config.composition,
                hybrid_rule=kernel_config.hybrid_rule,
            )
        elif dataset.N_features_mol == 0 and dataset.N_features_add != 0:
            # multiple graph kernel + additional features.
            precomputed_kernel_configs = []
            for i in range(n_MGK):
                kc = kernel_config.kernel_configs[i]
                kernel_dict = kc.get_kernel_dict(
                    dataset.X_mol[:, i], dataset.X_graph_repr[:, i].ravel()
                )
                precomputed_kernel_configs.append(
                    PreComputedKernelConfig(kernel_dict=kernel_dict)
                )
            return HybridKernelConfig(
                kernel_configs=precomputed_kernel_configs
                + kernel_config.kernel_configs[n_MGK:],
                composition=kernel_config.composition,
                hybrid_rule=kernel_config.hybrid_rule,
            )
        elif dataset.N_features_mol != 0 and dataset.N_features_add == 0:
            # multiple graph kernels + molecular features, no feature kernel.
            if n_MGK == 1:
                assert len(kernel_config.kernel_configs) == 2
                kernel_dict = kernel_config.get_kernel_dict(
                    dataset.X_mol, dataset.X_graph_repr.ravel()
                )
                return PreComputedKernelConfig(kernel_dict=kernel_dict)
            else:
                raise NotImplementedError
        else:
            # multiple graph kernels + molecular features + additional features.
            raise NotImplementedError
