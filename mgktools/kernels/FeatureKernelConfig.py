#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import json
import pickle

import numpy as np
from hyperopt import hp
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct


class FeatureKernelConfig:
    def __init__(self,
                 features_kernel_type: Literal['dot_product', 'rbf'] = None,
                 n_features: int = 0,
                 features_hyperparameters: List[float] = None,
                 features_hyperparameters_bounds: List[Tuple[float]] = None):
        self.features_kernel_type = features_kernel_type
        self.n_features = n_features
        self.features_hyperparameters = features_hyperparameters
        self.features_hyperparameters_bounds = features_hyperparameters_bounds
        if self.__class__ == FeatureKernelConfig:
            self._update_kernel()

    def get_kernel_dict(self, X: np.ndarray, X_labels: List[str]) -> Dict:
        K = self.kernel(X)
        return {
            'X': X_labels,
            'K': K,
            'theta': self.kernel.theta
        }

    def _update_kernel(self):
        kernel = self._get_features_kernel()
        if len(kernel) != 0:
            self.kernel = self._get_features_kernel()[0]

    def _get_features_kernel(self) -> List:
        if self.features_kernel_type == 'dot_product':
            return [DotProduct(sigma_0=self.features_hyperparameters[0],
                               sigma_0_bounds=self.features_hyperparameters_bounds[0]
                               if self.features_hyperparameters_bounds != 'fixed' else 'fixed')]
        elif self.features_kernel_type == 'rbf':
            assert self.n_features != 0
            if len(self.features_hyperparameters) != 1 and len(self.features_hyperparameters) != self.n_features:
                raise ValueError('features_mol and hyperparameters must be the'
                                 ' same length')
            add_kernel = RBF(length_scale=self.features_hyperparameters,
                             length_scale_bounds=self.features_hyperparameters_bounds)
            # ConstantKernel(1.0, (1e-3, 1e3)) * \
            return [add_kernel]
        else:
            return []

    # functions for Bayesian optimization of hyperparameters.
    def get_space(self):
        SPACE = dict()
        if self.features_hyperparameters is not None and self.features_hyperparameters_bounds != 'fixed':
            for i in range(len(self.features_hyperparameters)):
                hp_key = 'features_kernel:%d:' % i
                hp_ = self._get_hp(hp_key, [self.features_hyperparameters[i],
                                            self.features_hyperparameters_bounds[i]])
                if hp_ is not None:
                    SPACE[hp_key] = hp_
        return SPACE

    def update_from_space(self, hyperdict: Dict[str, Union[int, float]]):
        for key, value in hyperdict.items():
            n, term, microterm = key.split(':')
            # RBF kernels
            if n == 'features_kernel':
                n_features = int(term)
                self.features_hyperparameters[n_features] = value
        self._update_kernel()

    @staticmethod
    def _get_hp(key, value):
        if value[1] == 'fixed':
            return None
        elif value[0] in ['Additive', 'Tensorproduct']:
            return hp.choice(key, value[1])
        elif len(value) == 2:
            return hp.uniform(key, low=value[1][0], high=value[1][1])
        elif len(value) == 3:
            return hp.quniform(key, low=value[1][0], high=value[1][1],
                               q=value[2])
        else:
            raise RuntimeError('.')

    # save functions.
    def save_hyperparameters(self, path: str):
        if self.features_hyperparameters is not None:
            rbf = {
                'features_kernel_type': self.features_kernel_type,
                'features_hyperparameters': self.features_hyperparameters,
                'features_hyperparameters_bounds': self.features_hyperparameters_bounds
            }
            open(os.path.join(path, 'features_hyperparameters.json'), 'w').write(
                json.dumps(rbf, indent=1, sort_keys=False))

    def save_kernel_matrix(self, path: str, X: np.ndarray, X_labels: List[str]):
        """Save kernel.pkl file that used for preCalc kernels."""
        kernel_dict = self.get_kernel_dict(X, X_labels)
        kernel_pkl = os.path.join(path, 'kernel.pkl')
        pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)
