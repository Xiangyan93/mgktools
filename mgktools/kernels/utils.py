# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pickle
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from sklearn.gaussian_process.kernels import RBF, DotProduct
from ..data import Dataset


def get_kernel(graph_kernel_type: Literal['graph', 'pre-computed'] = None,
               mgk_files: List[str] = None,
               features_kernel_type: Literal['linear', 'dot_product', 'rbf'] = None,
               features_hyperparameters: Union[float, List[float]] = None,
               features_hyperparameters_file: str = None,
               dataset: Dataset = None,
               kernel_pkl_path: str = None,
               ):
    if mgk_files is None:
        assert graph_kernel_type is None
        # no graph kernel involved.
        if features_kernel_type is None:
            return None
        elif features_kernel_type == 'linear':
            return 'linear'
        elif features_kernel_type == 'dot_product':
            if features_hyperparameters.__class__ == list:
                assert len(features_hyperparameters) == 1
                sigma_0 = features_hyperparameters[0]
            else:
                sigma_0 = features_hyperparameters
            return DotProduct(sigma_0=sigma_0)
        elif features_kernel_type == 'rbf':
            return RBF(length_scale=features_hyperparameters)
        else:
            raise ValueError
    else:
        if graph_kernel_type == 'graph':
            return get_kernel_config(
                dataset=dataset,
                graph_kernel_type='graph',
                mgk_hyperparameters_files=mgk_files,
                features_kernel_type=features_kernel_type,
                features_hyperparameters=features_hyperparameters,
                features_hyperparameters_bounds="fixed",
                features_hyperparameters_file=features_hyperparameters_file
            ).kernel
        elif graph_kernel_type == 'pre-computed':
            assert kernel_pkl_path is not None
            if os.path.exists(kernel_pkl_path):
                return get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type='pre-computed',
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file,
                    kernel_pkl=kernel_pkl_path
                ).kernel
            else:
                dataset.graph_kernel_type = 'graph'
                kernel_config = get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type='graph',
                    mgk_hyperparameters_files=mgk_files,
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file
                )
                kernel_dict = kernel_config.get_kernel_dict(dataset.X, dataset.X_repr.ravel())
                dataset.graph_kernel_type = 'pre-computed'
                pickle.dump(kernel_dict, open(kernel_pkl_path, 'wb'), protocol=4)
                return get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type='pre-computed',
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file,
                    kernel_dict=kernel_dict
                ).kernel
        else:
            raise ValueError


def get_kernel_config(dataset: Dataset,
                      graph_kernel_type: Literal['graph', 'pre-computed', None],
                      # arguments for vectorized features.
                      features_kernel_type: Literal['dot_product', 'rbf'] = None,
                      features_hyperparameters: List[float] = None,
                      features_hyperparameters_bounds: Union[List[Tuple[float]], Literal['fixed']] = None,
                      features_hyperparameters_file: str = None,
                      # arguments for marginalized graph kernel
                      mgk_hyperparameters_files: List[str] = None,
                      # arguments for pre-computed kernel
                      kernel_dict: Dict = None,
                      kernel_pkl: str = 'kernel.pkl'):
    if features_hyperparameters_file is not None:
        assert features_hyperparameters is None
        assert features_hyperparameters_bounds is None
        f = json.load(open(features_hyperparameters_file))
        features_kernel_type = f['features_kernel_type']
        features_hyperparameters = f['features_hyperparameters']
        features_hyperparameters_bounds = f['features_hyperparameters_bounds']
        assert features_kernel_type in ['rbf', 'dot_product']

    if features_kernel_type is None:
        n_features = 0
        assert features_hyperparameters is None
        assert features_hyperparameters_bounds in ["fixed", None]
    else:
        n_features = dataset.N_features_mol + dataset.N_features_add
        assert n_features != 0
        if features_kernel_type == 'dot_product':
            assert len(features_hyperparameters) == 1
            if features_hyperparameters_bounds != "fixed":
                assert len(features_hyperparameters_bounds) == 1
        elif features_kernel_type == 'rbf':
            if len(features_hyperparameters) != 1 and len(features_hyperparameters) != n_features:
                raise ValueError(f'The number of features({n_features}) not equal to the number of hyperparameters'
                                 f'({len(features_hyperparameters)})')
        else:
            raise ValueError

    if graph_kernel_type is None:
        params = {
            'features_kernel_type': features_kernel_type,
            'n_features': n_features,
            'features_hyperparameters': features_hyperparameters,  # np.concatenate(rbf_length_scale),
            'features_hyperparameters_bounds': features_hyperparameters_bounds,  # * n_features,
        }
        from mgktools.kernels.FeatureKernelConfig import FeatureKernelConfig
        return FeatureKernelConfig(**params)
    elif graph_kernel_type == 'graph':
        mgk_hyperparameters_files = [
            json.load(open(j)) for j in mgk_hyperparameters_files
        ]
        assert dataset.N_MGK + dataset.N_conv_MGK == len(mgk_hyperparameters_files)
        params = {
            'N_MGK': dataset.N_MGK,
            'N_conv_MGK': dataset.N_conv_MGK,
            'graph_hyperparameters': mgk_hyperparameters_files,
            'unique': False,
            'features_kernel_type': features_kernel_type,
            'n_features': n_features,
            'features_hyperparameters': features_hyperparameters,  # np.concatenate(rbf_length_scale),
            'features_hyperparameters_bounds': features_hyperparameters_bounds,  # * n_features,
        }
        from mgktools.kernels.GraphKernel import GraphKernelConfig
        return GraphKernelConfig(**params)
    elif graph_kernel_type == 'pre-computed':
        if dataset.data[0].features_add is None:
            n_features = 0
        else:
            n_features = dataset.data[0].features_add.shape[1]

        if kernel_dict is None:
            kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
        params = {
            'kernel_dict': kernel_dict,
            'features_kernel_type': features_kernel_type,
            'n_features': n_features,
            'features_hyperparameters': features_hyperparameters,
            'features_hyperparameters_bounds': features_hyperparameters_bounds,  # * n_features,
        }
        from mgktools.kernels.PreComputed import PreComputedKernelConfig
        return PreComputedKernelConfig(**params)
    else:
        raise ValueError()
