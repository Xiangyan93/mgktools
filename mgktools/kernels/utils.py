# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pickle
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from ..data import Dataset


def get_features_hyperparameters(
        n_features: int,
        features_hyperparameters: List[float] = None,
        features_hyperparameters_bounds: Union[List[Tuple[float, float]], Literal['fixed']] = None,
        features_hyperparameters_file: str = None,
) -> Tuple[Optional[List[float]], Optional[List[Tuple[float, float]]]]:
    """

    Parameters
    ----------
    n_features: dimension of RBF kernel
    features_hyperparameters
    features_hyperparameters_bounds
    features_hyperparameters_file

    Returns
    -------

    """
    if n_features == 0:
        rbf_length_scale, rbf_length_scale_bounds = None, None
    elif features_hyperparameters_file is not None:
        rbf = json.load(open(features_hyperparameters_file))
        rbf_length_scale = rbf['rbf_length_scale']
        rbf_length_scale_bounds = rbf['rbf_length_scale_bounds']
    else:
        rbf_length_scale = features_hyperparameters
        rbf_length_scale_bounds = features_hyperparameters_bounds
        if len(rbf_length_scale) != 1 and len(rbf_length_scale) != n_features:
            raise ValueError(f'The number of features({n_features}) not equal to the number of hyperparameters'
                             f'({len(rbf_length_scale)})')
    return rbf_length_scale, rbf_length_scale_bounds


def get_kernel_config(dataset: Dataset,
                      graph_kernel_type: Literal['graph', 'pre-computed', None],
                      # arguments for vectorized features.
                      features_kernel_type: Literal['dot_product', 'rbf'] = None,
                      rbf_length_scale: List[float] = None,
                      rbf_length_scale_bounds: Literal[List[Tuple[float]], "fixed"] = None,
                      features_hyperparameters_file: str = None,
                      # arguments for marginalized graph kernel
                      mgk_hyperparameters_files: List[str] = None,
                      # arguments for pre-computed kernel
                      kernel_dict: Dict = None,
                      kernel_pkl: str = 'kernel.pkl'):
    if features_kernel_type is None:
        n_features = 0
        rbf_length_scale, rbf_length_scale_bounds = None, None
    elif features_kernel_type == 'dot_product':
        n_features = dataset.N_features_mol + dataset.N_features_add
        rbf_length_scale = None
        rbf_length_scale_bounds = None
    else:
        n_features = dataset.N_features_mol + dataset.N_features_add
        assert n_features != 0
        rbf_length_scale, rbf_length_scale_bounds = get_features_hyperparameters(
            n_features,
            rbf_length_scale,
            rbf_length_scale_bounds,
            features_hyperparameters_file,
        )

    if graph_kernel_type is None:
        params = {
            'features_kernel_type': features_kernel_type,
            'n_features': n_features,
            'rbf_length_scale': rbf_length_scale,  # np.concatenate(rbf_length_scale),
            'rbf_length_scale_bounds': rbf_length_scale_bounds,  # * n_features,
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
            'rbf_length_scale': rbf_length_scale,  # np.concatenate(rbf_length_scale),
            'rbf_length_scale_bounds': rbf_length_scale_bounds,  # * n_features,
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
            'rbf_length_scale': rbf_length_scale,
            'rbf_length_scale_bounds': rbf_length_scale_bounds,  # * n_features,
        }
        from mgktools.kernels.PreComputed import PreComputedKernelConfig
        return PreComputedKernelConfig(**params)
    else:
        raise ValueError()
