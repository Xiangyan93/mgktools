#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *

pure = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO']
targets = [3.1, 14.5, 25.6, 56.7]
df = pd.DataFrame({'pure': pure, 'targets': targets})


@pytest.mark.parametrize('mgk_file', [additive, additive_norm, additive_pnorm, additive_msnorm,
                                      product, product_norm, product_pnorm, product_msnorm])
def test_only_graph(mgk_file):
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    N = len(dataset)
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    K = kernel_config.kernel(dataset.X)
    assert K.shape == (N, N)
    # invertable
    assert np.linalg.det(K) > 10 ** -3
    for i in range(N):
        for j in range(i + 1, N):
            # symmetric
            assert K[i, j] == pytest.approx(K[j, i], 1e-5)
            # diagonal largest
            assert np.sqrt(K[i, i] * K[j, j]) > K[i, j]


@pytest.mark.parametrize('mgk_file', [additive, additive_norm, additive_pnorm, additive_msnorm,
                                      product, product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('features_generator', [
    ['rdkit_2d'],
    ['rdkit_2d_normalized'],
    ['morgan'],
    ['morgan_count'],
    ['rdkit_2d_normalized', 'morgan'],
])
@pytest.mark.parametrize('features_kernel', ['dot_product', 'rbf'])
@pytest.mark.parametrize('normalize_feature', [True, False])
def test_graph_features(mgk_file, features_generator, features_kernel, normalize_feature):
    features_generator = features_generator
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets'],
                              features_generator=features_generator)
    if normalize_feature:
        dataset.normalize_features()
    dataset.graph_kernel_type = 'graph'
    N = len(dataset)
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file],
                                      features_kernel_type=features_kernel,
                                      rbf_length_scale=None if features_kernel == 'dot_product' else [0.5])
    print(kernel_config.kernel.composition)
    K = kernel_config.kernel(dataset.X)
    assert K.shape == (N, N)
    # invertable
    assert np.linalg.det(K) > 10 ** -3
    for i in range(N):
        for j in range(i + 1, N):
            # symmetric
            assert K[i, j] == pytest.approx(K[j, i], 1e-5)
            # diagonal largest
            assert np.sqrt(K[i, i] * K[j, j]) > K[i, j]
