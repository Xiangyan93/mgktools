#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import json
import pandas as pd
import numpy as np
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *

mixture = [['CCCC', 0.5, 'CCOCCN', 0.5],
           ['CCCCCO', 0.3, 'c1ccccc1O', 0.7],
           ['c1ccccc1', 0.2, 'CCCCC', 0.8],
           ['CCNCCO', 0.4, 'c1ccccc1CC', 0.6]]
mixture = [json.dumps(m) for m in mixture]
targets = [3.1, 14.5, 25.6, 56.7]
df = pd.DataFrame({'mixture': mixture, 'targets': targets})


@pytest.mark.parametrize('mgk_file', [additive, additive_pnorm, product, product_pnorm])
def test_only_graph(mgk_file):
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
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


@pytest.mark.parametrize('features_generator', [
    ['rdkit_2d'],
    ['rdkit_2d_normalized'],
    ['morgan'],
    ['morgan_count'],
    ['rdkit_2d_normalized', 'morgan'],
])
@pytest.mark.parametrize('features_kernel', ['dot_product', 'rbf'])
@pytest.mark.parametrize('features_combination', ['concat', 'mean'])
@pytest.mark.parametrize('normalize_feature', [True, False])
def test_only_features(features_generator, features_kernel, features_combination, normalize_feature):
    features_generator = features_generator
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
                              target_columns=['targets'],
                              features_generator=features_generator,
                              features_combination=features_combination)
    if normalize_feature:
        dataset.normalize_features_mol()
    dataset.graph_kernel_type = None
    N = len(dataset)
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=None,
                                      mgk_hyperparameters_files=None,
                                      features_kernel_type=features_kernel,

                                      features_hyperparameters=[0.5],
                                      features_hyperparameters_bounds='fixed')
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


@pytest.mark.parametrize('mgk_file', [additive, additive_pnorm, product, product_pnorm])
@pytest.mark.parametrize('features_generator', [
    ['rdkit_2d'],
    ['rdkit_2d_normalized'],
    ['morgan'],
    ['morgan_count'],
    ['rdkit_2d_normalized', 'morgan'],
])
@pytest.mark.parametrize('features_kernel', ['dot_product', 'rbf'])
@pytest.mark.parametrize('features_combination', ['concat', 'mean'])
@pytest.mark.parametrize('normalize_feature', [True, False])
def test_graph_features(mgk_file, features_generator, features_kernel, features_combination, normalize_feature):
    features_generator = features_generator
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
                              target_columns=['targets'],
                              features_generator=features_generator,
                              features_combination=features_combination)
    if normalize_feature:
        dataset.normalize_features_mol()
    dataset.graph_kernel_type = 'graph'
    N = len(dataset)
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file],
                                      features_kernel_type=features_kernel,

                                      features_hyperparameters=[0.5],
                                      features_hyperparameters_bounds='fixed')
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
