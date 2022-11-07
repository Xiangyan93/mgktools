#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import pytest
from mgktools.data import Dataset


pure = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO']
targets_1 = [3.4, 4.5, 5.6, 6.7]
targets_2 = [3.1, 14.5, 25.6, 56.7]
df = pd.DataFrame({'pure': pure, 'targets_1': targets_1, 'targets_2': targets_2})


@pytest.mark.parametrize('testset', [
    (['targets_1'], (4,)),
    (['targets_1', 'targets_2'], (4, 2)),
])
def test_only_graph(testset):
    targets_column, yshape = testset
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=targets_column)
    dataset.graph_kernel_type = 'graph'
    assert dataset.X.shape == (4, 1)
    assert dataset.y.shape == yshape


@pytest.mark.parametrize('testset', [
    ('morgan', 2048),
    ('rdkit_2d', 200),
    ('rdkit_2d_normalized', 200),
    ('rdkit_208', 208),
])
def test_only_fingerprints(testset):
    features_generator, n_features = testset
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets_1'],
                              features_generator=[features_generator])
    assert dataset.X.shape == (4, n_features)
    assert dataset.y.shape == (4,)


@pytest.mark.parametrize('testset', [
    ('morgan', 2048),
    ('rdkit_2d', 200),
    ('rdkit_2d_normalized', 200),
    ('rdkit_208', 208),
])
def test_graph_fingerprints(testset):
    features_generator, n_features = testset
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets_1'],
                              features_generator=[features_generator])
    dataset.graph_kernel_type = 'graph'
    assert dataset.X.shape == (4, 1 + n_features)
    assert dataset.y.shape == (4,)
