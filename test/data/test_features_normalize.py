#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from mgktools.data import Dataset


pure = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO']
feature = [3.2, 4.3, 9.9, 1.1]
targets_1 = [3.4, 4.5, 5.6, 6.7]
targets_2 = [3.1, 14.5, 25.6, 56.7]
df = pd.DataFrame({'pure': pure, 'targets_1': targets_1, 'targets_2': targets_2, 'feature': feature})


@pytest.mark.parametrize('features_generator', [
    ['rdkit_2d'],
    ['rdkit_2d_normalized'],
    ['morgan'],
    ['morgan_count'],
    ['rdkit_2d_normalized', 'morgan'],
])
def test_only_molfeatures(features_generator):
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets_1'],
                              features_generator=features_generator)
    assert dataset.features_mol_scaler is None
    dataset.normalize_features_mol()
    assert dataset.features_mol_scaler is not None
    X = dataset.features_mol_scaler.transform(dataset.X_raw_features_mol)
    assert dataset.X == pytest.approx(X, 1e-5)
    assert X.mean(axis=0) == pytest.approx(np.zeros(X.shape[1]), 1e-5)
    for i, std in enumerate(X.std(axis=0)):
        assert std == pytest.approx(1.0, 1e-5) or std == pytest.approx(0.0, 1e-5)
        if std == pytest.approx(0.0, 1e-5):
            assert X[:, i] == pytest.approx(np.zeros(len(X)), 1e-5) or X[:, i] == pytest.approx(np.ones(len(X)), 1e-5)


@pytest.mark.parametrize('features_generator', [
    ['rdkit_2d'],
    ['rdkit_2d_normalized'],
    ['morgan'],
    ['morgan_count'],
    ['rdkit_2d_normalized', 'morgan'],
])
def test_graph_molfeatures(features_generator):
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets_1'],
                              features_generator=features_generator)
    dataset.graph_kernel_type = 'graph'
    assert dataset.features_mol_scaler is None
    dataset.normalize_features_mol()
    assert dataset.features_mol_scaler is not None
    X = dataset.features_mol_scaler.transform(dataset.X_raw_features_mol)
    assert dataset.X_features_mol == pytest.approx(X, 1e-5)
    assert X.mean(axis=0) == pytest.approx(np.zeros(X.shape[1]), 1e-5)
    for i, std in enumerate(X.std(axis=0)):
        assert std == pytest.approx(1.0, 1e-5) or std == pytest.approx(0.0, 1e-5)
        if std == pytest.approx(0.0, 1e-5):
            assert X[:, i] == pytest.approx(np.zeros(len(X)), 1e-5) or X[:, i] == pytest.approx(np.ones(len(X)), 1e-5)


def test_only_addfeatures():
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets_1'],
                              feature_columns=['feature'])
    assert dataset.features_add_scaler is None
    dataset.normalize_features_add()
    assert dataset.features_add_scaler is not None
    X = dataset.features_add_scaler.transform(dataset.X_raw_features_add)
    assert dataset.X == pytest.approx(X, 1e-5)
    assert X.mean(axis=0) == pytest.approx(np.zeros(X.shape[1]), 1e-5)
    for i, std in enumerate(X.std(axis=0)):
        assert std == pytest.approx(1.0, 1e-5) or std == pytest.approx(0.0, 1e-5)
        if std == pytest.approx(0.0, 1e-5):
            assert X[:, i] == pytest.approx(np.zeros(len(X)), 1e-5) or X[:, i] == pytest.approx(np.ones(len(X)), 1e-5)


def test_graph_addfeatures():
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets_1'],
                              feature_columns=['feature'])
    dataset.graph_kernel_type = 'graph'
    assert dataset.features_add_scaler is None
    dataset.normalize_features_add()
    assert dataset.features_add_scaler is not None
    X = dataset.features_add_scaler.transform(dataset.X_raw_features_add)
    assert dataset.X_features_add == pytest.approx(X, 1e-5)
    assert X.mean(axis=0) == pytest.approx(np.zeros(X.shape[1]), 1e-5)
    for i, std in enumerate(X.std(axis=0)):
        assert std == pytest.approx(1.0, 1e-5) or std == pytest.approx(0.0, 1e-5)
        if std == pytest.approx(0.0, 1e-5):
            assert X[:, i] == pytest.approx(np.zeros(len(X)), 1e-5) or X[:, i] == pytest.approx(np.ones(len(X)), 1e-5)
