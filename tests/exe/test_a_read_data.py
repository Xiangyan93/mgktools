# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import shutil
from mgktools.exe.run import mgk_read_data


CWD = os.path.dirname(os.path.abspath(__file__))
if os.path.exists('%s/tmp' % CWD):
    shutil.rmtree('%s/tmp' % CWD, ignore_errors=True)
os.mkdir('%s/tmp' % CWD)


@pytest.mark.parametrize('input1', [
    ('freesolv', ['smiles'], ['freesolv'], []),
    ('bace', ['smiles'], ['bace'], []),
    ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX'], []),
    ('np', ['smiles1', 'smiles2'], ['np'], []),
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('input2', [
    (None, None, None),
    (['morgan'], 'concat', 2048),
    (['rdkit_2d_normalized'], 'concat', 200),
    (['rdkit_2d', 'morgan_count'], 'concat', 2248),
    (['rdkit_2d', 'morgan_count'], 'mean', 2248)
])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData(input1, input2, features_scaling):
    dataset, smiles_columns, targets_columns, features_columns = input1
    features_generators, features_combination, n_features = input2
    save_dir = f'{CWD}/tmp/{dataset}_{','.join(smiles_columns)}_' \
               f'{','.join(targets_columns)}_{','.join(features_columns)}_' \
               f'{None if features_generators is None else ",".join(features_generators)}_' \
               f'{features_combination}_{features_scaling}'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--smiles_columns'] + smiles_columns + [
        '--targets_columns'] + targets_columns + [
        '--n_jobs', '6',
    ]
    if features_columns:
        arguments += ['--features_columns'] + features_columns
    if features_generators:
        arguments += ['--features_generator'] + features_generators + [
            '--features_combination', features_combination
        ]
    if features_scaling:
        if features_columns:
            arguments += ['--features_add_normalize']
        if features_generators:
            arguments += ['--features_mol_normalize']
    mgk_read_data(arguments)
    assert os.path.exists('%s/dataset.pkl' % save_dir)
