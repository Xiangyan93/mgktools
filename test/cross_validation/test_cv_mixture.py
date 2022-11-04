#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil

import pandas as pd
import pytest
import json
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *
from mgktools.models import set_model
from mgktools.evaluators.cross_validation import Evaluator


mixture = [['CCCC', 0.5, 'CCOCCN', 0.5],
           ['CCCCCO', 0.3, 'c1ccccc1O', 0.7],
           ['c1ccccc1', 0.2, 'CCCCC', 0.8],
           ['CCNCCO', 0.4, 'c1ccccc1CC', 0.6],
           ['NCCCC', 0.5, 'NCCOCCN', 0.5],
           ['NCCCCCO', 0.3, 'Nc1ccccc1O', 0.7],
           ['Nc1ccccc1', 0.2, 'NCCCCC', 0.8],
           ['NCCNCCO', 0.4, 'Nc1ccccc1CC', 0.6]]
mixture = [json.dumps(m) for m in mixture]
targets = [3.1, 14.5, 25.6, 56.7, 9.1, 17.5, 22.6, 36.7]
df = pd.DataFrame({'mixture': mixture, 'targets': targets})


@pytest.mark.parametrize('mgk_file', [additive_pnorm, product_pnorm])
@pytest.mark.parametrize('modelsets', [('gpr', None, None, None),
                                       ('gpr-sod', 2, 3, 'smallest_uncertainty'),
                                       ('gpr-sod', 2, 3, 'weight_uncertainty'),
                                       ('gpr-sod', 2, 3, 'mean'),
                                       ('gpr-nystrom', None, 3, None),
                                       ('gpr-nle', None, 3, None)])
@pytest.mark.parametrize('split_type', ['random'])
def test_only_graph(mgk_file, modelsets, split_type):
    model_type, n_estimators, n_samples, consensus_rule = modelsets
    dataset = Dataset.from_df(df=df,
                              mixture_columns=['mixture'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    model = set_model(model_type,
                      kernel=kernel_config.kernel,
                      alpha=0.01,
                      n_estimators=n_estimators,
                      n_samples=n_samples,
                      consensus_rule=consensus_rule
                      )
    Evaluator(save_dir='tmp',
              dataset=dataset,
              model=model,
              task_type='regression',
              metrics=['rmse', 'mae', 'r2'],
              split_type=split_type,
              split_sizes=[0.75, 0.25],
              num_folds=2,
              return_std=True,
              verbose=True,
              n_core=n_samples if model_type == 'gpr-nystrom' else None).evaluate()
    shutil.rmtree('tmp')
