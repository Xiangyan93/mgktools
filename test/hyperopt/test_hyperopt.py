#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *
from mgktools.hyperparameters.hyperopt import bayesian_optimization
from mgktools.models.regression.GPRgraphdot import GPR


pure = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO', 'OCCCO']
targets = [3.1, 14.5, 25.6, 56.7, 12.3]
df = pd.DataFrame({'pure': pure, 'targets': targets})


@pytest.mark.parametrize('mgk_file', [additive, additive_norm, additive_pnorm, additive_msnorm,
                                      product, product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('split_type', ['random', 'loocv'])
def test_bayesian(mgk_file, split_type):
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    best_hyperdict, results, hyperdicts = bayesian_optimization(save_dir=None,
                                                                dataset=dataset,
                                                                kernel_config=kernel_config,
                                                                model_type='gpr',
                                                                task_type='regression',
                                                                metric='rmse',
                                                                split_type=split_type,
                                                                num_iters=2,
                                                                alpha_bounds=(0.001, 0.02),
                                                                d_alpha=0.001)
    best_hyperdict, results, hyperdicts = bayesian_optimization(save_dir=None,
                                                                dataset=dataset,
                                                                kernel_config=kernel_config,
                                                                model_type='gpr',
                                                                task_type='regression',
                                                                metric='log_likelihood',
                                                                num_iters=2,
                                                                alpha_bounds=(0.001, 0.02),
                                                                d_alpha=0.001)


@pytest.mark.parametrize('mgk_file', [additive, additive_norm, additive_pnorm, additive_msnorm,
                                      product, product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('loss_function', ['loocv', 'likelihood'])
@pytest.mark.parametrize('optimizer', ['L-BFGS-B', 'SLSQP'])
def test_gradient(mgk_file, loss_function, optimizer):
    dataset = Dataset.from_df(df=df,
                              pure_columns=['pure'],
                              target_columns=['targets'])
    dataset.graph_kernel_type = 'graph'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    gpr = GPR(kernel=kernel_config.kernel,
              optimizer=optimizer,
              alpha=0.01,
              normalize_y=True)
    gpr.fit(dataset.X, dataset.y, loss=loss_function, verbose=True)
