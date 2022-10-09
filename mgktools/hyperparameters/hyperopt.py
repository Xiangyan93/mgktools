#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
from hyperopt import fmin, hp, tpe
from ..data import Dataset
from ..models import set_model
from ..evaluators.cross_validation import Evaluator, Metric


def save_best_params(save_dir: str,
                     results: List[float],
                     hyperdicts: List[Dict],
                     kernel_config,
                     maximize: bool):
    if maximize:
        best_idx = np.where(results == np.max(results))[0][0]
    else:
        best_idx = np.where(results == np.min(results))[0][0]
    best = hyperdicts[best_idx].copy()
    #
    if save_dir is not None:
        if 'alpha' in best:
            open('%s/alpha' % save_dir, 'w').write('%s' % best.pop('alpha'))
        elif 'C' in best:
            open('%s/C' % save_dir, 'w').write('%s' % best.pop('C'))
        kernel_config.update_from_space(best)
        kernel_config.save_hyperparameters(save_dir)
    return best


def bayesian_optimization(save_dir: Optional[str],
                          datasets: List[Dataset],
                          kernel_config,
                          task_type: Literal['regression', 'binary', 'multi-class'],
                          model_type: Literal['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle', 'svr', 'gpc', 'svc'],
                          metric: Literal[Metric, 'log_likelihood'],
                          split_type: Literal['random', 'scaffold_balanced', 'loocv', 'assigned'] = None,
                          num_iters: int = 100,
                          alpha: float = 0.01,
                          alpha_bounds: Tuple[float, float] = None,
                          d_alpha: float = None,
                          C: float = 10,
                          C_bounds: Tuple[float, float] = None,
                          d_C: float = None,
                          seed: int = 0,
                          # external_test_dataset: Optional[Dataset] = None,
                          ):
    if task_type == 'regression':
        assert model_type in ['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle', 'svr']
    elif task_type == 'binary':
        assert model_type in ['gpr', 'gpc', 'svc']
    else:
        assert model_type in ['gpc', 'svc']

    if metric in ['rmse', 'mae', 'mse', 'max']:
        maximize = False
    else:
        maximize = True

    hyperdicts = []
    results = []

    def objective(hyperdict) -> Union[float, np.ndarray]:
        hyperdicts.append(hyperdict.copy())
        alpha_ = hyperdict.pop('alpha', alpha)
        C_ = hyperdict.pop('C', C)
        kernel_config.update_from_space(hyperdict)
        obj = []
        if metric == 'log_likelihood':
            assert model_type in ['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle']
            kernel = kernel_config.kernel
            model = set_model(model_type=model_type,
                              kernel=kernel,
                              alpha=alpha_)
            for dataset in datasets:
                obj.append(model.log_marginal_likelihood(X=dataset.X, y=dataset.y))
                dataset.clear_cookie()
            result = np.mean(obj)
            results.append(result)
            return result
        else:
            for dataset in datasets:
                if dataset.graph_kernel_type == 'graph':
                    kernel = kernel_config.get_precomputed_kernel_config(dataset).kernel
                    model = set_model(model_type=model_type,
                                      kernel=kernel,
                                      alpha=alpha_,
                                      C=C_)
                    dataset.graph_kernel_type = 'pre-computed'
                    evaluator = Evaluator(save_dir=save_dir,
                                          dataset=dataset,
                                          model=model,
                                          task_type=task_type,
                                          metrics=[metric],
                                          split_type=split_type,
                                          split_sizes=[0.8, 0.2],
                                          num_folds=1 if split_type == 'loocv' else 10,
                                          return_std=True if task_type == 'regression' else False,
                                          return_proba=False if task_type == 'regression' or model_type == 'gpr' else True,
                                          n_similar=None,
                                          verbose=False)
                    obj.append(evaluator.evaluate())
                    dataset.graph_kernel_type = 'graph'
                    dataset.clear_cookie()
                else:
                    kernel = kernel_config.kernel
                    model = set_model(model_type=model_type,
                                      kernel=kernel,
                                      alpha=alpha_,
                                      C=C_)
                    evaluator = Evaluator(save_dir=save_dir,
                                          dataset=dataset,
                                          model=model,
                                          task_type=task_type,
                                          metrics=[metric],
                                          split_type=split_type,
                                          split_sizes=[0.8, 0.2],
                                          num_folds=1 if split_type == 'loocv' else 10,
                                          return_std=True if task_type == 'regression' else False,
                                          return_proba=False if task_type == 'regression' or model_type == 'gpr' else True,
                                          n_similar=None,
                                          verbose=False)
                    obj.append(evaluator.evaluate())
            result = np.mean(obj)
            if maximize:
                results.append(-result)
                return -result
            else:
                results.append(result)
                return result

    SPACE = kernel_config.get_space()

    if alpha_bounds is None:
        pass
    elif d_alpha is None:
        assert model_type in ['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle']
        SPACE['alpha'] = hp.loguniform('alpha',
                                       low=np.log(alpha_bounds[0]),
                                       high=np.log(alpha_bounds[1]))
    else:
        assert model_type in ['gpr', 'gpr-sod', 'gpr-nystrom', 'gpr-nle']
        SPACE['alpha'] = hp.quniform('alpha',
                                     low=alpha_bounds[0],
                                     high=alpha_bounds[1],
                                     q=d_alpha)

    if C_bounds is None:
        pass
    elif d_C is None:
        SPACE['C'] = hp.loguniform('C',
                                   low=np.log(C_bounds[0]),
                                   high=np.log(C_bounds[1]))
    else:
        SPACE['C'] = hp.loguniform('C',
                                   low=C_bounds[0],
                                   high=C_bounds[1],
                                   q=d_C)

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=num_iters,
         rstate=np.random.seed(seed))
    if maximize:
        results = [-r for r in results]
    best_hyperdict = save_best_params(save_dir=save_dir,
                                      results=results,
                                      hyperdicts=hyperdicts,
                                      kernel_config=kernel_config,
                                      maximize=maximize)

    return best_hyperdict, results, hyperdicts
