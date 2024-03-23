#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from mgktools.data import Dataset
from mgktools.models import set_model
from mgktools.evaluators.cross_validation import Evaluator, Metric
from mgktools.kernels.PreComputed import calc_precomputed_kernel_config


def save_best_params(
    save_dir: str,
    best_hyperdict: Dict,
    kernel_config,
):
    if save_dir is not None:
        if "alpha" in best_hyperdict:
            open("%s/alpha" % save_dir, "w").write("%s" % best_hyperdict.pop("alpha"))
        elif "C" in best_hyperdict:
            open("%s/C" % save_dir, "w").write("%s" % best_hyperdict.pop("C"))
        kernel_config.update_from_space(best_hyperdict)
        kernel_config.save(path=save_dir)


def bayesian_optimization(
    save_dir: Optional[str],
    datasets: List[Dataset],
    kernel_config,
    task_type: Literal["regression", "binary", "multi-class"],
    model_type: Literal[
        "gpr", "gpr-sod", "gpr-nystrom", "gpr-nle", "svr", "gpc", "svc"
    ],
    metric: Literal[Metric, "log_likelihood"],
    cross_validation: Literal["nfold", "loocv", "Monte-Carlo"],
    nfold: int = None,
    split_type: Literal["random", "scaffold_balanced", "assigned"] = None,
    split_sizes: List[float] = None,
    num_folds: int = 10,
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
    if task_type == "regression":
        assert model_type in ["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle", "svr"]
    elif task_type == "binary":
        assert model_type in ["gpr", "gpc", "svc"]
    else:
        assert model_type in ["gpc", "svc"]

    if metric in ["rmse", "mae", "mse", "max"]:
        maximize = False
    else:
        maximize = True
    if cross_validation == "loocv":
        assert num_folds == 1

    def objective(trial) -> Union[float, np.ndarray]:
        hyperdict = kernel_config.get_trial(trial)
        if alpha_bounds is None:
            pass
        elif d_alpha is None:
            assert model_type in ["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle"]
            hyperdict["alpha"] = trial.suggest_float(
                name="alpha", low=alpha_bounds[0], high=alpha_bounds[1], log=True
            )
        else:
            assert model_type in ["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle"]
            hyperdict["alpha"] = trial.suggest_float(
                name="alpha", low=alpha_bounds[0], high=alpha_bounds[1], step=d_alpha
            )

        if C_bounds is None:
            pass
        elif d_C is None:
            hyperdict["C"] = trial.suggest_float(
                name="C", low=C_bounds[0], high=C_bounds[1], log=True
            )
        else:
            hyperdict["C"] = trial.suggest_float(
                name="C", low=C_bounds[0], high=C_bounds[1], step=d_C
            )

        alpha_ = hyperdict.pop("alpha", alpha)
        C_ = hyperdict.pop("C", C)
        kernel_config.update_from_trial(hyperdict)
        kernel_config.update_kernel()
        obj = []
        if metric == "log_likelihood":
            assert model_type in ["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle"]
            kernel = kernel_config.kernel
            model = set_model(model_type=model_type, kernel=kernel, alpha=alpha_)
            for dataset in datasets:
                obj.append(model.log_marginal_likelihood(X=dataset.X, y=dataset.y))
                dataset.clear_cookie()
            result = np.mean(obj)
            return result
        else:
            for dataset in datasets:
                if dataset.graph_kernel_type == "graph":
                    kernel = calc_precomputed_kernel_config(
                        kernel_config, dataset
                    ).kernel
                    model = set_model(
                        model_type=model_type, kernel=kernel, alpha=alpha_, C=C_
                    )
                    dataset.graph_kernel_type = "pre-computed"
                    evaluator = Evaluator(
                        save_dir=save_dir,
                        dataset=dataset,
                        model=model,
                        task_type=task_type,
                        metrics=[metric],
                        cross_validation=cross_validation,
                        nfold=nfold,
                        split_type=split_type,
                        split_sizes=split_sizes,
                        num_folds=num_folds,
                        return_std=True if task_type == "regression" else False,
                        return_proba=(
                            False
                            if task_type == "regression" or model_type == "gpr"
                            else True
                        ),
                        n_similar=None,
                        verbose=False,
                    )
                    obj.append(evaluator.evaluate())
                    dataset.graph_kernel_type = "graph"
                    dataset.clear_cookie()
                else:
                    # optimize hyperparameters for features with fixed graph kernel hyperparameters.
                    kernel = kernel_config.kernel
                    model = set_model(
                        model_type=model_type, kernel=kernel, alpha=alpha_, C=C_
                    )
                    evaluator = Evaluator(
                        save_dir=save_dir,
                        dataset=dataset,
                        model=model,
                        task_type=task_type,
                        metrics=[metric],
                        cross_validation=cross_validation,
                        nfold=nfold,
                        split_type=split_type,
                        split_sizes=split_sizes,
                        num_folds=num_folds,
                        return_std=True if task_type == "regression" else False,
                        return_proba=(
                            False
                            if task_type == "regression" or model_type == "gpr"
                            else True
                        ),
                        n_similar=None,
                        verbose=False,
                    )
                    obj.append(evaluator.evaluate())
            result = np.mean(obj)
            if maximize:
                return -result
            else:
                return result

    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=seed),
        storage="sqlite:///%s/optuna.db" % save_dir,
        load_if_exists=True,
    )
    n_to_run = num_iters - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
    save_best_params(save_dir=save_dir, best_hyperdict=study.best_params, kernel_config=kernel_config)


def bayesian_optimization_gpr_multi_datasets(
    save_dir: Optional[str],
    datasets: List[Dataset],
    kernel_config,
    tasks_type: List[Literal["regression", "binary", "multi-class"]],
    metrics: List[Literal[Metric]],
    num_iters: int = 100,
    alpha: float = 0.01,
    alpha_bounds: Tuple[float, float] = None,
    d_alpha: float = None,
    seed: int = 0,
):
    def objective(trial) -> Union[float, np.ndarray]:
        hyperdict = kernel_config.get_trial(trial)
        if alpha_bounds is None:
            pass
        elif d_alpha is None:
            hyperdict["alpha"] = trial.suggest_float(
                name="alpha", low=alpha_bounds[0], high=alpha_bounds[1], log=True
            )
        else:
            hyperdict["alpha"] = trial.suggest_float(
                name="alpha", low=alpha_bounds[0], high=alpha_bounds[1], step=d_alpha
            )

        alpha_ = hyperdict.pop("alpha", alpha)
        kernel_config.update_from_space(hyperdict)
        kernel_config.update_kernel()
        obj = []
        for i, dataset in enumerate(datasets):
            metric = metrics[i]
            assert dataset.graph_kernel_type == "graph"
            kernel = kernel_config.kernel
            model = set_model(model_type="gpr", kernel=kernel, alpha=alpha_)
            evaluator = Evaluator(
                save_dir=save_dir,
                dataset=dataset,
                model=model,
                task_type=tasks_type[i],
                metrics=[metric],
                cross_validation="leave-one-out",
                num_folds=1,
                return_std=True,
                return_proba=False,
                n_similar=None,
                verbose=False,
            )
            dataset.clear_cookie()
            if metric in ["rmse", "mae", "mse", "max"]:
                obj.append(evaluator.evaluate())
            elif metric in [
                "r2",
                "spearman",
                "kendall",
                "pearson",
                "roc-auc",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "mcc",
            ]:
                obj.append(-evaluator.evaluate())
            else:
                raise ValueError(f'metric "{metric}" not supported.')
        result = np.mean(obj)
        return result

    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=seed),
        storage="sqlite:///%s/optuna.db" % save_dir,
        load_if_exists=True,
    )
    n_to_run = num_iters - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
    save_best_params(save_dir=save_dir, best_hyperdict=study.best_params, kernel_config=kernel_config)
