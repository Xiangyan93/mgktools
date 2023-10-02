#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from ..interpret.utils import save_mols_pkl
from ..data import Dataset, dataset_split
from .metric import Metric, eval_metric_func


class Evaluator:
    def __init__(self,
                 save_dir: str,
                 dataset: Dataset,
                 model,
                 task_type: Literal['regression', 'binary', 'multi-class'],
                 metrics: List[Metric],
                 split_type: Literal['random', 'scaffold_balanced', 'loocv'],
                 split_sizes: List[float] = None,
                 num_folds: int = 1,
                 return_std: bool = False,
                 return_proba: bool = False,
                 evaluate_train: bool = False,
                 n_similar: Optional[int] = None,
                 kernel=None,
                 n_core: int = None,
                 atomic_attribution: bool = False,
                 seed: int = 0,
                 verbose: bool = True
                 ):
        """Evaluator that evaluate the performance of Monte Carlo cross-validation.

        Parameters
        ----------
        save_dir:
            The directory that save all output files.
        dataset:
        model:
        task_type:
        split_type
        split_sizes
        metrics
        num_folds
        return_std:
            If True, the regression model will output posterior uncertainty.
        return_proba:
            If True, the classification model will output probability.
        evaluate_train
        n_similar:
            n_similar molecules in the training set that are most similar to the molecule to be predicted will be
            outputed.
        kernel:
            if n_similar is not None, kernel must not be None, too.
        n_core:
            useful for nystrom approximation. number of sample to be randomly selected in the core set.
        seed
        """
        self.save_dir = save_dir
        if self.write_file and not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.dataset = dataset
        self.model = model
        self.task_type = task_type
        self.split_type = split_type
        self.split_sizes = split_sizes
        self.metrics = metrics
        self.num_folds = num_folds
        self.return_std = return_std
        self.return_proba = return_proba
        self.evaluate_train = evaluate_train
        self.n_similar = n_similar
        self.kernel = kernel
        self.n_core = n_core
        self.atomic_attribution = atomic_attribution
        self.seed = seed
        self.verbose = verbose

    @property
    def write_file(self) -> bool:
        if self.save_dir is None:
            return False
        else:
            return True

    def evaluate(self, external_test_dataset: Optional[Dataset] = None):
        # Leave-One-Out cross validation
        if self.split_type == 'loocv':
            return self._evaluate_loocv()
        # Initialization
        train_metrics_results = dict()
        test_metrics_results = dict()
        for metric in self.metrics:
            train_metrics_results[metric] = []
            test_metrics_results[metric] = []

        if external_test_dataset is not None:
            assert self.split_type is None
            dataset_train = self.dataset
            dataset_test = external_test_dataset
            train_metrics, test_metrics = self.evaluate_train_test(dataset_train, dataset_test,
                                                                   output_tag='ext')
            for j, metric in enumerate(self.metrics):
                if train_metrics is not None:
                    train_metrics_results[metric].append(train_metrics[j])
                if test_metrics is not None:
                    test_metrics_results[metric].append(test_metrics[j])
        else:
            for i in range(self.num_folds):
                # data splits
                if len(self.split_sizes) == 2:
                    dataset_train, dataset_test = dataset_split(
                        self.dataset,
                        split_type=self.split_type,
                        sizes=self.split_sizes,
                        seed=self.seed + i)
                # the second part, validation set, is abandoned.
                elif len(self.split_sizes) == 3:
                    dataset_train, _, dataset_test = dataset_split(
                        self.dataset,
                        split_type=self.split_type,
                        sizes=self.split_sizes,
                        seed=self.seed + i)
                else:
                    raise ValueError('split_sizes must be 2 or 3.')
                train_metrics, test_metrics = self.evaluate_train_test(
                    dataset_train, dataset_test,
                    output_tag='%d' % i)
                for j, metric in enumerate(self.metrics):
                    if train_metrics is not None:
                        train_metrics_results[metric].append(train_metrics[j])
                    if test_metrics is not None:
                        test_metrics_results[metric].append(test_metrics[j])
        if self.evaluate_train:
            self._log('\nTraining set:')
            for metric, result in train_metrics_results.items():
                self._log('%s: %.5f +/- %.5f' % (metric, np.nanmean(result), np.nanstd(result)))
                # self._log(np.asarray(result).ravel())
        self._log('\nTest set:')
        for metric, result in test_metrics_results.items():
            self._log('%s: %.5f +/- %.5f' % (metric, np.nanmean(result), np.nanstd(result)))
        return np.nanmean(test_metrics_results[self.metrics[0]])

    def evaluate_train_test(self, dataset_train: Dataset,
                            dataset_test: Dataset,
                            output_tag: str = '0') -> Tuple[Optional[List[float]],
                                                            Optional[
                                                            List[float]]]:
        train_log = 'train_%s.csv' % output_tag
        test_log = 'test_%s.csv' % output_tag

        X_train = dataset_train.X
        y_train = dataset_train.y
        repr_train = dataset_train.repr.ravel()
        X_test = dataset_test.X
        y_test = dataset_test.y
        repr_test = dataset_test.repr.ravel()
        # Find the most similar sample in training sets.
        if self.n_similar is None:
            y_similar = None
        else:
            y_similar = self.get_similar_info(X_test, X_train, repr_train, self.n_similar)

        train_metrics = None
        self.fit(X_train, y_train)

        # save results test_*.log
        test_metrics = self._eval(X_test, y_test, repr_test, y_similar,
                                  logfile=None if test_log is None else '%s/%s' % (self.save_dir, test_log))

        if self.atomic_attribution:
            self.interpret(dataset_test, output_tag=output_tag)

        if self.evaluate_train:
            train_metrics = self._eval(X_train, y_train, repr_train, repr_train,
                                       logfile=None if train_log is None else '%s/%s' % (self.save_dir, train_log))

        return train_metrics, test_metrics

    def _evaluate_loocv(self):
        X, y, repr = self.dataset.X, self.dataset.y, self.dataset.repr.ravel()
        if self.n_similar is not None:
            y_similar = self.get_similar_info(X, X, repr, self.n_similar)
        else:
            y_similar = None
        """
        # optimize hyperparameters.
        if self.args.optimizer is not None:
            self.model.fit(X, y, loss='loocv', verbose=True)
        """
        loocv_metrics = self._eval(X, y, repr, y_similar,
                                   logfile='%s/%s' % (self.save_dir, 'loocv.csv'))
        self._log('LOOCV:')
        for i, metric in enumerate(self.metrics):
            self._log('%s: %.5f' % (metric, loocv_metrics[i]))
        return loocv_metrics[0]

    def fit(self, X, y):
        if self.n_core is not None:
            idx = np.random.choice(np.arange(len(X)), self.n_core, replace=False)
            C_train = X[idx]
            self.model.fit(C_train, X, y)
        # elif self.args.dataset_type == 'regression' and self.args.model_type == 'gpr' and not self.args.ensemble:
        #    self.model.fit(X_train, y_train, loss=self.args.loss, verbose=True)
        else:
            self.model.fit(X, y)

    def predict(self, X, y: np.ndarray, repr: List[str], y_similar: List[str] = None):
        if self.split_type == 'loocv':
            y_pred, y_std = self.model.predict_loocv(X, y, return_std=True)
        elif self.return_std:
            y_pred, y_std = self.model.predict(X, return_std=True)
        elif self.return_proba:
            y_pred = self.model.predict_proba(X)
            y_std = None
        else:
            y_pred = self.model.predict(X)
            y_std = None
        if y is None:
            pred_dict = {
                'predict': y_pred,
                'repr': repr,
            }
        elif y.ndim == 2:
            assert y_pred.ndim == 2
            assert y_std is None or y_std.ndim == 2
            pred_dict = {}
            for i in range(y.shape[1]):
                pred_dict['target_%d' % i] = y[:, i]
                pred_dict['predict_%d' % i] = y_pred[:, i]
                if y_std is not None:
                    pred_dict['uncertainty_%d' % i] = y_std[:, i]
            pred_dict['repr'] = repr
        else:
            pred_dict = {
                'target': y,
                'predict': y_pred,
                'repr': repr,
            }
            if y_std is not None:
                pred_dict['uncertainty'] = y_std
        if y_similar is not None:
            pred_dict['y_similar'] = y_similar
        return self.df_output(**pred_dict)

    def interpret(self, dataset_test, output_tag: str):
        X_test = dataset_test.X
        mols_to_be_interpret = dataset_test.mols
        batch_size = 100
        N_batch = math.ceil(len(mols_to_be_interpret) / batch_size)
        for i in tqdm(range(N_batch)):
            start = batch_size * i
            end = batch_size * (i + 1)
            if end > len(mols_to_be_interpret):
                end = len(mols_to_be_interpret)
            g = X_test[start:end, :]
            y_nodes = self.model.predict_nodal(g)
            k = 0
            for j in range(start, end):
                m = mols_to_be_interpret[j]
                assert len(m) == 1, 'interpretability is only valid for single-graph data'
                mol = m[0]
                for atom in mol.GetAtoms():
                    atom.SetProp('atomNote', '%.6f' % y_nodes[k])
                    k += 1
            assert k == len(y_nodes)
        save_mols_pkl(mols=[m[0]for m in mols_to_be_interpret], path=self.save_dir, filename='imgk_%s.pkl' % output_tag)

    def get_similar_info(self, X, X_train, X_repr, n_most_similar) -> List[str]:
        K = self.kernel(X, X_train)
        assert (K.shape == (len(X), len(X_train)))
        similar_info = []
        kindex = self.get_most_similar_graphs(K, n=n_most_similar)
        for i, index in enumerate(kindex):
            def round5(x):
                return ',%.5f' % x

            k = list(map(round5, K[i][index]))
            repr = np.asarray(X_repr)[index]
            info = ';'.join(list(map(str.__add__, repr, k)))
            similar_info.append(info)
        return similar_info

    @staticmethod
    def get_most_similar_graphs(K, n=5):
        return np.argsort(-K)[:, :min(n, K.shape[1])]

    @staticmethod
    def df_output(**kwargs):
        df = kwargs.copy()
        for key, value in kwargs.items():
            if value is None:
                df.pop(key)
        return pd.DataFrame(df)

    def _eval(self, X,
              y: np.ndarray,  # 1-d or 2-d array
              repr: List[str],
              y_similar: List[str],
              logfile: str):
        df_output = self.predict(X, y, repr, y_similar=y_similar)
        if logfile is not None and self.write_file:
            df_output.to_csv(logfile, sep='\t', index=False, float_format='%15.10f')
        if y is None:
            return None
        else:
            if y.ndim == 2:
                y_pred = df_output[['predict_%d' % i for i in range(y.shape[1])]].to_numpy()
            else:
                y_pred = df_output['predict']
            return [self._eval_metric(y, y_pred, metric, self.task_type) for metric in self.metrics]

    def _eval_metric(self,
                     y: np.ndarray,  # 1-d or 2-d array.
                     y_pred: np.ndarray,  # 1-d or 2-d array.
                     metric: Metric,
                     task_type: Literal['regression', 'binary', 'multi-class']) -> float:
        if y.ndim == 2 and y_pred.ndim == 2:
            num_tasks = y.shape[1]
            results = []
            for i in range(num_tasks):
                results.append(self._metric_func(y[:, i],
                                                 y_pred[:, i],
                                                 metric,
                                                 task_type))
            return np.nanmean(results)
        else:
            return self._metric_func(y, y_pred, metric, task_type)

    def _metric_func(self,
                     y: np.ndarray,  # 1-d array.
                     y_pred: np.ndarray,  # 1-d array.
                     metric: Metric,
                     task_type: Literal['regression', 'binary', 'multi-class']) -> float:
        # y_pred has nan may happen when train_y are all 1 (or 0).
        if task_type == 'binary' and y_pred.dtype != object and True in np.isnan(y_pred):
            return np.nan
        # y may be unlabeled in some index. Select index of labeled data.
        if y.dtype == float:
            idx = ~np.isnan(y)
            y = y[idx]
            y_pred = y_pred[idx]
        if task_type in ['binary', 'multi-class']:
            if len(set(y)) == 1:
                return np.nan

        if task_type in ['regression', 'binary']:
            return eval_metric_func(y, y_pred=y_pred, metric=metric)
        elif task_type == 'multi-class':
            if metric == 'accuracy':
                return accuracy_score(y, y_pred)
            elif metric == 'precision':
                return precision_score(y, y_pred, average='macro')
            elif metric == 'recall':
                return recall_score(y, y_pred, average='macro')
            elif metric == 'f1_score':
                return f1_score(y, y_pred, average='macro')
            else:
                raise RuntimeError(f'Unsupported metrics {metric} for multi-class classification task.')
        else:
            raise RuntimeError(f'Unsupported task_type {task_type}.')

    def _log(self, info: str):
        if self.verbose:
            print(info)
        else:
            pass
