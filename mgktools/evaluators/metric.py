#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
Metric = Literal['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc',
                 'rmse', 'mae', 'mse', 'r2', 'max']


def p2v(y: List[float], y_pred: List[float]):
    avail_values = list(set(y))
    y_pred = [min(avail_values, key=lambda x: abs(x - v)) for v in y_pred]
    return y_pred


def eval_metric_func(y: List[float], y_pred: List[float], metric: Metric) -> float:
    if metric == 'roc-auc':
        return roc_auc_score(y, y_pred)
    elif metric == 'accuracy':
        y_pred = p2v(y, y_pred)
        return accuracy_score(y, y_pred)
    elif metric == 'precision':
        y_pred = p2v(y, y_pred)
        return precision_score(y, y_pred)
    elif metric == 'recall':
        y_pred = p2v(y, y_pred)
        return recall_score(y, y_pred)
    elif metric == 'f1_score':
        y_pred = p2v(y, y_pred)
        return f1_score(y, y_pred)
    elif metric == 'mcc':
        y_pred = p2v(y, y_pred)
        return matthews_corrcoef(y, y_pred)
    elif metric == 'r2':
        return r2_score(y, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y, y_pred)
    elif metric == 'rmse':
        return np.sqrt(eval_metric_func(y, y_pred, 'mse'))
    elif metric == 'max':
        return np.max(abs(y - y_pred))
    else:
        raise RuntimeError(f'Unsupported metrics {metric}')
