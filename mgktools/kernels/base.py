#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Callable, Any
import os
import pickle
import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, RBF
from hyperopt import hp
from graphdot.microkernel import (
    Additive,
    Constant as Const,
    TensorProduct,
    SquareExponential as sExp,
    KroneckerDelta as kDelta,
    Convolution as kConv,
    Normalize,
)
from graphdot.microprobability import (
    Additive as Additive_p,
    Constant,
    UniformProbability,
    AssignProbability,
)


class MicroKernel:
    def __init__(
        self,
        idx: int,
        name: str,
        kernel_type: str,
        value,
        bounds: Tuple[float, float] = None,
        delta: float = None,
        available_values: List = None,
    ):
        """The smallest unit to build a complex kernel. Each microkernel contains only one hyperparameter.

        Parameters
        ----------
        idx : int
            The index of the kernel config this microkernel belongs to. This is important to separate the same microkernel in different kernel configs.
        name : str
            The name of the microkernel.
        kernel_type : str
            The function type of the microkernel.
        value :
            The hyperparameter of the microkernel.
        bounds : Tuple[float, float], optional
            A tuple with the lower and upper bounds for the hyperparameters optimization, by default None ("fixed").
        delta : float, optional
            The stepsize for the hyperparameters optimization, by default None.
        available_values : List, optional
            A list with the available values for the hyperparameters optimization, by default None.
        """
        self.name = name
        self.kernel_type = kernel_type
        self.unique_name = f"{idx}:{name}:{kernel_type}"
        self.value = value
        self.bounds = bounds
        self.delta = delta
        self.available_values = available_values
        if self.bounds is None:
            assert self.available_values is not None
            assert self.value in self.available_values
        else:
            assert self.available_values is None
            if self.bounds != "fixed":
                if isinstance(self.bounds, list):
                    self.bounds = tuple(self.bounds)
                assert (
                    len(self.bounds) == 2
                ), "Bounds must be a tuple with two elements."
                assert (
                    self.bounds[0] < self.bounds[1]
                ), "Lower bound must be less than upper bound."
                assert isinstance(self.value, float)

    def get_kernel(self) -> Callable:
        bounds = self.bounds or "fixed"
        if self.kernel_type == "rbf":
            return RBF(length_scale=self.value, length_scale_bounds=bounds)
        elif self.kernel_type == "dot_product":
            return DotProduct(sigma_0=self.value, sigma_0_bounds=bounds)
        elif self.kernel_type == "Const":
            return Const(self.value, bounds)
        elif self.kernel_type == "kDelta":
            return kDelta(self.value, bounds)
        elif self.kernel_type == "kConv":
            return kConv(kDelta(self.value, bounds))
        elif self.kernel_type == "sExp":
            return sExp(self.value, length_scale_bounds=bounds)
        elif self.kernel_type == "Uniform_p":
            return UniformProbability(self.value, bounds)
        elif self.kernel_type == "Const_p":
            return Constant(self.value, bounds)
        elif self.kernel_type == "Assign_p":
            return AssignProbability(self.value, bounds)
        elif self.kernel_type == "Normalization":
            from mgktools.kernels.normalization import Norm, NormalizationMolSize
            assert self.name == self.kernel_type
            if self.value == True:
                return Norm
            elif self.value == False:
                return lambda x: x
            else:
                assert isinstance(self.value, float)
                return lambda x: NormalizationMolSize(
                    kernel=x, s=self.value, s_bounds=bounds
                )
        elif self.kernel_type in ["a_type", "b_type", "p_type"]:
            if self.value == "Tensorproduct":
                return TensorProduct
            elif self.value == "Additive":
                return lambda **x: Normalize(Additive(**x))
            elif self.value == "Additive_p":
                return Additive_p
            else:
                raise ValueError(
                    "For kernel type (%s), the value (%s) is not supported."
                    % (self.kernel_type, self.value)
                )
        else:
            raise ValueError("Invalid kernel type %s." % self.kernel_type)

    def get_space(self) -> Dict:
        if self.available_values is not None:
            return {
                self.unique_name: hp.choice(self.unique_name, self.available_values)
            }
        elif self.bounds == "fixed":
            return {}
        else:
            if self.delta is None:
                return {
                    self.unique_name: hp.uniform(
                        self.unique_name, low=self.bounds[0], high=self.bounds[1]
                    )
                }
            else:
                return {
                    self.unique_name: hp.quniform(
                        self.unique_name,
                        low=self.bounds[0],
                        high=self.bounds[1],
                        q=self.delta,
                    )
                }

    def update_from_space(self, space: Dict[str, Any]):
        if self.unique_name in space:
            assert self.bounds != "fixed"
            self.value = space[self.unique_name]
        else:
            assert self.bounds == "fixed", f"{self.unique_name};{self.bounds};{space}"

    def get_trial(self, trial) -> Dict:
        if self.available_values is not None:
            return {
                self.unique_name: trial.suggest_categorical(
                    name=self.unique_name, choices=self.available_values
                )
            }
        elif self.bounds == "fixed":
            return {}
        else:
            if self.delta is None:
                return {
                    self.unique_name: trial.suggest_float(
                        name=self.unique_name, low=self.bounds[0], high=self.bounds[1]
                    )
                }
            else:
                return {
                    self.unique_name: trial.suggest_float(
                        name=self.unique_name,
                        low=self.bounds[0],
                        high=self.bounds[1],
                        step=self.delta,
                    )
                }

    def update_from_trial(self, trial: Dict[str, Any]):
        self.update_from_space(trial)

    def update_from_theta(self, values: List):
        assert self.available_values is None
        if self.bounds != "fixed":
            assert self.bounds[0] * 0.99 < values[0] < self.bounds[1] * 1.01
            if values[0] < self.bounds[0]:
                values[0] = self.bounds[0]
            elif values[0] > self.bounds[1]:
                values[0] = self.bounds[1]
            self.value = values.pop(0)

    def get_microdict(self) -> Dict:
        values = [self.value, self.bounds, self.delta, self.available_values]
        return {f"{self.kernel_type}": values}

    def update_hyperdict(self, hyperdict: Dict):
        if self.name == self.kernel_type:
            assert self.kernel_type not in hyperdict
            hyperdict.update(self.get_microdict())
        else:
            if self.name not in hyperdict:
                hyperdict.update({self.name: {}})
            assert self.kernel_type not in hyperdict[self.name]
            hyperdict[self.name].update(self.get_microdict())

    @classmethod
    def from_microdict(cls, idx: int, name: str, microdict: Dict[str, List]):
        assert len(microdict) == 1
        for kernel_type, values in microdict.items():
            return cls(
                idx=idx,
                name=name,
                kernel_type=kernel_type,
                value=values[0],
                bounds=values[1],
                delta=values[2],
                available_values=values[3],
            )


class ABCKernelConfig(ABC):
    @abstractmethod
    def update_kernel(self):
        pass

    @abstractmethod
    def get_space(self) -> Dict:
        pass

    @abstractmethod
    def update_from_space(self, space: Dict[str, Any]):
        pass
    
    @abstractmethod
    def get_trial(self, trial) -> Dict:
        pass

    @abstractmethod
    def update_from_trial(self, trial: Dict[str, Any]):
        pass

    @abstractmethod
    def update_from_theta(self):
        pass


class BaseKernelConfig(ABCKernelConfig):
    def __init__(
        self,
        kernel_type: str,
        kernel_hyperparameters: list,
        kernel_hyperparameters_bounds: list,
    ):
        self.kernel_type = kernel_type
        self.kernel_hyperparameters = kernel_hyperparameters
        self.kernel_hyperparameters_bounds = kernel_hyperparameters_bounds
        self.kernel = self._get_kernel()

    def get_kernel_dict(self, X: np.ndarray, X_labels: List[str]) -> Dict:
        """Calculate a kernel matrix and save in a dictionary.

        Parameters
        ----------
        X : np.ndarray
            A numpy array with the data to compute the kernel matrix.
        X_labels : List[str]
            A list with the unique text labels for the data.

        Returns
        -------
        dict:
            A dictionary with the kernel matrix and the data labels.
        """
        K = self.kernel(X)
        return {"X": X_labels, "K": K, "theta": self.kernel.theta}

    def save_kernel_matrix(self, path: str, X: np.ndarray, X_labels: List[str]):
        """Save kernel.pkl file that used for preCalc kernels."""
        kernel_dict = self.get_kernel_dict(X, X_labels)
        kernel_pkl = os.path.join(path, "kernel.pkl")
        pickle.dump(kernel_dict, open(kernel_pkl, "wb"), protocol=4)

    @staticmethod
    def combine_dicts(dicts: List[Dict]) -> Dict:
        """Combine a list of dictionaries into one.

        Parameters
        ----------
        dicts : List[Dict]
            A list with dictionaries to be combined.

        Returns
        -------
        Dict:
            A dictionary with the combined dictionaries.
        """
        combined_dict = {}
        n = 0
        for d in dicts:
            combined_dict.update(d)
            n += len(d)
        assert n == len(combined_dict)
        return combined_dict
