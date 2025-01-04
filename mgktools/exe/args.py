#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from typing import List, Literal, Tuple, Optional
import os
from mgktools.evaluators.metric import Metric
from mgktools.features_mol.features_generators import FeaturesGenerator


class CommonArgs(Tap):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    smiles_columns: List[str] = None
    """
    Name of the columns containing single SMILES string.
    """
    features_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    features_generators_name: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal["concat", "mean"] = None
    """How to combine features vector for mixtures."""
    targets_columns: List[str] = None
    """
    Name of the columns containing target values. Multi-targets are not implemented yet.
    """
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)

    @property
    def features_generators(self) -> Optional[List[FeaturesGenerator]]:
        if self.features_generators_name is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generators_name]

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if self.features_generators_name is not None and self.features_combination is None:
            self.features_combination = "concat"


class KArgs(Tap):
    graph_kernel_type: Literal["graph", "pre-computed", "no"]
    """The type of kernel to use."""
    graph_hyperparameters: List[str] = None
    """hyperparameters file for graph kernel."""
    features_kernel_type: Literal["dot_product", "rbf"] = None
    """choose dot product kernel or rbf kernel for features."""
    features_hyperparameters: List[float] = None
    """hyperparameters for molecular features."""
    features_hyperparameters_min: float = None
    """hyperparameters for molecular features."""
    features_hyperparameters_max: float = None
    """hyperparameters for molecular features."""
    features_hyperparameters_file: str = None
    """JSON file contains features hyperparameters"""
    single_features_hyperparameter: bool = True
    """Use the same hyperparameter for all features."""

    @property
    def features_hyperparameters_bounds(self):
        if self.features_hyperparameters_min is None or self.features_hyperparameters_max is None:
            if self.features_hyperparameters is None:
                return None
            else:
                return "fixed"
        else:
            return (self.features_hyperparameters_min, self.features_hyperparameters_max)

    @property
    def ignore_features_add(self) -> bool:
        if self.feature_columns is None and \
                self.features_hyperparameters is None and \
                self.features_hyperparameters_file is None:
            return True
        else:
            return False


class KernelArgs(CommonArgs, KArgs):
    def process_args(self) -> None:
        super().process_args()


class ModelArgs(Tap):
    model_type: Literal["gpr", "svc", "svr", "gpc", "gpr-nystrom", "gpr-nle"]
    """The machine learning model to use."""
    alpha: str = None
    """data noise used in gpr."""
    C: str = None
    """C parameter used in Support Vector Machine."""
    ensemble: bool = False
    """use ensemble model."""
    n_estimators: int = 1
    """Ensemble model with n estimators."""
    n_samples_per_model: int = None
    """The number of samples use in each estimator."""
    ensemble_rule: Literal["smallest_uncertainty", "weight_uncertainty",
                           "mean"] = "weight_uncertainty"
    """The rule to combining prediction from estimators."""
    n_local: int = 500
    """The number of samples used in Naive Local Experts."""
    n_core: int = None
    """The number of samples used in Nystrom core set."""


class TrainArgs(KernelArgs, ModelArgs):
    task_type: Literal["regression", "binary", "multi-class"] = None
    """Type of task."""
    cross_validation: Literal["kFold", "leave-one-out", "Monte-Carlo", "no"] = "Monte-Carlo"
    """The way to split data for cross-validation."""
    n_splits: int = None
    """The number of fold for kFold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """Random seed."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    evaluate_train: bool = False
    """If set True, evaluate the model on training set."""
    n_similar: int = None
    """The number of most similar molecules in the training set will be saved."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    atomic_attribution: bool = False
    """Output interpretable results on atomic attribution."""
    molecular_attribution: bool = False
    """Output interpretable results on molecular attribution."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)

    @property
    def C_(self) -> float:
        if self.C is None:
            return None
        elif isinstance(self.C, float):
            return self.C
        elif os.path.exists(self.C):
            return float(open(self.C, "r").read())
        else:
            return float(self.C)

    def kernel_args(self):
        return super()

    def process_args(self) -> None:
        super().process_args()
        if self.task_type == "regression":
            assert self.model_type in ["gpr", "gpr-nystrom", "gpr-nle", "svr"]
            for metric in self.metrics:
                assert metric in ["rmse", "mae", "mse", "r2", "max"]
        elif self.task_type == "binary":
            assert self.model_type in ["gpc", "svc", "gpr"]
            for metric in self.metrics:
                assert metric in ["roc_auc", "accuracy", "precision", "recall", "f1_score", "mcc"]
        elif self.task_type == "multi-class":
            raise NotImplementedError("Multi-class classification is not implemented yet.")

        if self.cross_validation == "leave-one-out":
            assert self.num_folds == 1
            assert self.model_type == "gpr"

        if self.model_type in ["gpr", "gpr-nystrom"]:
            assert self.alpha is not None

        if self.model_type == "svc":
            assert self.C is not None

        if self.ensemble:
            assert self.n_samples_per_model is not None

        if self.atomic_attribution:
            assert self.graph_kernel_type == "graph", "Set graph_kernel_type to graph for interpretability"
            assert self.model_type == "gpr", "Set model_type to gpr for interpretability"
            assert self.ensemble is False


class GradientOptArgs(TrainArgs):
    loss: Literal["loocv", "likelihood"] = "loocv"
    """The target loss function to minimize or maximize."""
    optimizer: str = None
    """Optimizer implemented in scipy.optimize.minimize are valid: L-BFGS-B, SLSQP, Nelder-Mead, Newton-CG, etc."""

    def process_args(self) -> None:
        assert self.model_type == "gpr"


class OptunaArgs(TrainArgs):
    num_iters: int = 100
    """Number of hyperparameter choices to try."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    d_alpha: float = None
    """The step size of alpha to be optimized."""
    C_bounds: Tuple[float, float] = None #  (1e-3, 1e3)
    """Bounds of C used in SVC."""
    d_C: float = None
    """The step size of C to be optimized."""
    num_splits: int = 1
    """split the dataset randomly into no. subsets to reduce computational costs."""

    def process_args(self) -> None:
        super().process_args()


class OptunaMultiDatasetArgs(KArgs):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_paths: List[str]
    """The Path of input data CSV files."""
    smiles_columns: str = None
    """
    Name of the columns containing single SMILES string.
    E.g.: "smiles;smiles;smiles1,smiles2"
    """
    features_columns: str = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    targets_columns: str = None
    """
    Name of the columns containing target values.
    """
    features_generators_name: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal["concat", "mean"] = None
    """How to combine features vector for mixtures."""
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    tasks_type: List[Literal["regression", "binary", "multi-class"]]
    """
    Type of task.
    """
    metrics: List[Metric]
    """taget metrics to be optimized."""
    num_iters: int = 100
    """Number of hyperparameter choices to try."""
    alpha: str = None
    """data noise used in gpr."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    d_alpha: float = None
    """The step size of alpha to be optimized."""
    seed: int = 0
    """Random seed."""

    @property
    def features_generators(self) -> Optional[List[FeaturesGenerator]]:
        if self.features_generators_name is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generators_name]

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        none_list = [None] * len(self.data_paths)
        self.smiles_columns_ = [i.split(",") for i in self.smiles_columns.split(";")]
        self.features_columns_ = [None if i == '' else i.split(",") for i in self.features_columns.split(";")] if self.features_columns is not None else none_list
        self.targets_columns_ = [i.split(",") for i in self.targets_columns.split(";")]


class EmbeddingArgs(KernelArgs):
    embedding_algorithm: Literal["tSNE", "kPCA"] = "tSNE"
    """Algorithm for data embedding."""
    n_components: int = 2
    """Dimension of the embedded space."""
    perplexity: float = 30.0
    """
    The perplexity is related to the number of nearest neighbors that
    is used in other manifold learning algorithms. Larger datasets
    usually require a larger perplexity. Consider selecting a value
    different results.
    """
    n_iter: int = 1000
    """Maximum number of iterations for the optimization. Should be at least 250."""

    def process_args(self) -> None:
        super().process_args()
