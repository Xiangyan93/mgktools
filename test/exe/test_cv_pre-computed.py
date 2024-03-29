# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os

CWD = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append("%s/.." % CWD)
from mgktools.hyperparameters import (
    additive,
    additive_pnorm,
    additive_msnorm,
    additive_norm,
    product,
    product_pnorm,
    product_msnorm,
    product_norm,
)
from mgktools.exe.run import mgk_kernel_calc, mgk_model_evaluate


@pytest.mark.parametrize(
    "dataset",
    [
        ("freesolv", ["smiles"], ["freesolv"]),
    ],
)
@pytest.mark.parametrize(
    "testset",
    [
        ("random", "10"),
        ("scaffold_order", "10"),
    ],
)
@pytest.mark.parametrize(
    "graph_hyperparameters",
    [
        additive,
        additive_pnorm,
        additive_msnorm,
        additive_norm,
        product,
        product_pnorm,
        product_msnorm,
        product_norm,
    ],
)
def test_cv_PreComputed_PureGraph_Regression(dataset, testset, graph_hyperparameters):
    task = "regression"
    model = "gpr"
    dataset, pure_columns, target_columns = dataset
    save_dir = "%s/data/_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
    )
    split, num_folds = testset
    # kernel computation
    assert not os.path.exists("%s/kernel.pkl" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--graph_hyperparameters",
        graph_hyperparameters,
    ]
    mgk_kernel_calc(arguments)
    assert os.path.exists("%s/kernel.pkl" % save_dir)
    # cross validation
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "pre-computed",
        "--task_type",
        task,
        "--model_type",
        model,
        "--split_type",
        split,
        "--split_sizes",
        "0.8",
        "0.2",
        "--alpha",
        "0.01",
        "--metric",
        "rmse",
        "--extra_metrics",
        "r2",
        "mae",
        "--num_folds",
        num_folds,
    ]
    mgk_model_evaluate(arguments)
    os.remove("%s/kernel.pkl" % save_dir)


@pytest.mark.parametrize(
    "dataset",
    [
        ("st", ["smiles"], ["st"], ["T"]),
    ],
)
@pytest.mark.parametrize("group_reading", [True, False])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("model", ["gpr"])
@pytest.mark.parametrize("graph_hyperparameters", [additive])
def test_cv_PreComputed_PureGraph_FeaturesAdd_Regression(
    dataset, group_reading, features_scaling, model, graph_hyperparameters
):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = "%s/data/_%s_%s_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
        group_reading,
        features_scaling,
    )
    task = "regression"
    split, num_folds = "random", "10"
    metric = "rmse"
    # kernel computation
    assert not os.path.exists("%s/kernel.pkl" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--graph_hyperparameters",
        graph_hyperparameters,
        "--features_kernel_type",
        "rbf",
        "--features_hyperparameters",
        "100.0",
    ]
    mgk_kernel_calc(arguments)
    assert os.path.exists("%s/kernel.pkl" % save_dir)
    # cross validation
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "pre-computed",
        "--task_type",
        task,
        "--model_type",
        model,
        "--split_sizes",
        "0.8",
        "0.2",
        "--split_type",
        split,
        "--metric",
        metric,
        "--num_folds",
        num_folds,
        "--alpha",
        "0.01",
        "--features_kernel_type",
        "rbf",
        "--features_hyperparameters",
        "100.0",
    ]
    mgk_model_evaluate(arguments)
    os.remove("%s/kernel.pkl" % save_dir)


@pytest.mark.parametrize(
    "dataset",
    [
        ("freesolv", ["smiles"], ["freesolv"]),
        ("np", ["smiles1", "smiles2"], ["np"]),
    ],
)
@pytest.mark.parametrize(
    "features_generator",
    [["rdkit_2d_normalized"], ["morgan"], ["rdkit_2d", "morgan_count"]],
)
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("model", ["gpr"])
@pytest.mark.parametrize("graph_hyperparameters", [additive_msnorm])
def test_cv_PreComputed_PureGraph_FeaturesMol_Regression(
    dataset, features_generator, features_scaling, model, graph_hyperparameters
):
    dataset, pure_columns, target_columns = dataset
    save_dir = "%s/data/_%s_%s_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
        ",".join(features_generator),
        features_scaling,
    )
    task = "regression"
    split, num_folds = "random", "10"
    metric = "rmse"
    # kernel computation
    assert not os.path.exists("%s/kernel.pkl" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--features_kernel_type",
        "rbf",
        "--features_hyperparameters",
        "10.0",
    ]
    arguments += ["--graph_hyperparameters"] + [graph_hyperparameters] * len(
        pure_columns
    )
    mgk_kernel_calc(arguments)
    assert os.path.exists("%s/kernel.pkl" % save_dir)
    # cross validation
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "pre-computed",
        "--task_type",
        task,
        "--model_type",
        model,
        "--split_sizes",
        "0.8",
        "0.2",
        "--split_type",
        split,
        "--metric",
        metric,
        "--num_folds",
        num_folds,
        "--alpha",
        "0.01",
        "--features_kernel_type",
        "rbf",
        "--features_hyperparameters",
        "10.0",
    ]
    mgk_model_evaluate(arguments)
    os.remove("%s/kernel.pkl" % save_dir)


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize("model", ["gpr"])
@pytest.mark.parametrize("graph_hyperparameters", [additive])
def test_cv_PreComputed_PureGraph_FeaturesAddMol_Regression(dataset, group_reading, features_generator, features_scaling, model, graph_hyperparameters):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                               group_reading, ','.join(features_generator), features_scaling)
    task = "regression"
    split, num_folds = "random", "10"
    metric = "rmse"
    # kernel computation
    assert not os.path.exists("%s/kernel.pkl" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--graph_hyperparameters",
        graph_hyperparameters,
        "--features_kernel_type",
        "rbf",
        "--features_hyperparameters",
        "10.0",
    ]
    mgk_kernel_calc(arguments)
    assert os.path.exists("%s/kernel.pkl" % save_dir)
    # cross validation
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "pre-computed",
        "--task_type",
        task,
        "--model_type",
        model,
        "--split_sizes",
        "0.8",
        "0.2",
        "--split_type",
        split,
        "--metric",
        metric,
        "--num_folds",
        num_folds,
        "--alpha",
        "0.01",
        "--features_kernel_type",
        "rbf",
        "--features_hyperparameters",
        "100.0",
    ]
    mgk_model_evaluate(arguments)
    os.remove("%s/kernel.pkl" % save_dir)
    

@pytest.mark.parametrize(
    "dataset",
    [
        ("bace", ["smiles"], ["bace"]),
        ("np", ["smiles1", "smiles2"], ["np"]),
    ],
)
@pytest.mark.parametrize("model", ["gpr", "gpc", "svc"])
@pytest.mark.parametrize(
    "testset",
    [
        ("random", "10"),
    ],
)
@pytest.mark.parametrize(
    "metric", ["roc-auc", "accuracy", "precision", "recall", "f1_score", "mcc"]
)
@pytest.mark.parametrize("graph_hyperparameters", [additive_msnorm])
def test_cv_PreComputed_PureGraph_Binary(
    dataset, model, testset, metric, graph_hyperparameters
):
    task = "binary"
    dataset, pure_columns, target_columns = dataset
    save_dir = "%s/data/_%s_%s_%s" % (
        CWD,
        dataset,
        ",".join(pure_columns),
        ",".join(target_columns),
    )
    split, num_folds = testset
    # kernel computation
    assert not os.path.exists("%s/kernel.pkl" % save_dir)
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "graph",
        "--graph_hyperparameters",
    ] + [graph_hyperparameters] * len(pure_columns)
    mgk_kernel_calc(arguments)
    assert os.path.exists("%s/kernel.pkl" % save_dir)
    # cross validation
    arguments = [
        "--save_dir",
        "%s" % save_dir,
        "--graph_kernel_type",
        "pre-computed",
        "--task_type",
        task,
        "--model_type",
        model,
        "--split_type",
        split,
        "--split_sizes",
        "0.8",
        "0.2",
        "--metric",
        metric,
        "--num_folds",
        num_folds,
    ]
    if model == "gpr":
        arguments += ["--alpha", "0.01"]
    elif model == "svc":
        arguments += ["--C", "1.0"]
    mgk_model_evaluate(arguments)
    os.remove("%s/kernel.pkl" % save_dir)
