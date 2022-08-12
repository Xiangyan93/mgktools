#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import rdkit.Chem.AllChem as Chem
from ..hyperparameters import additive_pnorm
from ..graph.hashgraph import HashGraph
from ..data import Dataset
from ..kernels.utils import get_kernel_config
from ..kernels.GraphKernel import GraphKernelConfig
from .gpr import InterpretableGaussianProcessRegressor as GPR


def get_node_graphs(mol: Chem.Mol) -> List[HashGraph]:
    """This function returns a list of graphs of the same molecule. In each graph, the starting probability is non-zero
    only for one atom, and the prediction on this graph is exactly the contribution of the atom to the molecule.

    Parameters
    ----------
    mol: RDKit mol object.

    Returns
    -------
    A list of HashGraphs.
    """
    graphs = []
    for i, atom in enumerate(mol.GetAtoms()):
        graph = HashGraph.from_rdkit(mol)
        for j in range(mol.GetNumAtoms()):
            if i != j:
                graph.nodes['Concentration'][j] = 0.
                graph.nodes['Concentration_norm'][j] = 0.
        graphs.append(graph)
    return graphs


def interpret_training_mols(smiles_to_be_interpret: str,
                            smiles_train: List[str],
                            targets_train: List[float],
                            alpha: float = 0.01,
                            n_mol: int = 10,
                            output_order: Literal['sort_by_value', 'sort_by_percentage_contribution'] = 'sort_by_value',
                            mgk_hyperparameters_file: str = additive_pnorm,
                            n_jobs: int = 1):
    """Interpret molecular property prediction by the sum of the contribution of the molecules in the training set.

    Parameters
    ----------
    smiles_to_be_interpret: string
        SMILES string of the molecule to be predicted and interpreted.
    smiles_train: list of string
        SMILES of training set.
    targets_train: list of float
        target property of traning set.
    alpha: float
        data noise of Gaussian process regression.
    n_mol: int
        The number of molecules show in the interpretation.
    output_order: bool
        If 'sort_by_value', the interpretation will be ranked by the value contribution.
        If 'sort_by_percentage_contribution', the interpretation will be ranked by the percentage contribution.
    mgk_hyperparameters_file: str
        hyperparameters for marginalized graph kernel.
    n_jobs: int
        number of processes when transforming smiles into graphs.

    Returns
    -------
    predicted value, predicted uncertainty, interpretation dataframe.
    """
    # graph_to_be_interpret = HashGraph.from_smiles(smiles_to_be_interpret)
    df = pd.DataFrame({'smiles': smiles_train, 'target': targets_train})
    train = Dataset.from_df(df, pure_columns=['smiles'], target_columns=['target'], n_jobs=n_jobs)
    train.graph_kernel_type = 'graph'
    df = pd.DataFrame({'smiles': [smiles_to_be_interpret], 'target': [0.]})
    test = Dataset.from_df(df, pure_columns=['smiles'], target_columns=['target'], n_jobs=n_jobs)
    test.graph_kernel_type = 'graph'
    full = train.copy()
    full.data = train.data + test.data
    full.unify_datatype()
    kernel_config = get_kernel_config(
        train,
        graph_kernel_type='graph',
        # arguments for marginalized graph kernel
        mgk_hyperparameters_files=[mgk_hyperparameters_file],
    )
    kernel = kernel_config.kernel
    gpr = GPR(kernel=kernel, alpha=alpha, normalize_y=False).fit(train.X, train.y)
    y_pred, y_std = gpr.predict(test.X, return_std=True)
    c_percentage, c_y = gpr.predict_interpretable(test.X)
    if output_order == 'sort_by_value':
        idx = np.argsort(-np.fabs(c_y))[:, :min(n_mol, c_y.shape[1])]
    else:
        idx = np.argsort(-np.fabs(c_percentage))[:, :min(n_mol, c_percentage.shape[1])]
    df_out = pd.DataFrame({'smiles_train': np.asarray(smiles_train)[idx[0]],
                           'contribution_percentage': c_percentage[0][idx[0]],
                           'contribution_value': c_y[0][idx[0]]})
    return y_pred[0], y_std[0], df_out


def interpret_atoms(smiles_to_be_interpret: str,
                    smiles_train: List[str],
                    targets_train: List[float],
                    alpha: float = 0.01,
                    mgk_hyperparameters_file: str = additive_pnorm,
                    n_jobs: int = 1):
    """Interpret molecular property prediction by the sum of the contribution of the atoms of the molecule to be
    predicted.

    Parameters
    ----------
    smiles_to_be_interpret: string
        SMILES string of the molecule to be predicted and interpreted.
    smiles_train: list of string
        SMILES of training set.
    targets_train: list of float
        target property of traning set.
    alpha: float
        data noise of Gaussian process regression.
    mgk_hyperparameters_file: str
        hyperparameters for marginalized graph kernel.
    n_jobs: int
        number of processes when transforming smiles into graphs.
    Returns
    -------
    predicted value, predicted uncertainty, RDKit Mol with interpretation for all atoms.
    """
    graph_to_be_interpret = HashGraph.from_smiles(smiles_to_be_interpret)
    mol = Chem.MolFromSmiles(smiles_to_be_interpret)
    node_graphs = get_node_graphs(mol)
    graphs_train = [HashGraph.from_smiles(s) for s in smiles_train]
    HashGraph.unify_datatype(graphs_train + node_graphs + [graph_to_be_interpret], inplace=True)
    graph_hyperparameters = [json.load(open(mgk_hyperparameters_file))]
    kernel = GraphKernelConfig(N_MGK=1,
                               graph_hyperparameters=graph_hyperparameters).kernel
    # normalize_y must be false.
    gpr = GPR(kernel=kernel, alpha=alpha, normalize_y=False).fit(graphs_train, targets_train)
    y_pred, y_std = gpr.predict([graph_to_be_interpret], return_std=True)
    y_nodes = gpr.predict(node_graphs)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('atomNote', '%.6f' % y_nodes[i])
    return y_pred[0], y_std[0], mol


def interpret_substructures(smarts: str,
                            mols: List[Chem.Mol]):
    """Find all occurrence of input substructure (smarts) and calculate their interpreted contributions.

    Parameters
    ----------
    smarts: str
        SMARTS string of the substructure
    mols: list of RDKit Mol object.
        The mols must have been interpreted using 'get_interpretable_mols'.

    Returns
    -------
    list of interpreted contributions of the substructure.
    """
    smarts = Chem.MolFromSmarts(smarts)
    contributed_values = []
    for mol in mols:
        for idx in mol.GetSubstructMatches(smarts, useChirality=True):
            values = [float(mol.GetAtomWithIdx(i).GetProp('atomNote')) for i in idx]
            contributed_values.append(sum(values))
    return contributed_values


def get_interpreted_mols(smiles_train: List[str],
                         targets_train: List[float],
                         smiles_to_be_interpret: List[str] = None,
                         alpha: float = 0.01,
                         mgk_hyperparameters_file: str = additive_pnorm,
                         return_mols_only: bool = True):
    """Interpret all molecules, and save the interpretation in the RDKit Mol object.

    Parameters
    ----------
    smiles_train: list of string
        SMILES strings of molecules of training set.
    targets_train: list of float
        the target property.
    smiles_to_be_interpret: list of string
        SMILES strings of molecules to be interpreted.
    alpha: float
        data noise of Gaussian process regression.
    mgk_hyperparameters_file: dict
        hyperparameters for marginalized graph kernel.
    return_mols_only: bool
        if True, return the RDKit mols

    Returns
    -------
    list of RDKit Mol object, the interpretation of all atoms is saved inside.
    """
    mols_train = [Chem.MolFromSmiles(s) for s in smiles_train]
    graphs = [HashGraph.from_rdkit(mol) for mol in mols_train]
    HashGraph.unify_datatype(graphs, inplace=True)
    graph_hyperparameters = [json.load(open(mgk_hyperparameters_file))]
    kernel = GraphKernelConfig(N_MGK=1,
                               graph_hyperparameters=graph_hyperparameters).kernel
    gpr = GPR(kernel=kernel, alpha=alpha, normalize_y=False).fit(graphs, targets_train)
    mols_to_be_interpret = [Chem.MolFromSmiles(s) for s in smiles_to_be_interpret]
    for mol in tqdm(mols_to_be_interpret, total=len(mols_to_be_interpret)):
        node_graphs = get_node_graphs(mol)
        HashGraph.unify_datatype(graphs + node_graphs, inplace=True)
        y_nodes = gpr.predict(node_graphs)
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('atomNote', '%.6f' % y_nodes[i])

    if return_mols_only:
        return mols_to_be_interpret
    else:
        graphs_to_be_interpret = [HashGraph.from_rdkit(mol) for mol in mols_to_be_interpret]
        HashGraph.unify_datatype(graphs + graphs_to_be_interpret, inplace=True)
        y_pred, y_std = gpr.predict(graphs_to_be_interpret, return_std=True)
        return y_pred, y_std, mols_to_be_interpret
