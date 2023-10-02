#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from collections import defaultdict
from random import Random
from typing import Dict, List, Set, Tuple, Union
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import math
from logging import Logger
import numpy as np
from .data import Dataset


def generate_scaffold(mol: Union[str, Chem.Mol],
                      include_chirality: bool = False) -> str:
    """ Computes the Bemis-Murcko scaffold for a SMILES string.

    Parameters
    ----------
    mol: A SMILES string or an RDKit molecule.
    include_chirality: bool
        Whether to include chirality in the computed scaffold..

    Returns
    -------
    The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """ Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    Parameters
    ----------
    mols: A list of SMILES strings or RDKit molecules.
    use_indices:
        Whether to map to the SMILES's index in :code:`mols` rather than mapping to the smiles string itself.
        This is necessary if there are duplicate smiles.

    Returns
    -------
    A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)
    return scaffolds


def get_split_sizes(n_samples: int,
                    split_ratio: List[float]):
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"Split split_ratio do not sum to 1. Received splits: {split_ratio}")
    if any([size < 0 for size in split_ratio]):
        raise ValueError(f"Split split_ratio must be non-negative. Received splits: {split_ratio}")
    acc_ratio = np.cumsum([0.] + split_ratio)
    split_sizes = [math.ceil(acc_ratio[i + 1] * n_samples) - math.ceil(acc_ratio[i] * n_samples)
                   for i in range(len(acc_ratio) - 1)]
    if sum(split_sizes) == n_samples + 1:
        split_sizes[-1] -= 1
    assert sum(split_sizes) == n_samples
    return split_sizes


def data_split_index(n_samples: int,
                     mols: List[Union[str, Chem.Mol]] = None,
                     targets: List = None,
                     split_type: Literal['random', 'scaffold_order', 'scaffold_random', 'init_al', 'stratified',
                                         'n_heavy'] = 'random',
                     sizes: List[float] = [0.8, 0.2],
                     n_samples_per_class: int = None,
                     n_heavy_cutoff: int = None,
                     seed: int = 0,
                     logger: Logger = None):
    if logger is not None:
        info = logger.info
        warn = logger.warning
    else:
        info = print
        warn = print

    random = Random(seed)
    np.random.seed(seed)
    split_index = [[] for size in sizes]
    if split_type == 'random':
        indices = list(range(n_samples))
        random.shuffle(indices)
        index_size = get_split_sizes(n_samples, split_ratio=sizes)
        end = 0
        for i, size in enumerate(index_size):
            start = end
            end = start + size
            split_index[i] = indices[start:end]
    elif split_type == 'stratified':
        class_list = list(np.unique(targets))
        assert len(class_list) > 1
        num_class = len(class_list)
        if num_class > 10:
            warn('You are splitting a classification dataset with more than 10 classes.')
        class_index = [[] for c in class_list]
        for i, y in enumerate(targets):
            class_index[class_list.index(y)].append(i)
        for c_index in class_index:
            for i, idx in enumerate(data_split_index(n_samples=len(c_index),
                                                     split_type='random',
                                                     sizes=sizes,
                                                     seed=seed)):
                split_index[i] += list(np.array(c_index)[idx])
    elif split_type in ['scaffold_random', 'scaffold_order']:
        index_size = get_split_sizes(n_samples, split_ratio=sizes)
        if mols[0].__class__ == 'str':
            mols = [Chem.MolFromSmiles(s) for s in mols]
        scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

        scaffold_count = [0 for size in sizes]
        index = list(range(len(sizes)))
        for index_set in index_sets:
            if split_type == 'scaffold_random':
                random.shuffle(index)
            for i in index:
                s_index = split_index[i]
                if len(s_index) + len(index_set) <= index_size[i]:
                    s_index += index_set
                    scaffold_count[i] += 1
                    break
            else:
                split_index[0] += index_set
        info(f'Total scaffolds = {len(scaffold_to_indices):,} | ')
        for i, count in enumerate(scaffold_count):
            info(f'split {i} scaffolds = {count:,} | ')
    elif split_type == 'n_heavy':
        assert n_heavy_cutoff is not None
        if mols[0].__class__ == 'str':
            mols = [Chem.MolFromSmiles(s) for s in mols]
        split_index = [[], []]
        for i, mol in enumerate(mols):
            if mol.GetNumAtoms() < n_heavy_cutoff:
                split_index[0].append(i)
            else:
                split_index[1].append(i)
    elif split_type == 'init_al':
        class_list = np.unique(targets)
        assert len(class_list) > 1
        num_class = len(class_list)
        if num_class > 10:
            warn('You are splitting a classification dataset with more than 10 classes.')
        if n_samples_per_class is None:
            assert len(sizes) == 2
            n_samples_per_class = int(sizes[0] * n_samples / num_class)
            assert n_samples_per_class > 0

        for c in class_list:
            index = []
            for i, t in enumerate(targets):
                if t == c:
                    index.append(i)
            split_index[0].extend(np.random.choice(index, n_samples_per_class, replace=False).tolist())
        for i in range(n_samples):
            if i not in split_index[0]:
                split_index[1].append(i)
    else:
        raise ValueError(f'split_type "{split_type}" not supported.')
    assert sum([len(i) for i in split_index]) == n_samples
    return split_index


def dataset_split(dataset,
                  split_type: Literal['random', 'scaffold_order', 'scaffold_random', 'init_al', 'stratified',
                                      'n_heavy'] = None,
                  sizes: List[float] = [0.8, 0.2],
                  n_heavy_cutoff: int = 15,
                  seed: int = 0) -> List:
    """ Split the data set into two data sets: training set and test set.

    Parameters
    ----------
    split_type: The algorithm used for data splitting.
    sizes: [float, float].
        If split_type == 'random' or 'scaffold_balanced'.
        sizes are the percentages of molecules in training and test sets.
    n_heavy_cutoff: int
        If split_type == 'n_heavy'.
        training set contains molecules with heavy atoms < n_heavy.
        test set contains molecules with heavy atoms >= n_heavy.
    seed

    Returns
    -------
    [Dataset, Dataset]
    """
    data = []
    if split_type in ['random', 'stratified']:
        mols = None
    else:
        mols = []
        for m in dataset.mols:
            assert len(m) == 1
            mols.append(m[0])
    split_index = data_split_index(n_samples=len(dataset),
                                   mols=mols,
                                   targets=dataset.y,
                                   split_type=split_type,
                                   sizes=sizes,
                                   n_samples_per_class=None,
                                   n_heavy_cutoff=n_heavy_cutoff,
                                   seed=seed,
                                   logger=None)
    for s_index in split_index:
        dataset_ = dataset.copy()
        dataset_.data = [dataset.data[i] for i in s_index]
        data.append(dataset_)
    return data
