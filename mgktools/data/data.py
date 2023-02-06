#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple, Callable
import copy
import ase
import os
import pickle
import json
import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem
from rxntools.reaction import ChemicalReaction
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import networkx as nx
from graphdot.graph._from_networkx import _from_networkx
from ..features_mol import FeaturesGenerator
from ..graph.hashgraph import HashGraph
from .public import QM7, QM9

# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}


def remove_none(X: List):
    X_ = []
    for x in X:
        if x is not None and len(x) != 0:
            X_.append(x)
    if len(X_) == 0:
        return None
    else:
        return X_


def concatenate(X: List, axis: int = 0, dtype=None):
    X_ = remove_none(X)
    if X_:
        return np.concatenate(X_, axis=axis, dtype=dtype)
    else:
        return None


class MolecularGraph2D:
    """
    SingleMolDatapoint: 2D graph of a single molecule.
    """

    def __init__(self, smiles: str, features_mol: np.ndarray = None):
        self.smiles = smiles
        self.features_mol = features_mol
        self.graph = HashGraph.from_rdkit(self.mol, self.smiles)

    def __repr__(self) -> str:
        return self.smiles

    @property
    def mol(self):
        if self.smiles in SMILES_TO_MOL:
            return SMILES_TO_MOL[self.smiles]
        else:
            mol = Chem.MolFromSmiles(self.smiles)
            SMILES_TO_MOL[self.smiles] = mol
            return mol

    def set_features_mol(self, features_generator: List[Union[str, Callable]]):
        if features_generator is None:
            self.features_mol = None
            return
        self.features_mol = []
        for fg in features_generator:
            features_generator_ = FeaturesGenerator(features_generator_name=fg)
            self.features_mol.append(
                self.calc_features_mol(self.mol, features_generator_))
        self.features_mol = np.concatenate(self.features_mol)
        # Fix nans in features_mol
        replace_token = 0
        if self.features_mol is not None:
            self.features_mol = np.where(
                np.isnan(self.features_mol), replace_token, self.features_mol)

    @staticmethod
    def calc_features_mol(mol: Chem.Mol, features_generator: FeaturesGenerator):
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            features_mol = features_generator(mol)
        # for H2
        elif mol is not None and mol.GetNumHeavyAtoms() == 0:
            # not all features_mol are equally long, so use methane as dummy
            # molecule to determine length
            features_mol = np.zeros(len(features_generator(Chem.MolFromSmiles('C'))))
        else:
            features_mol = None
        return np.asarray(features_mol)


class ReactionGraph2D:
    """
    SingleReactionDatapoint： 2D graph of a single chemical reaction.
    """

    def __init__(self, reaction_smarts: str,
                 reaction_type: Literal['reaction', 'agent', 'reaction+agent'] = 'reaction'):
        # get chemical reaction object
        self.reaction_smarts = reaction_smarts
        self.chemical_reaction = ChemicalReaction(reaction_smarts)
        self.rxn = self.chemical_reaction.rxn
        # get graph
        self.reaction_type = reaction_type
        self.graph = HashGraph.from_cr(
            self.chemical_reaction, self.reaction_smarts)
        self.graph_agent = HashGraph.agent_from_cr(
            self.chemical_reaction, self.reaction_smarts)
        self.features_mol = None

    def __repr__(self) -> str:
        return self.reaction_smarts

    @property
    def X_single_graph(self) -> np.ndarray:  # 2d array
        if self.reaction_type == 'reaction':
            return np.asarray([[self.graph]])
        elif self.reaction_type == 'agent':
            return np.asarray([[self.graph_agent]])
        else:
            return np.asarray([[self.graph, self.graph_agent]])

    @property
    def X_multi_graph(self) -> Optional[np.ndarray]:
        return None


class MultiMolecularGraph2D:
    def __init__(self, data: List[MolecularGraph2D],
                 concentration: List[float] = None,
                 graph_type: Literal['single_graph', 'multi_graph'] = 'single_graph'):
        # read data point
        self.data = data
        # features_mol set None
        self.features_mol = None
        # set concentration
        if concentration is None:
            self.concentration = [1.0 / len(data)] * len(data)
        else:
            self.concentration = concentration
        graphs = [d.graph for d in self.data]
        map(lambda x, y: x.update_concentration(y), graphs, self.concentration)
        # set graph
        self.graph_type = graph_type
        if graph_type == 'single_graph':
            # combine several graphs into a disconnected graph
            self.graph = nx.disjoint_union_all(
                [g.to_networkx() for g in graphs])
            self.graph = _from_networkx(HashGraph, self.graph)
            self.graph.normalize_concentration()
        elif graph_type == 'multi_graph':
            self.graph = [rv for r in zip(graphs, self.concentration) for rv in r]

    def __repr__(self):
        return ';'.join(list(map(lambda x, y: x.__repr__() + ',%.3f' % y,
                                 self.data, self.concentration)))

    @property
    def mol(self) -> List[Chem.Mol]:
        mol = self.data[0].mol
        for d in self.data[1:]:
            mol = Chem.CombineMols(mol, d.mol)
        return mol

    @property
    def X_single_graph(self) -> Optional[np.ndarray]:  # 2d array.
        if self.graph_type == 'single_graph':
            return np.asarray([[self.graph]])
        else:
            return None

    @property
    def X_multi_graph(self) -> Optional[np.ndarray]:  # 2d array.
        if self.graph_type == 'multi_graph':
            return np.asarray([self.graph])
        else:
            return None

    @classmethod
    def from_smiles(cls, smiles: str):
        return cls([MolecularGraph2D(smiles)])

    @classmethod
    def from_smiles_list(cls, smiles: List[str], concentration: List[float] = None,
                         graph_type: Literal['single_graph', 'multi_graph'] = 'single_graph'):
        return cls([MolecularGraph2D(s) for s in smiles], concentration,
                   graph_type)

    def set_features_mol(self, features_generator: List[Union[str, Callable]] = None,
                         features_combination: Literal['concat', 'mean'] = None):
        if features_generator is None:
            return
        if len(self.data) != 1:
            self.features_mol = []
            for i, d in enumerate(self.data):
                d.set_features_mol(features_generator)
                self.features_mol.append(d.features_mol * self.concentration[i])
            if features_combination == 'mean':
                self.features_mol = np.mean(self.features_mol, axis=0).reshape(1, -1)
            elif features_combination == 'concat':
                self.features_mol = np.concatenate(self.features_mol).reshape(1, -1)
            else:
                raise ValueError(f'unknown feature combination: f{features_combination}')
        else:
            self.data[0].set_features_mol(features_generator)
            self.features_mol = self.data[0].features_mol.reshape(1, -1)


class Graph3D:
    def __init__(self, ASE: ase.atoms.Atoms):
        self.ASE = ASE
        self.graph = HashGraph.from_ase(ASE)

    @property
    def X_single_graph(self) -> Optional[np.ndarray]:  # 2d array.
        return np.asarray([[self.graph]])

    @property
    def X_multi_graph(self) -> Optional[np.ndarray]:  # 2d array.
        return None

    @property
    def features_mol(self) -> Optional[np.ndarray]:
        return None


class CompositeDatapoint:
    def __init__(self, data_p: List[MultiMolecularGraph2D],
                 data_m: List[MultiMolecularGraph2D],
                 data_cr: List[ReactionGraph2D],
                 data_3d: List[Graph3D]):
        # pure, mixture and chemical reactions.
        self.data_p = data_p
        self.data_m = data_m
        self.data_cr = data_cr
        self.data_3d = data_3d
        self.data = data_p + data_m + data_cr + data_3d

    def __repr__(self) -> str:
        return ';'.join(list(map(lambda x: x.__repr__(), self.data)))

    @property
    def mols(self) -> List[Chem.Mol]:
        mols = []
        for data in self.data_p:
            mols.append(data.mol)
        for data in self.data_m:
            mols.append(data.mol)
        assert len(self.data_cr) == 0
        assert len(self.data_3d) == 0
        return mols

    @property
    def n_heavy(self) -> int:
        if len(self.data_p) == 1:
            assert len(self.data_m) == 0
            assert len(self.data_cr) == 0
            assert len(self.data_3d) == 0
            return self.data_p[0].data[0].mol.GetNumAtoms()
        elif len(self.data_3d) == 1:
            assert len(self.data_p) == 0
            assert len(self.data_m) == 0
            assert len(self.data_cr) == 0
            return len([n for n in self.data_3d[0].ASE.arrays['numbers'] if n != 1])

    @property
    def X(self) -> np.ndarray:  # 2d array.
        return concatenate([self.X_single_graph, self.X_multi_graph, self.X_features_mol], axis=1)

    @property
    def X_single_graph(self) -> np.ndarray:  # 2d array.
        return concatenate([d.X_single_graph for d in self.data], axis=1)

    @property
    def X_multi_graph(self) -> np.ndarray:  # 2d array.
        return concatenate([d.X_multi_graph for d in self.data], axis=1)

    @property
    def X_features_mol(self) -> np.ndarray:  # 2d array.
        return concatenate([d.features_mol for d in self.data], axis=1)

    def set_features_mol(self, features_generator: List[Union[str, Callable]] = None,
                         features_combination: Literal['concat', 'mean'] = None):
        if features_generator is None:
            return
        assert len(self.data_3d) == 0
        assert len(self.data_cr) == 0
        for d in self.data_p:
            d.set_features_mol(features_generator, features_combination)
        for d in self.data_m:
            d.set_features_mol(features_generator, features_combination)


class SubDataset:
    def __init__(self, data: CompositeDatapoint,
                 targets: np.ndarray,  # 2d array.
                 features_add: Optional[np.ndarray] = None):  # 2d array.
        self.data = data
        # set targets
        assert targets is None or targets.ndim == 2
        self.targets = targets
        # set features_add
        if features_add is not None:
            assert features_add.ndim == 2
        self.features_add = features_add
        self.ignore_features_add = False

    def __len__(self) -> int:
        if self.targets is not None:
            return self.targets.shape[0]
        elif self.features_add is not None:
            return self.features_add.shape[0]
        else:
            return 1

    @property
    def mol(self) -> Chem.Mol:
        return self.data.mols

    @property
    def repr(self) -> np.ndarray:  # 2d array str.
        if self.features_add is None or self.ignore_features_add:
            return np.asarray([[self.data.__repr__()]])
        else:
            return np.asarray([list(map(lambda x: self.data.__repr__() + ';' + str(x), self.features_add.tolist()))]).T

    @property
    def X(self):
        return self.expand_features_add(self.data.X, features_add=True)

    @property
    def X_graph(self) -> np.ndarray:  # 2d array graph.
        return concatenate([self.X_single_graph, self.X_multi_graph], axis=1)

    @property
    def X_repr(self) -> np.ndarray:  # 2d array str.
        return self.expand_features_add(np.asarray([[self.data.__repr__()]]))

    @property
    def X_single_graph(self) -> np.ndarray:  # 2d array graph.
        return self.expand_features_add(self.data.X_single_graph)

    @property
    def X_multi_graph(self) -> np.ndarray:  # 2d array graph.
        return self.expand_features_add(self.data.X_multi_graph)

    @property
    def X_features_mol(self) -> np.ndarray:  # 2d array.
        return self.expand_features_add(self.data.X_features_mol)

    def expand_features_add(self, X, features_add=False):
        if X is None:
            return None
        if self.features_add is None or self.ignore_features_add:
            return X
        else:
            if features_add:
                return np.c_[X.repeat(len(self), axis=0),
                             self.features_add]
            else:
                return X.repeat(len(self), axis=0)

    def set_features(self, features_generator: List[Union[str, Callable]] = None,
                     features_combination: Literal['concat', 'mean'] = None):
        self.data.set_features_mol(features_generator, features_combination)


class Dataset:
    def __init__(self, data: List[SubDataset] = None,
                 features_mol_scaler: StandardScaler = None,
                 features_add_scaler: StandardScaler = None,
                 graph_kernel_type: Literal['graph', 'pre-computed'] = None):
        self.data = data
        self.unify_datatype()
        self.features_mol_scaler = features_mol_scaler
        self.features_add_scaler = features_add_scaler
        # Determine the Dataset.X.
        self.graph_kernel_type = graph_kernel_type
        self.set_ignore_features_add(False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[SubDataset, List[SubDataset]]:
        return self.data[item]

    @property
    def mols(self) -> List[Chem.Mol]:
        return [d.mol for d in self.data]

    @property
    def X(self) -> np.ndarray:
        if self.graph_kernel_type is None:
            return concatenate([self.X_features], axis=1)
        elif self.graph_kernel_type == 'graph':
            return concatenate([self.X_graph, self.X_features], axis=1, dtype=object)
        else:
            return concatenate([self.X_repr, self.X_features_add], axis=1, dtype=object)

    @property
    def y(self):
        y = concatenate([d.targets for d in self.data], axis=0)
        if y is not None and y.shape[1] == 1:
            return y.ravel()
        else:
            return y

    @property
    def repr(self) -> np.ndarray:  # 2d array str.
        return concatenate([d.repr for d in self.data])

    @property
    def X_repr(self) -> np.ndarray:  # 2d array str.
        return concatenate([d.X_repr for d in self.data])

    @property
    def X_graph(self) -> Optional[np.ndarray]:
        if self.data is None:
            return None
        else:
            return concatenate([d.X_graph for d in self.data])

    @property
    def X_mol(self):
        return concatenate([self.X_graph, self.X_features_mol], axis=1)

    @property
    def X_raw_features_mol(self) -> Optional[np.ndarray]:
        # assert self.graph_kernel_type == 'graph'
        return concatenate([d.X_features_mol for d in self.data])

    @property
    def X_features_mol(self) -> np.ndarray:
        features_mol = self.X_raw_features_mol
        if self.features_mol_scaler is not None:
            features_mol = self.features_mol_scaler.transform(features_mol)
        return features_mol

    @property
    def X_raw_features_add(self) -> Optional[np.ndarray]:
        if self.ignore_features_add:
            return None
        else:
            return concatenate([d.features_add for d in self.data])

    @property
    def X_features_add(self) -> np.ndarray:
        features_add = self.X_raw_features_add
        if self.features_add_scaler is not None and features_add is not None:
            features_add = self.features_add_scaler.transform(features_add)
        return features_add

    @property
    def X_features(self) -> np.ndarray:
        return concatenate([self.X_features_mol, self.X_features_add], axis=1)

    @property
    def N_MGK(self) -> int:
        if self.data[0].data.X_single_graph is None:
            return 0
        else:
            return self.data[0].data.X_single_graph.size

    @property
    def N_conv_MGK(self) -> int:
        if self.data[0].data.X_multi_graph is None:
            return 0
        else:
            return self.data[0].data.X_multi_graph.size

    @property
    def N_tasks(self) -> int:
        return self.data[0].targets.shape[1]

    @property
    def N_features_mol(self):
        if self.data[0].data.X_features_mol is None:
            return 0
        else:
            return self.data[0].data.X_features_mol.shape[1]

    @property
    def N_features_add(self):
        if self.data[0].features_add is None or self.ignore_features_add:
            return 0
        else:
            return self.data[0].features_add.shape[1]

    def features_size(self):
        return self.N_features_mol + self.N_features_add

    def copy(self):
        return copy.deepcopy(self)

    def set_ignore_features_add(self, ignore_features_add: bool) -> bool:
        self.ignore_features_add = ignore_features_add
        if self.data is not None:
            for d in self.data:
                d.ignore_features_add = ignore_features_add
        return ignore_features_add

    def normalize_features_mol(self):
        if self.X_raw_features_mol is not None:
            self.features_mol_scaler = StandardScaler().fit(self.X_raw_features_mol)
        else:
            self.features_mol_scaler = None

    def normalize_features_add(self):
        if self.X_raw_features_add is not None:
            self.features_add_scaler = StandardScaler().fit(self.X_raw_features_add)
        else:
            self.features_add_scaler = None

    def unify_datatype(self, X=None):
        if X is None:
            X = self.X_graph
        else:
            X = np.concatenate([X, self.X_graph], axis=0)
        if X is None:
            return
        for i in range(X.shape[1]):
            self._unify_datatype(X[:, i])

    @staticmethod
    def _unify_datatype(X: List[HashGraph]):
        if X[0].__class__ == list:
            graphs = []
            for x in X:
                graphs += x[::2]
            HashGraph.unify_datatype(graphs, inplace=True)
        else:
            HashGraph.unify_datatype(X, inplace=True)

    def clear_cookie(self):
        if self.X_graph is not None:
            for X in self.X_graph:
                for g in X:
                    g.cookie.clear()

    def save(self, path, filename='dataset.pkl', overwrite=False):
        f_dataset = os.path.join(path, filename)
        if os.path.isfile(f_dataset) and not overwrite:
            raise RuntimeError(
                f'Path {f_dataset} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_dataset, 'wb'), protocol=4)

    @classmethod
    def load(cls, path, filename='dataset.pkl'):
        f_dataset = os.path.join(path, filename)
        store = pickle.load(open(f_dataset, 'rb'))
        dataset = cls()
        dataset.__dict__.update(**store)
        return dataset

    @staticmethod
    def get_subDataset(
            pure: List[str],
            mixture: List[List[Union[str, float]]],
            mixture_type: Literal['single_graph', 'multi_graph'],
            reaction: List[str],
            reaction_type: Literal['reaction', 'agent', 'reaction+agent'],
            targets: np.ndarray,
            features: Optional[np.ndarray] = None,
            features_generator: List[Union[str, Callable]] = None,
            features_combination: Literal['concat', 'mean'] = None
    ) -> SubDataset:
        data_p = []
        data_m = []
        data_r = []
        data_3d = []
        pure = [] if pure is None else list(pure)
        for smiles in pure:
            data_p.append(MultiMolecularGraph2D.from_smiles(smiles))
        for m in mixture:
            data_m.append(MultiMolecularGraph2D.from_smiles_list(
                m[0::2], concentration=m[1::2], graph_type=mixture_type))
        for rg in reaction:
            data_r.append(ReactionGraph2D(rg, reaction_type))
        data = SubDataset(CompositeDatapoint(data_p, data_m, data_r, data_3d), targets, features)
        data.set_features(features_generator, features_combination)
        return data

    """
    @classmethod
    def from_public(cls, args: CommonArgs):
        if args.data_public == 'qm7':
            qm_data = QM7(ase=True)
        elif args.data_public == 'qm9':
            qm_data = QM9(ase=True)
        else:
            raise RuntimeError(f'Unknown public data set {args.data_public}')
        data = Parallel(
            n_jobs=args.n_jobs, verbose=True, prefer='processes')(
            delayed(cls.get_subDataset)(
                [],
                [],
                args.mixture_type,
                [],
                args.reaction_type,
                tolist(qm_data.iloc[i].get('atoms')),
                to_numpy(qm_data.iloc[i:i+1][args.target_columns]),
                to_numpy(qm_data.iloc[i:i+1].get(args.feature_columns)),
                args.features_generator,
            )
            for i in qm_data.index)
        return cls(data)
    """

    @classmethod
    def from_df(cls, df: pd.DataFrame,
                pure_columns: List[str] = None,
                mixture_columns: List[str] = None,
                reaction_columns: List[str] = None,
                feature_columns: List[str] = None,
                target_columns: List[str] = None,
                features_generator: List[Union[str, Callable]] = None,
                features_combination: Literal['concat', 'mean'] = None,
                mixture_type: Literal['single_graph', 'multi_graph'] = 'single_graph',
                reaction_type: Literal['reaction', 'agent', 'reaction+agent'] = 'reaction',
                group_reading: bool = False,
                n_jobs: int = 8):
        if group_reading:
            pure_columns = pure_columns or []
            mixture_columns = mixture_columns or []
            reaction_columns = reaction_columns or []
            n1 = len(pure_columns)
            n2 = len(mixture_columns)
            n3 = len(reaction_columns)
            groups = df.groupby(pure_columns + mixture_columns + reaction_columns)
            data = Parallel(
                n_jobs=n_jobs, verbose=True, prefer='processes')(
                delayed(cls.get_subDataset)(
                    (lambda x: [x] if x.__class__ == str else tolist(x))(g[0])[0:n1],
                    (lambda x: tolist([x]) if x.__class__ == str else tolist(x))(g[0])[n1:n1 + n2],
                    mixture_type,
                    (lambda x: [x] if x.__class__ == str else tolist(x))(g[0])[n1 + n2:n1 + n2 + n3],
                    reaction_type,
                    to_numpy(g[1][target_columns]),
                    to_numpy(g[1][feature_columns]),
                    features_generator,
                    features_combination
                )
                for g in groups)
        else:
            data = Parallel(
                n_jobs=n_jobs, verbose=True, prefer='processes')(
                delayed(cls.get_subDataset)(
                    tolist(df.iloc[i].get(pure_columns)),
                    tolist(df.iloc[i].get(mixture_columns)),
                    mixture_type,
                    tolist(df.iloc[i].get(reaction_columns)),
                    reaction_type,
                    to_numpy(df.iloc[i:i + 1][target_columns]),
                    to_numpy(df.iloc[i:i + 1].get(feature_columns)),
                    features_generator,
                    features_combination
                )
                for i in df.index)
        return cls(data)


def tolist(list_: Union[pd.Series, List]) -> List[str]:
    if list_ is None:
        return []
    else:
        result = []
        for string_ in list_:
            if ',' in string_:
                result.append(json.loads(string_))
            else:
                result.append(string_)
        return result


def to_numpy(list_: pd.Series) -> Optional[np.ndarray]:
    if list_ is None:
        return None
    else:
        ndarray = list_.to_numpy()
        if ndarray.size == 0:
            return None
        else:
            return ndarray


def get_data(path: str,
             pure_columns: List[str] = None,
             mixture_columns: List[str] = None,
             reaction_columns: List[str] = None,
             feature_columns: List[str] = None,
             target_columns: List[str] = None,
             features_generator: List[str] = None,
             features_combination: Literal['concat', 'mean'] = None,
             mixture_type: Literal['single_graph', 'multi_graph'] = 'single_graph',
             reaction_type: Literal['reaction', 'agent', 'reaction+agent'] = 'reaction',
             group_reading: bool = False,
             graph_kernel_type: Literal['graph', 'pre-computed'] = None,
             n_jobs: int = 8) -> Dataset:
    df = pd.read_csv(path)
    data = Dataset.from_df(df, pure_columns=pure_columns,
                           mixture_columns=mixture_columns,
                           reaction_columns=reaction_columns,
                           feature_columns=feature_columns,
                           target_columns=target_columns,
                           features_generator=features_generator,
                           features_combination=features_combination,
                           mixture_type=mixture_type,
                           reaction_type=reaction_type,
                           group_reading=group_reading,
                           n_jobs=n_jobs)
    data.graph_kernel_type = graph_kernel_type
    return data
