#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import pickle
import os
import rdkit.Chem.AllChem as Chem


def save_mols_pkl(mols: List[Chem.Mol], path, filename='mols.pkl'):
    f_mols = os.path.join(path, filename)
    atomic_attribution = []
    for mol in mols:
        atomNote = []
        for atom in mol.GetAtoms():
            atomNote.append(atom.GetProp('atomNote'))
        atomic_attribution.append(atomNote)
    with open(f_mols, "wb") as f:
        pickle.dump([mols, atomic_attribution], f)


def load_mols_pkl(path, filename='mols.pkl'):
    f_mols = os.path.join(path, filename)
    with open(f_mols, "rb") as f:
        mols, atomic_attribution = pickle.load(f)
    for i, mol in enumerate(mols):
        for j, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('atomNote', atomic_attribution[i][j])
    return mols
