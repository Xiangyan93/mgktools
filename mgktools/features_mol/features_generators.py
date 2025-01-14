#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP, GetAvalonCountFP
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


AVAILABLE_FEATURES_GENERATORS = ['morgan', 'morgan_count', 'rdkit_208', 'rdkit_2d', 'rdkit_2d_normalized', 'rdkit_topol', 'layered', 'torsion', 'atom_pair', 'avalon', 'avalon_count', 'maccskey', 'pattern']


class FeaturesGenerator:
    def __init__(self, features_generator_name: str,
                 radius: int = 2,
                 num_bits: int = 2048,
                 atomInvariantsGenerator: bool = False):
        self.features_generator_name = features_generator_name
        self.radius = radius
        self.num_bits = num_bits
        self.atomInvariantsGenerator = atomInvariantsGenerator
        if features_generator_name == 'rdkit_2d':
            self.generator = rdDescriptors.RDKit2D()
        elif features_generator_name == 'rdkit_2d_normalized':
            self.generator = rdNormalizedDescriptors.RDKit2DNormalized()

    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        if self.features_generator_name == 'morgan':
            return self.morgan_binary_features_generator(mol)
        elif self.features_generator_name == 'morgan_count':
            return self.morgan_counts_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d':
            return self.rdkit_2d_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d_normalized':
            return self.rdkit_2d_normalized_features_generator(mol)
        elif self.features_generator_name == 'rdkit_208':
            return self.rdkit_208_features_generator(mol)
        elif self.features_generator_name == 'rdkit_topol':
            return self.rdkit_topological_features_generator(mol)
        elif self.features_generator_name == 'layered':
            return self.layered_features_generator(mol)
        elif self.features_generator_name == 'torsion':
            return self.torsion_features_generator(mol)
        elif self.features_generator_name == 'atom_pair':
            return self.atom_pair_features_generator(mol)
        elif self.features_generator_name == 'avalon':
            return self.avalon_features_generator(mol)
        elif self.features_generator_name == 'avalon_count':
            return self.avalon_count_features_generator(mol)
        elif self.features_generator_name == 'maccskey':
            return self.maccskey_features_generator(mol)
        elif self.features_generator_name == 'pattern':
            return self.pattern_features_generator(mol)
        else:
            raise ValueError(f'unknown features generator: {self.features_generator_name}')

    def morgan_binary_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a binary Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the binary Morgan fingerprint.
        """
        if self.atomInvariantsGenerator:
            invgen = AllChem.GetMorganFeatureAtomInvGen()
            generator = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.num_bits, atomInvariantsGenerator=invgen)
        else:
            generator = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.num_bits)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(generator.GetFingerprint(mol).ToList())

    def morgan_counts_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a counts-based Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the counts-based Morgan fingerprint.
        """
        if self.atomInvariantsGenerator:
            invgen = AllChem.GetMorganFeatureAtomInvGen()
            generator = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.num_bits, atomInvariantsGenerator=invgen)
        else:
            generator = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.num_bits)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(generator.GetCountFingerprint(mol).ToList())

    def rdkit_2d_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates RDKit 2D features_mol for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features_mol.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if isinstance(mol, Chem.Mol) else mol
        return np.array(self.generator.process(smiles)[1:])

    def rdkit_2d_normalized_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates RDKit 2D normalized features_mol for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features_mol.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if isinstance(mol, Chem.Mol) else mol
        return np.array(self.generator.process(smiles)[1:])

    @staticmethod
    def rdkit_208_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        # define chemical features for molecular descriptions
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        descr = Descriptors._descList
        calc = [x[1] for x in descr]
        ds_n = []
        for d in calc:
            v = d(mol)
            if v > np.finfo(np.float32).max:  # postprocess descriptors for freak large values
                ds_n.append(np.finfo(np.float32).max)
            elif math.isnan(v):
                ds_n.append(np.float32(0.0))
            else:
                ds_n.append(np.float32(v))
        return np.array(ds_n)

    def layered_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a layered feature vector for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the layered feature vector.
        """
        return np.array(Chem.LayeredFingerprint(mol, fpSize=self.num_bits).ToList())
    
    def torsion_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        generator = AllChem.GetTopologicalTorsionGenerator(fpSize=self.num_bits)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(generator.GetFingerprint(mol).ToList())

    def atom_pair_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        generator = AllChem.GetAtomPairGenerator(fpSize=self.num_bits)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(generator.GetFingerprint(mol).ToList())

    def avalon_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(GetAvalonFP(mol, nBits=self.num_bits).ToList())
    
    def avalon_count_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(GetAvalonCountFP(mol, nBits=self.num_bits).ToList())

    def maccskey_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(AllChem.GetMACCSKeysFingerprint(mol).ToList())
    
    def pattern_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(AllChem.rdmolops.PatternFingerprint(mol, fpSize=self.num_bits).ToList())

    def rdkit_topological_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        generator = AllChem.GetRDKitFPGenerator(fpSize=self.num_bits)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return np.array(generator.GetFingerprint(mol).ToList())
