#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pickle
from mgktools.features_mol.features_generators import FeaturesGenerator, AVAILABLE_FEATURES_GENERATORS


@pytest.mark.parametrize('features_generator_name', AVAILABLE_FEATURES_GENERATORS)
def test_features_generator_picklable(features_generator_name):
    fg = FeaturesGenerator(features_generator_name)
    pickle.dumps(fg)
