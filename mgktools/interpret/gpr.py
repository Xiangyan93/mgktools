#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.model.gaussian_process import GaussianProcessRegressor


class InterpretableGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.normalize_y is False

    def predict_interpretable(self, Z):
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(None, Z, self.X)
        Linv = np.linalg.inv(self.Kinv.L)
        K_left = Ks @ Linv.T @ Linv
        return K_left, np.einsum('ij,j->ij', Ks @ Linv.T @ Linv, self.y * self._ystd + self._ymean)

    def predict_nodal(self, Z):
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self.kernel(Z, self.X, nodal_X=True)
        ymean = (Ks @ self.Ky) * self._ystd + self._ymean
        return ymean
