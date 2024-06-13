# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:07:06 2024

@author: ASUS
"""

from sklearn import preprocessing


def data_normalize(data):
    scaler = preprocessing.RobustScaler(
                            with_centering=True,
                            with_scaling=True,
                            quantile_range=(25.0, 75.0),
                            copy=True,
                            unit_variance=False)





    return scaler.fit_transform(data)