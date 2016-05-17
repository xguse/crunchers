"""Provide functions for performing non-standard-ish column-wise transformations."""
import itertools as itr

import pandas as pd
import numpy as np



# def test_apply_ignore_null(func, s, fillwith=None):
#     """
#     """
#     if fillwith is None:
#         fillwith = s.min()
#
#     idx = s.index
#     isnull = s.isnull()
#
#     applied = pd.Series(func(s.fillna(fillwith)), index=idx)
#
#     return pd.concat([applied[s.notnull()], s[s.isnull()]])
#
#
# def test_apply_pairwise(series, func):
#     """"""
#     func = sum
#     names = series.index.values
#     df = pd.DataFrame(index=names, columns=names)
#
#     for idx, col  in itr.combinations(names,2):
#         df[col][idx] = func([series[idx], series[col]])
#
#     return df
