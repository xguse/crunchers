"""Provide functions for performing non-standard-ish column-wise transformations."""
import itertools as itr

import pandas as pd
import numpy as np

def apply_ignore_null(func, s, fillwith=None):
    """Perform `func` on values on `s` that are not 'nan' or equivalent.

    `func` applied to `s` after filling the 'nan' with `fillwith`.
    If `fillwith` is None, min(s) is used.

    You may prefer to use the mean or median like this:

    >>> apply_ignore_null(func, s, fillwith=np.mean(s))

    Returns a reconstituted pandas.Series with 'nan' everywhere there was an original 'nan',
    but with the transformed values everywhere else.
    """
    if fillwith is None:
        fillwith = s.min()

    idx = s.index
    isnull = s.isnull()

    applied = pd.Series(func(s.fillna(fillwith)), index=idx)

    return pd.concat([applied[s.notnull()], s[s.isnull()]])


def apply_pairwise(series, func):
    """Apply `func` to items in `series` pairwise: return dataframe."""
    func = sum
    names = series.index.values
    df = pd.DataFrame(index=names, columns=names)

    for idx, col  in itr.combinations(names,2):
        df[col][idx] = func([series[idx], series[col]])

    return df
