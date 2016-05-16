"""Provide helper functions for working with pandas dataframes."""
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
