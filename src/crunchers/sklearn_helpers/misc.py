"""Collect misc sklearn helpers here."""

import pandas as pd

def repandasify(array, y_names, X_names=None):
    """Convert numpy array into pandas dataframe using provided index and column names."""
    df = pd.DataFrame(data=array, index=y_names, columns=X_names)
    return df
