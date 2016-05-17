"""Collect misc sklearn helpers here."""

from hypothesis import given
from hypothesis import strategies as st

import pandas as pd

import crunchers as crun


def dkeys():
    return st.one_of(st.integers()|st.text())

def dvalues(col_len=None):
    if col_len is None:
        col_len = st.integers(min_size=2)

    return st.lists(elements=st.integers()|st.text()|st.booleans()|datetimes(),
                        min_size=col_len,
                        max_size=col_len,
                        unique_by=None, unique=False)


@given(headers=st.lists(elements=st.text(), min_size=2), col_maker=dvalues)
def test_repandasify_roundtrip(headers, col_maker):
    """Test repandasify roundtrip."""
    # generate a dataframe, then convert it to an np.array
    d = st.dictionaries(keys=headers,values=col_maker())
    df = pd.DataFrame.from_dict(d)
    arr = np.array(df)

    # require that the repandasified df == original df
    assert df == crun.sklearn_helpers.misc.repandasify(array=arr,
                                                       y_names=df.index.values,
                                                       X_names=df.columns.values)
