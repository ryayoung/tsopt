# Maintainer:     Ryan Young
# Last Modified:  Jul 28, 2022

import pandas as pd
import numpy as np

def nrows(df) -> int:
    return df.shape[0]

def ncols(df) -> int:
    return df.shape[1]

def valid_dtype(val) -> bool:
    """
    A foolproof way to determine if a value is a number.
    We need this (instead of type() or isnumeric())
    because pandas and numpy are annoying.
    """
    try:
        float(val)
        return True
    except ValueError:
        return False


def isfull(df, idx=None, col=None) -> bool:
    '''
    Check if a dataframe or series is full
    What is 'full'?
        - No nulls. No infinity.
    '''
    def vec_full(s) -> bool:
        return ~ s.isna().any().any() and ~ np.isinf(s).any().any()

    if type(df) == pd.Series:
        assert (idx == None and col == None), "Can't check idx or col for series"
        return vec_full(df)

    if idx == None and col == None:
        return vec_full(df)
    if idx != None:
        return vec_full(df.T[idx])
    if col != None:
        return vec_full(df[col])


def strip_null_ending_cols_and_rows(df) -> pd.DataFrame:
    '''
    Removes null rows from the bottom, and null columns from end (right) of df
    '''
    col_mask = df.notna().any(0)[::-1].cumsum()[::-1].astype(bool)
    row_mask = df.notna().any(1)[::-1].cumsum()[::-1].astype(bool)
    return df.loc[:,col_mask].T.loc[:,row_mask].T


def staged(iterable):
    '''
    Iterate while knowing the next element.
    Example: data = ('a', 'b', 'c'):
        >>> for first, next in data.staged:
        >>>     print(first, next)

        (output):
            a b
            b c
    '''
    iterator = iter(iterable)
    curr = next(iterator)
    for nxt in iterator:
        yield (curr, nxt)
        curr = nxt


