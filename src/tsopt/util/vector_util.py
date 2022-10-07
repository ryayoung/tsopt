# Maintainer:     Ryan Young
# Last Modified:  Oct 04, 2022

import pandas as pd
import numpy as np


def binary_reverse(val:int, iterable):
    if val == 0:
        return iterable
    if val == 1:
        return iterable[::-1]


def is_list_or_tuple(val, inherit_from=True) -> bool:

    if inherit_from:
        return (isinstance(val, list) or isinstance(val, tuple))

    return (type(val) == list or type(val) == tuple)


def is_list_tuple_or_series(val, inherit_from=True) -> bool:
    if is_list_or_tuple(val, inherit_from):
        return True
    if inherit_from:
        return isinstance(val, pd.Series)
    return type(val) == pd.Series


def is_frame(val) -> bool:
    return isinstance(val, pd.core.generic.NDFrame)

def nodes_from_stage_dfs(dfs) -> tuple:
    nodes = [ df.index for df in dfs ] + [ dfs[-1].columns ]
    return tuple( tuple(group) for group in nodes )


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


def staged(iterable):
    '''
    Iterate while knowing the next element.
    Example: data = ('a', 'b', 'c'):
        >>> for first, next in staged(data):
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


def combine_if(sr1, sr2, func) -> pd.Series:
    '''
    Combine two series with same index. For each pair of values,
    if both are null, return null. If one is null, return the other.
    If both are present, compare them using provided function (min or max usually)
    '''
    def pick(a,b):
        if not np.isnan(a) and not np.isnan(b):
            return func(a,b)
        if not np.isnan(a):
            return a
        return b
    return sr1.combine(sr2, pick)


def read_file(name, excel_file) -> pd.DataFrame:
    loc = dict(index_col=None, header=None)
    if excel_file:
        if type(excel_file) == str:
            excel_file = pd.ExcelFile(excel_file)
        return pd.read_excel(excel_file, name, **loc)
    elif name.endswith('xlsx'):
        return pd.read_excel(name, **loc)
    elif name.endswith('csv'):
        return pd.read_csv(name, **loc)
    else:
        raise ValueError(f'Invalid filename, {name}')


def raw_df_from_file(name, excel_file=None) -> pd.DataFrame:
    '''
    Removes all column headers, indexes, and empty cols and rows
    '''
    if str(name).endswith('pkl'):
        return pd.read_pickle(name)

    df = read_file(name, excel_file)

    df = df.replace(' ', np.nan
        ).replace(r"[a-zA-Z]", np.nan, regex=True
        ).strip_null_borders(
        ).reset(
        ).astype(float)

    return df


def raw_sr_from_file(name, excel_file=None) -> pd.Series:
    df = raw_df_from_file(name, excel_file)

    if df.nrows > 1 and df.ncols > 1:
        raise ValueError(f'Invalid shape. 1-dimensional vector required.')

    if df.ncols > df.nrows:
        return df.T[0]
    return df[0]


