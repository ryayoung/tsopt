# Maintainer:     Ryan Young
# Last Modified:  Aug 20, 2022

import pandas as pd
import numpy as np

import tsopt.pandas_added_methods


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


