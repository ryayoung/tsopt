import pandas as pd
import numpy as np
import re

'''
This module lets us create functions and add them to any existing
class, (like pd.DataFrame), or to the packages/libraries themselves,
like pandas.
---
Instructions:
- Ignore the code for _add_func_to_object() and _add_all_funcs_to...
- At the very bottom of this file, call _add_all_funcs_to_objects().
- Declare functions. If you plan on adding it as an instance method
    of a class, it must take 'self' as a parameter, and return 'self'
    if you want the updated object to be returned.
- To add your function as an instance method to a class,
    write ::<classname> at the BEGINNING of the docstring of the
    function. Example: """ ::pd.Series """
- Or you can list multiple classes if you want the same code to be
    added to both. Example: """ ::pd.Series,pd.DataFrame """. Make
    sure there are no spaces in between class names, only a comma.
- Any function without these declarations will be ignored.

How do you add a function to an entire package/library like pandas?
- (Example: `pd.isfull(my_df)` instead of my_df.isfull()).
- Use the same technique, but instead of declaring the class name,
    use the name as you've imported it. Example: """ ::pd """

How do you write multiple implementations of the same function?
- Just place an additional underscore at the end of your function
    name, each time you repeat it. The _add_all_funcs_to_objects()
    function will strip away those underscores when determining the
    name to assign the function that gets added to an instance.
    Example: `nrows(df)`, `nrows_(self)`, `nrows__(self)`

Accepted items are functions, and properties. If you haven't used
properties before, just think of them as a way to call an instance
method without using parentheses. Pandas uses this for `shape`.
Instead of calling `df.shape()`, you call `df.shape`, and it does
the same thing. You can declare one by placing `@property` right
above the method declaration. (Note: properties are intended to
be treated as variables, so your user will expect a value returned,
instead of an object or None)

Note the order in which I'm declaring different implementations
of a function. In the case of `nrows`, I'll put the logic in a
regular function that takes a df as an argument, and when declaring
`nrows` again as an instance method, call the original function.

HOWEVER, this order should be REVERSED when implementing a function
with different logic for desired classes; For `isfull()`, I put
the logic in the instance method versions for Series and DataFrame.
Then when declaring the regular pandas function for it, it takes
an object of unknown type (df or series), doesn't care what the
type is, and just calls its instance method without any validation,
since the variable's type will automatically determine which function
gets called. Since we don't know what arguments will be passed,
just pass along `*args` and `**kwargs` to the instance method you're
calling.

'''


def _add_func_to_object(name, func=None):
    if not func:
        func = globals().get(name)
    doc = func.__doc__
    if doc:
        objects = re.findall(r'^\s*:{2}(\S+)', doc)
        if len(objects) > 0:
            objects = objects[0].split(',')
            for obj in objects:
                setattr(eval(obj), name, func)

def _add_all_funcs_to_objects(obj=None, name=None):
    for name,func in list(globals().items()):
        if callable(func) or isinstance(func, property):
            name = name.rstrip('_')
            _add_func_to_object(name, func)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def reset(self, axis=None, names=False) -> pd.DataFrame:
    ''' ::pd.DataFrame
    Resets indexes and columns to ordered ints, zero-based.
    '''
    if not axis or axis == 1:
        self = self.reset_index(drop=True)
        if names:
            self.index.name = None
    if not axis or axis == 0:
        self.columns = range(0, self.ncols)
        if names:
            self.columns.name = None
    return self

def reset_(self, names=False) -> pd.Series:
    ''' ::pd.Series
    Resets indexes, index name, and series name
    '''
    self = self.reset_index(drop=True)
    if names:
        self.index.name = None
        self.name = None
    return self

def reset__(df, *args, **kwargs) -> pd.Series or pd.DataFrame:
    ''' ::pd '''
    return df.reset(*args, **kwargs)


# ----------------------------------------------------------------------------

# def _repr_html_(self):
    # ''' ::pd.Series '''
    # df = pd.DataFrame(self, columns=[self.name if self.name else ""])
    # return df._repr_html_()

# ----------------------------------------------------------------------------

def sums(self, full=False, concat=True) -> pd.Series:
    ''' ::pd.DataFrame '''
    T = self.T
    sums = self.sum()
    sums_t = T.sum()
    if full:
        sums[self.isna().any()] = np.nan
        sums_t[T.isna().any()] = np.nan

    if concat:
        return pd.concat([sums_t, sums])

    return [sums_t, sums]


def _sums(df, *args, **kwargs) -> pd.Series:
    ''' ::pd '''
    return df.sums(*args, **kwargs)


def maxs(self, full=False) -> pd.Series:
    ''' ::pd.DataFrame '''
    T = self.T
    maxs = self.max()
    maxs_t = T.max()
    if full:
        maxs[self.isna().any()] = np.nan
        maxs_t[T.isna().any()] = np.nan

    return pd.concat([maxs_t, maxs])

def _maxs(df, *args, **kwargs) -> pd.Series:
    ''' ::pd '''
    return df.maxs(*args, **kwargs)


def mins(self, full=False) -> pd.Series:
    ''' ::pd.DataFrame '''
    T = self.T
    mins = self.min()
    mins_t = T.min()
    if full:
        mins[self.isna().any()] = np.nan
        mins_t[T.isna().any()] = np.nan

    return pd.concat([mins_t, mins])

def _mins(df, *args, **kwargs) -> pd.Series:
    ''' ::pd '''
    return df.mins(*args, **kwargs)


# ----------------------------------------------------------------------------

def getsum(self, node, full=False) -> float:
    ''' ::pd.DataFrame '''
    def find_series(node):
        if node not in self.columns:
            if node in self.index:
                return self.T[node]
            raise ValueError("Invalid node")
        return self[node]

    sr = find_series(node)

    if full == True and sr.isna().any() == True:
        return np.nan

    return sr.sum()


# ----------------------------------------------------------------------------

def nrows(df) -> int:
    ''' ::pd
    Reader-friendly row count of dataframe
    '''
    return df.shape[0]

@property
def nrows_(self) -> int:
    ''' ::pd.Series '''
    return nrows(self)

@property
def nrows__(self) -> int:
    ''' ::pd.DataFrame '''
    return nrows(self)

# ----------------------------------------------------------------------------

def ncols(df) -> int:
    ''' ::pd
    Reader-friendly column count of dataframe
    '''
    return df.shape[1]

@property
def ncols_(self) -> int:
    ''' ::pd.DataFrame '''
    return ncols(self)

# ----------------------------------------------------------------------------

@property
def has_default_index(self) -> bool:
    ''' ::pd.Series,pd.DataFrame
    Checks if series has default indexes
    '''
    return tuple(self.index) == tuple(range(0,self.shape[0]))

def has_default_index_(self) -> bool:
    ''' ::pd '''
    return df.has_default_index

# ----------------------------------------------------------------------------

@property
def has_default_columns(self) -> bool:
    ''' ::pd.DataFrame
    Checks if dataframe has default columns
    '''
    return tuple(self.columns) == tuple(range(0, self.shape[1]))

def has_default_columns_(df) -> bool:
    ''' ::pd '''
    return df.has_default_columns

# ----------------------------------------------------------------------------

def strip_null_borders(self, axis=None) -> pd.DataFrame:
    ''' ::pd.DataFrame
    Removes null rows from the bottom & top
    Removes null columns from left and right
    '''
    if axis == None or axis == 1:
        first_idx = self.first_valid_index()
        last_idx = self.last_valid_index()
        self = self.loc[first_idx:last_idx]

    if axis == None or axis == 0:
        t = self.T
        first_col = t.first_valid_index()
        last_col = t.last_valid_index()
        t = t.loc[first_col:last_col]
        self = t.T

    return self


def strip_null_borders_(self) -> pd.Series:
    ''' ::pd.Series '''
    first_idx = self.first_valid_index()
    last_idx = self.last_valid_index()
    self = self.loc[first_idx:last_idx]

    return self


def strip_null_borders__(df, *args, **kwargs) -> pd.DataFrame or pd.Series:
    ''' ::pd '''
    return df.strip_null_borders(*args, **kwargs)

# ----------------------------------------------------------------------------

def strip_object_borders(self, axis=None) -> pd.DataFrame:
    ''' ::pd.DataFrame
    Removes null rows from the bottom & top
    Removes null columns from left and right
    '''
    if axis == None or axis == 1:
        first_idx = self.first_valid_index()
        last_idx = self.last_valid_index()
        self = self.loc[first_idx:last_idx]

    if axis == None or axis == 0:
        t = self.T
        first_col = t.first_valid_index()
        last_col = t.last_valid_index()
        t = t.loc[first_col:last_col]
        self = t.T

    return self


def strip_object_borders_(self) -> pd.Series:
    ''' ::pd.Series '''
    first_idx = self.first_valid_index()
    last_idx = self.last_valid_index()
    self = self.loc[first_idx:last_idx]

    return self


def strip_object_borders__(df, *args, **kwargs) -> pd.DataFrame or pd.Series:
    ''' ::pd '''
    return df.strip_object_borders(*args, **kwargs)

# ----------------------------------------------------------------------------

def isfull(self, idx=None, col=None) -> bool:
    ''' ::pd.DataFrame
    -
    Check if a dataframe is full
    What is 'full'?
        - No nulls, AND no infinities (np.inf).
    '''
    if self.empty:
        return False

    if idx and col:
        raise ValueError(f".isfull() can take either param, 'idx' or 'col', but not both")

    def full(df) -> bool:
        return ~ df.isna().any().any() and ~ np.isinf(df).any().any()
    if idx != None:
        return full(self.T[idx])
    if col != None:
        return full(self[col])

    return full(self)

def isfull_(self) -> bool:
    ''' ::pd.Series
    -
    Alt. impl. for pd.Series
    '''
    if self.empty:
        return False
    # return ~ self.isna().any() and ~ np.isinf(self).any()
    return ~ self.isna().any()

def isfull__(df, *args, **kwargs):
    ''' ::pd '''
    return df.isfull(*args, **kwargs)

# ----------------------------------------------------------------------------

def standardize_str(self, col, substring, **kwargs) -> pd.DataFrame:
    ''' ::pd.DataFrame '''
    self.loc[self[col].str.contains(substring, **kwargs), col] = substring
    return self

def standardize_str_(self, substring, **kwargs) -> pd.Series:
    ''' ::pd.Series '''
    self.loc[self.str.contains(substring, **kwargs)] = substring
    return self

def standardize_str__(df, *args, **kwargs):
    ''' ::pd '''
    return df.standardize_str(*args, **kwargs)

# ----------------------------------------------------------------------------

class DFShape(tuple):
    '''
    Improvement on pandas .shape for DataFrames
    - We're injecting this into pandas source code,
    so all .shape operations will work with this class
    instead of a regular tuple. There may be bugs, but none
    have been found yet.
    '''

    def __new__(self, rows, cols):
        self.__rows = rows
        self.__cols = cols
        return tuple.__new__(DFShape, (rows, cols))

    @property
    def rows(self):
        return self.__rows

    @property
    def cols(self):
        return self.__cols

    def __getitem__(self, key) -> int:
        if key == 'rows':
            return self.rows
        elif key == 'cols':
            return self.cols
        elif key == 0:
            return self.rows
        elif key == 1:
            return self.cols


class SRShape(tuple):
    '''
    Same as DFShape, but for pd.Series instead
    '''
    def __new__(self, rows):
        self.__rows = rows
        return tuple.__new__(SRShape, (rows,))

    @property
    def rows(self):
        return self.__rows

    def __getitem__(self, key) -> int:
        if key == 'rows':
            return self.rows
        elif key == 0:
            return self.rows

# ----------------------------------------------------------------------------

@property
def shape(self) -> DFShape:
    ''' ::pd.DataFrame
    '''
    return DFShape(len(self.index), len(self.columns))

@property
def shape_(self) -> SRShape:
    ''' ::pd.Series
    '''
    return SRShape(len(self.index))


def shape__(df, key=None) -> DFShape or SRShape:
    ''' ::pd
    The passed object will call either of its respective methods
    above, depending on its type (DataFrame or Series)
    '''
    if key:
        return df.shape[key]
    return df.shape


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
_add_all_funcs_to_objects()




# def strip_null_surroundings(self) -> pd.DataFrame:
    # ''' ::pd.DataFrame
    # Removes null rows from the bottom & top
    # Removes null columns from left and right
    # '''
    # top_left_empty = pd.isnull(self.iloc[0,0])
    # bot_right_empty = pd.isnull(self.iloc[-1,-1])
# 
    # if self.isna().all(0).any():
# 
        # col_mask_left = pd.Series(False, index=self.columns)
        # col_mask_right = pd.Series(False, index=self.columns)
# 
        # if top_left_empty:
            # col_mask_left = self.notna().any(0)[0::].cumsum()[0::].astype(bool)
        # if bot_right_empty:
            # col_mask_right = self.notna().any(0)[::-1].cumsum()[::-1].astype(bool)
# 
        # col_mask = ~(~col_mask_left + ~col_mask_right)
        # self = self.loc[:,col_mask]
# 
    # if self.isna().all(1).any():
# 
        # row_mask_top = pd.Series(False, index=self.index)
        # row_mask_bottom = pd.Series(False, index=self.index)
# 
        # if top_left_empty:
            # row_mask_top = self.notna().any(1)[0::].cumsum()[0::].astype(bool)
        # if bot_right_empty:
            # row_mask_bottom = self.notna().any(1)[::-1].cumsum()[::-1].astype(bool)
# 
        # row_mask = ~(~row_mask_top + ~row_mask_bottom)
        # self = self[row_mask]
# 
    # return self
