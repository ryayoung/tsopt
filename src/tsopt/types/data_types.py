# Maintainer:     Ryan Young
# Last Modified:  Dec 06, 2022
import pandas as pd, numpy as np
from dataclasses import dataclass

# class NodeSR(pd.Series):
    # def _repr_html_(self):
        # df = pd.DataFrame(self, columns=[self.name if self.name else ""])
        # return df._repr_html_()
# 
    # def pop(self, n=1):
        # updated = self.iloc[:-n]
        # return NodeSR(updated)
# 
    # def push(self, abbrev, n=1, fill_val=np.nan):
        # new = pd.Series(fill_val, index=[f'{abbrev}{i+self.shape.rows}' for i in range(n)])
        # updated = pd.concat([self, new])
        # return NodeSR(updated)


# class NodeBoundsDF(pd.DataFrame):
    # pass


# class EdgeDF(pd.DataFrame):
# 
    # def pop(self, n=1, axis=0):
        # if axis == 0:
            # updated = self.iloc[:-n]
        # elif axis == 1:
            # updated = self.iloc[:, :-n]
        # return EdgeDF(updated)
# 
    # def push(self, abbrev, n=1, axis=0, fill_val=np.nan):
        # idxs = [f'{abbrev}{i+self.shape.rows}' for i in range(n)] if axis == 0 else self.index
        # cols = [f'{abbrev}{i+self.shape.cols}' for i in range(n)] if axis == 1 else self.columns
        # new = pd.DataFrame(fill_val, index=idxs, columns=cols)
        # updated = pd.concat([self, new], axis=axis)
        # return EdgeDF(updated)


# class EdgeMeltedDF(pd.DataFrame):
    # pass
# 
# class EdgeMeltedBoundsDF(pd.DataFrame):
    # pass



