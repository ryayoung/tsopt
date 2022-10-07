# Maintainer:     Ryan Young
# Last Modified:  Oct 04, 2022

def smart_width(size:tuple, rows) -> tuple:
    '''
    Changes first element of size tuple based on num of rows.
    Only has effect when number of rows is less than 3
    '''
    w, h = size
    if rows < 5:
        w = w // 1.5 + 1
    if rows < 3:
        w = w // 2 + 1
    return (w, h)

