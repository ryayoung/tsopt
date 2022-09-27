# Maintainer:     Ryan Young
# Last Modified:  Jul 28, 2022

import textwrap
import re

def plural(word:str) -> str:
    if word.endswith('y'):
        return word[:-1] + 'ies'
    if word.endswith('s'):
        return word + 'es'
    return word + 's'


def comma_sep(iterable, max_len=5) -> str:
    if len(iterable) == 1:
        return str(iterable[0])
    elif len(iterable) <= max_len:
        main = iterable[:-1]
        last = iterable[-1]
        return ', '.join(main) + f' and {last}'
    else:
        return ', '.join(iterable[:max_len]) + f' (+{len(iterable)-max_len} more)'


def dedent_wrap(s, prefix_mask='InfeasibleEdgeConstraint: ') -> str:
    s = prefix_mask + ' '.join(s.split())
    wraps = textwrap.wrap(textwrap.dedent(s.strip()), width=80)
    wraps[0] = wraps[0].replace(prefix_mask, '')
    return '\n'.join(wraps)
