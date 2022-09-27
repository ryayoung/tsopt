# Maintainer:     Ryan Young
# Last Modified:  Aug 20, 2022

class InfeasibleConstraint(Exception):
    pass

class InfeasibleLayerConstraint(Exception):
    pass

class InfeasibleEdgeConstraint(Exception):
    pass

class InvalidConstraintData(Exception):
    pass
