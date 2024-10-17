from action.matrix_action import MatrixAction
from torch import zeros, rand
import torch
### Test 1: set fields to zero should return zero action (wit non-zero coeffs)
### Test 2: set coeffs to zero should return zero action (with non-zero fields)

def test_scalar_matrix_zero_field():
    action_dict = dict(beta=1)
    # generate zero-valued fields
    cfgs = zeros(100,10,10)  # 0 axis is the batch axis ie number of cfgs
    result = MatrixAction(**action_dict)
    S = result.action(cfgs)
    for i in range(100):
        assert S[i] == 0

def test_scalar_matrix_zero_beta():
    action_dict = dict(beta=0)
    # generate zero-valued fields
    cfgs = rand(100,10,10)  # 0 axis is the batch axis ie number of cfgs
    result = MatrixAction(**action_dict)
    S = result.action(cfgs)
    for i in range(100):
        assert S[i] == 0


