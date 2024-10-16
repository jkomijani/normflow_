from normflow.action import ScalarPhi4Action
from torch import zeros, rand

### Test 1: set fields to zero should return zero action (wit non-zero coeffs)
### Test 2: set coeffs to zero should return zero action (with non-zero fields)

def test_scalar_phi4_action_zero():
    action_dict = dict(kappa=1, m_sq=1, lambd=1)
    # generate zero-valued fields
    cfgs = zeros(100,10,10)  # 0 axis is the batch axis ie number of cfgs
    result = ScalarPhi4Action(**action_dict)
    S = result.action(cfgs)
    for i in range(100):
        assert S[i] == 0

def test_scalar_phi4_action_zero_coeff():
    action_dict = dict(kappa=0, m_sq=0, lambd=0)
    # generate random non-zero cfgs
    cfgs = rand(100,10,10)  # 0 axis is the batch axis ie number of cfgs
    result = ScalarPhi4Action(**action_dict)
    S = result.action(cfgs)
    for i in range(100):
        assert S[i] == 0


