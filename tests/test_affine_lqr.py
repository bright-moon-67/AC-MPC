import torch

from antmaze_ac.control.steady_state_lqr import affine_lqr


def test_affine_cost_changes_feedforward_and_satisfies_bellman_linear_equation():
    dtype = torch.float64
    A = torch.tensor([[0.7]], dtype=dtype)
    B = torch.tensor([[0.5]], dtype=dtype)
    Q = torch.tensor([[2.0]], dtype=dtype)
    R = torch.tensor([[0.4]], dtype=dtype)
    q = torch.tensor([1.2], dtype=dtype)
    r = torch.tensor([-0.3], dtype=dtype)
    result = affine_lqr(A, B, Q, R, q, r, tolerance=1e-12, jitter=0.0)
    assert abs(result.feedforward) > 1e-4
    acl = A - B @ result.gain
    lhs = result.value_linear_half
    rhs = 0.5 * (q - result.gain.mT @ r) + acl.mT @ result.value_linear_half
    torch.testing.assert_close(lhs, rhs)
