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


def test_affine_lqr_q_r_gradients_match_with_implicit_dare_backward():
    dtype = torch.float64
    A = torch.tensor([[0.72, 0.04], [0.0, 0.81]], dtype=dtype)
    B = torch.tensor([[0.3], [0.2]], dtype=dtype)
    Q = torch.tensor([[1.4, 0.1], [0.1, 0.9]], dtype=dtype)
    R = torch.tensor([[0.6]], dtype=dtype)
    base_q = torch.tensor([0.3, -0.4], dtype=dtype)
    base_r = torch.tensor([0.2], dtype=dtype)
    gradients = []
    outputs = []
    for implicit_backward in (False, True):
        q = base_q.clone().requires_grad_()
        r = base_r.clone().requires_grad_()
        result = affine_lqr(
            A,
            B,
            Q,
            R,
            q,
            r,
            tolerance=1e-13,
            max_iterations=1000,
            jitter=0.0,
            check_stabilizable=False,
            check_detectable=False,
            implicit_backward=implicit_backward,
            compute_closed_loop_spectral_radius=False,
        )
        loss = (
            result.feedforward.square().sum()
            + 0.2 * result.value_linear_half.square().sum()
        )
        gradients.append(torch.autograd.grad(loss, (q, r)))
        outputs.append(result)

    torch.testing.assert_close(outputs[0].feedforward, outputs[1].feedforward)
    for explicit_gradient, implicit_gradient in zip(
        gradients[0],
        gradients[1],
    ):
        torch.testing.assert_close(explicit_gradient, implicit_gradient)
