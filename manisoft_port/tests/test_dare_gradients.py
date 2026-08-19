import torch

from antmaze_ac.control.differentiable_dare import solve_dare


def test_gradcheck_through_gain():
    dtype = torch.float64
    A = torch.tensor([[0.8]], dtype=dtype)
    B = torch.tensor([[0.4]], dtype=dtype)
    R = torch.tensor([[0.7]], dtype=dtype)

    def function(q):
        return solve_dare(
            A,
            B,
            q.reshape(1, 1),
            R,
            tolerance=1e-13,
            max_iterations=1000,
            jitter=0.0,
            check_stabilizable=False,
        ).gain

    q = torch.tensor([1.3], dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(function, (q,), eps=1e-6, atol=2e-4, rtol=2e-3)


def test_implicit_gradcheck_through_gain():
    dtype = torch.float64
    A = torch.tensor([[0.8]], dtype=dtype)
    B = torch.tensor([[0.4]], dtype=dtype)
    R = torch.tensor([[0.7]], dtype=dtype)

    def function(q):
        return solve_dare(
            A,
            B,
            q.reshape(1, 1),
            R,
            tolerance=1e-13,
            max_iterations=1000,
            jitter=0.0,
            check_stabilizable=False,
            implicit_backward=True,
            compute_closed_loop_spectral_radius=False,
        ).gain

    q = torch.tensor([1.3], dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(
        function,
        (q,),
        eps=1e-6,
        atol=2e-4,
        rtol=2e-3,
    )


def test_implicit_and_explicit_dare_values_and_gradients_match():
    dtype = torch.float64
    base = (
        torch.tensor([[0.78, 0.08], [-0.03, 0.72]], dtype=dtype),
        torch.tensor([[0.2], [0.35]], dtype=dtype),
        torch.tensor([[1.2, 0.1], [0.1, 0.8]], dtype=dtype),
        torch.tensor([[0.65]], dtype=dtype),
    )
    solved = []
    gradients = []
    for implicit_backward in (False, True):
        inputs = tuple(value.clone().requires_grad_() for value in base)
        result = solve_dare(
            *inputs,
            tolerance=1e-13,
            max_iterations=1000,
            jitter=0.0,
            check_stabilizable=False,
            check_detectable=False,
            implicit_backward=implicit_backward,
            compute_closed_loop_spectral_radius=False,
        )
        loss = (
            0.37 * result.P.square().sum()
            + 0.81 * result.gain.square().sum()
            + 0.13 * result.P.sum()
            + 0.21 * result.gain.sum()
        )
        solved.append(result)
        gradients.append(torch.autograd.grad(loss, inputs))

    torch.testing.assert_close(solved[0].P, solved[1].P)
    torch.testing.assert_close(solved[0].gain, solved[1].gain)
    for explicit_gradient, implicit_gradient in zip(
        gradients[0],
        gradients[1],
    ):
        torch.testing.assert_close(
            explicit_gradient,
            implicit_gradient,
            rtol=2e-9,
            atol=2e-10,
        )
