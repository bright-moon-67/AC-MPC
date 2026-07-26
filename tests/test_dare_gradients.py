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
