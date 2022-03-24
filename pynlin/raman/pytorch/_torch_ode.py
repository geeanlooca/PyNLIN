import torch


def torch_rk4(func, y0, t, *args, **kwargs):
    """Integrate ODEs with a fourth-order fixed-step Runge-Kutta solver.

    Params
    ------
    func : callable
        A function describing the differential equation.
    y0 : torch.Tensor
        The initial conditions.
    t : torch.Tensor
        The evaluation points.

    Returns
    -------
    torch.Tensor
        The solution at the last evaluation point.
    """
    y = y0.clone()

    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k1 = h * func(y, t[i - 1], *args, **kwargs)
        k2 = h * func(y + k1 / 2, t[i - 1] + h / 2, *args, **kwargs)
        k3 = h * func(y + k2 / 2, t[i - 1] + h / 2, *args, **kwargs)
        k4 = h * func(y + k3, t[i - 1] + h, *args, **kwargs)

        new_state = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if torch.isnan(new_state).any() or torch.isinf(new_state).any():

            return y

        y = new_state

    return y


def rk4_step(func, y0, t0, stepsize, *args, **kwargs):
    """A single step of a fourth-order Runge-Kutta solver."""
    h = stepsize
    k1 = h * func(y0, t0, *args, **kwargs)
    k2 = h * func(y0 + k1 / 2, t0 + h / 2, *args, **kwargs)
    k3 = h * func(y0 + k2 / 2, t0 + h / 2, *args, **kwargs)
    k4 = h * func(y0 + k3, t0 + h, *args, **kwargs)
    return y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
