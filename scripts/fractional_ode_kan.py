"""Solve a fractional ODE using KAN and derive a symbolic form.

This script trains a simple 1D Kolmogorov--Arnold Network on a fractional
ordinary differential equation (ODE).  After fitting the network we call
``auto_symbolic`` to convert the learned splines to closed form symbolic
expressions and continue training with those expressions fixed.

The default equation solved here is

.. math:: \partial_t^\alpha u + u = f(t), \qquad u(0)=0,

with an analytic solution :math:`u(t)=t^{5+\alpha}` when
``f`` is chosen appropriately.  The order ``alpha`` as well as the time
grid can be changed from the command line.
"""

import argparse
import torch
from kan import KAN, LBFGS
from tqdm import tqdm
import matplotlib.pyplot as plt


def a(l, alpha):
    """Coefficient helper used in the Grunwald approximation."""
    return (l + 1) ** (1 - alpha) - l ** (1 - alpha)


def forcing(t, alpha):
    """Right-hand side ``f(t)`` yielding ``u(t)=t**(5+alpha)``."""
    gamma_part = torch.lgamma(torch.tensor(6 + alpha)).exp() / 120
    return gamma_part * t ** 5 + t ** (5 + alpha)


def loss_fn(model, t, tau, alpha, nt):
    """Compute the fractional ODE residual loss."""
    u_pred = model(t).view(-1)
    u0 = torch.tensor(0.0)

    ic_loss = (u_pred[0] - u0) ** 2
    eq_loss = 0.0
    for n in range(1, nt):
        t_n = t[n]
        k = torch.arange(1, n)
        history = torch.sum(
            (a(n - k - 1, alpha) - a(n - k, alpha)) * u_pred[k]
        )
        coeff = tau ** -alpha / torch.lgamma(torch.tensor(2 - alpha)).exp()
        d_alpha_u = coeff * (u_pred[n] * a(0, alpha) - history - a(n - 1, alpha) * u0)
        eq_loss += (d_alpha_u + u_pred[n] - forcing(t_n, alpha)) ** 2
    return ic_loss + eq_loss


def train(model, t, tau, alpha, nt, steps=500, log=100):
    """Minimal LBFGS training loop."""
    opt = LBFGS(model.parameters(), lr=0.1, history_size=10, line_search_fn="strong_wolfe")
    pbar = tqdm(range(steps), desc="train")
    for i in pbar:
        def closure():
            opt.zero_grad()
            loss = loss_fn(model, t, tau, alpha, nt)
            loss.backward()
            return loss

        opt.step(closure)
        if i % log == 0:
            with torch.no_grad():
                l = loss_fn(model, t, tau, alpha, nt)
            pbar.set_description(f"loss={l.item():.2e}")


def main():
    parser = argparse.ArgumentParser(description="Fractional ODE with KAN")
    parser.add_argument("--T", type=float, default=1.4, help="end time")
    parser.add_argument("--nt", type=int, default=20, help="number of time steps")
    parser.add_argument("--alpha", type=float, default=0.5, help="fractional order")
    parser.add_argument("--steps", type=int, default=500, help="training steps")
    parser.add_argument(
        "--symbolic_steps",
        type=int,
        default=200,
        help="additional steps after auto_symbolic",
    )
    args = parser.parse_args()

    tau = args.T / args.nt
    t = torch.linspace(0, args.T, steps=args.nt).view(-1, 1)

    model = KAN(width=[1, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)
    model.speed()  # disable symbolic bookkeeping during the numeric phase

    train(model, t, tau, args.alpha, args.nt, steps=args.steps)

    # switch on symbolic branch and search for analytic forms
    model.symbolic_enabled = True
    model.auto_symbolic()

    # fine tune with symbolic edges fixed
    train(model, t, tau, args.alpha, args.nt, steps=args.symbolic_steps)

    formulas, _ = model.symbolic_formula()
    print("Symbolic formula:")
    for i, expr in enumerate(formulas[0]):
        print(f"u_{i} = {expr}")

    # plot prediction against the analytic solution
    t_test = torch.linspace(0, args.T, steps=100).view(-1, 1)
    u_pred = model(t_test).detach().numpy()
    u_true = t_test.numpy() ** (5 + args.alpha)
    plt.plot(t_test, u_pred, label="Pred")
    plt.plot(t_test, u_true, "--", label="True")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
