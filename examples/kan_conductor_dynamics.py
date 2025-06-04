import numpy as np
import torch
from kan import KAN
from kan.utils import create_dataset_from_data


def acceleration(qv, qvdot, qw, qwdot):
    """Compute vertical and horizontal accelerations."""
    qv_ddot = (
        0.0436 * qvdot
        - 0.1514 * qvdot ** 2
        - 0.0440 * qvdot ** 3
        - 5.8809 * qv
        - 0.3507 * qv ** 2
        - 0.0610 * qw ** 2
        - 0.0713 * qv ** 3
        - 0.0365 * qv * qw ** 2
    )
    qw_ddot = (
        -0.0024 * qwdot
        - 0.3364 * qvdot
        + 0.0438 * qvdot ** 2
        - 0.0132 * qvdot ** 3
        - 5.4665 * qw
        - 0.2442 * qv * qw
        - 0.0379 * qw ** 3
        - 0.0730 * qv ** 2 * qw
    )
    return qv_ddot, qw_ddot


def simulate(t_end=10.0, dt=0.01, state=None):
    """Integrate the dynamics with a simple Euler scheme."""
    if state is None:
        state = [0.0, 0.0, 0.0, 0.0]
    qv, qvdot, qw, qwdot = state
    times = np.arange(0.0, t_end + dt, dt)
    qv_list, qw_list = [], []
    for _ in times:
        qv_list.append(qv)
        qw_list.append(qw)
        qv_ddot, qw_ddot = acceleration(qv, qvdot, qw, qwdot)
        qv += dt * qvdot
        qvdot += dt * qv_ddot
        qw += dt * qwdot
        qwdot += dt * qw_ddot
    return times, np.array(qv_list), np.array(qw_list)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate training data from the ODE model
    t, qv, qw = simulate(t_end=6.0, dt=0.02, state=[0.0, 0.1, 0.0, 0.1])
    inputs = torch.tensor(t, dtype=torch.float64).unsqueeze(1)
    labels = torch.tensor(np.stack([qv, qw], axis=1), dtype=torch.float64)

    # scale time to [-1, 1] for training
    t_min, t_max = inputs.min(), inputs.max()
    inputs_scaled = 2 * (inputs - t_min) / (t_max - t_min) - 1

    dataset = create_dataset_from_data(inputs_scaled, labels, device=device)

    model = KAN(width=[1, 16, 16, 2], grid=5, k=3, seed=0, device=device)
    model.fit(dataset, opt="LBFGS", steps=100)

    # simple evaluation
    with torch.no_grad():
        pred = model(inputs_scaled.to(device)).cpu()
    print("True vs Pred (first 5 samples):")
    for i in range(5):
        print(f"t={t[i]:.2f}, qv={qv[i]:.4f}, qw={qw[i]:.4f} | pred=({pred[i,0]:.4f}, {pred[i,1]:.4f})")
