import torch
from kan import KAN, LBFGS
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters
T = 1.4
nt = 20
alpha = 0.5
tau = T/nt

def a(l, alpha):
    return (l + 1)**(1 - alpha) - l**(1 - alpha)

# forcing term f(t)
def f(t, alpha):
    gamma_part = torch.lgamma(torch.tensor(6 + alpha)).exp() / 120
    return gamma_part * t**5 + t**(5 + alpha)

# build training points
t = torch.linspace(0, T, steps=nt).view(-1, 1)

# create KAN model
model = KAN(width=[1, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)
model.speed()  # disable slow symbolic bookkeeping during training

# loss for fractional ODE
def loss_fn(model, t, tau, alpha):
    u_pred = model(t).view(-1)
    u0 = torch.tensor(0.0)
    ic_loss = (u_pred[0] - u0)**2
    eq_loss = 0.0
    for n in range(1, nt):
        t_n = t[n]
        history = sum((a(n - k - 1, alpha) - a(n - k, alpha)) * u_pred[k] for k in range(1, n))
        D_alpha_u = tau**-alpha / torch.lgamma(torch.tensor(2 - alpha)).exp() * (
            u_pred[n] * a(0, alpha) - history - a(n - 1, alpha) * u0)
        eq_loss += (D_alpha_u + u_pred[n] - f(t_n, alpha))**2
    return ic_loss + eq_loss

# training loop
def train(model, steps=500, log=100):
    opt = LBFGS(model.parameters(), lr=0.1, history_size=10, line_search_fn="strong_wolfe")
    pbar = tqdm(range(steps), desc="train")
    for i in pbar:
        def closure():
            opt.zero_grad()
            loss = loss_fn(model, t, tau, alpha)
            loss.backward()
            return loss
        opt.step(closure)
        if i % log == 0:
            with torch.no_grad():
                l = loss_fn(model, t, tau, alpha)
            pbar.set_description(f"loss={l.item():.2e}")

if __name__ == "__main__":
    train(model)
    # enable symbolic functions and run automatic symbolic regression
    model.symbolic_enabled = True
    model.auto_symbolic()
    # retrain with symbolic edges fixed
    train(model, steps=200)
    formulas, _ = model.symbolic_formula()
    print("Symbolic formula:")
    for idx, expr in enumerate(formulas[0]):
        print(f"u_{idx} = {expr}")

    # visualize prediction
    t_test = torch.linspace(0, T, steps=100).view(-1,1)
    u_pred = model(t_test).detach().numpy()
    u_true = t_test.numpy()**(5 + alpha)
    plt.plot(t_test, u_pred, label="Pred")
    plt.plot(t_test, u_true, '--', label="True")
    plt.xlabel('t'); plt.ylabel('u(t)')
    plt.legend()
    plt.show()
