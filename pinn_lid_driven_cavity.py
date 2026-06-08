"""
Physics-Informed Neural Network (PINN) — 2D Lid-Driven Cavity
Solves incompressible Navier-Stokes at Re=100.

Governing equations:
  Continuity:   du/dx + dv/dy = 0
  x-momentum:   u·du/dx + v·du/dy = -dp/dx + (1/Re)(d²u/dx² + d²u/dy²)
  y-momentum:   u·dv/dx + v·dv/dy = -dp/dy + (1/Re)(d²v/dx² + d²v/dy²)

Boundary conditions (unit square [0,1]x[0,1]):
  Top wall    (y=1): u=1, v=0   ← moving lid
  Other walls      : u=0, v=0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" for live windows
import matplotlib.pyplot as plt

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── problem parameters ────────────────────────────────────────────────────────
Re = 100          # Reynolds number
EPOCHS = 3000
N_INTERIOR = 10000   # collocation (interior) points
N_BOUNDARY = 100     # points per wall segment
BC_WEIGHT = 5.0     # weight for boundary-condition loss term
LR = 1e-3


# ── neural network ────────────────────────────────────────────────────────────
class PINN(nn.Module):
    """Fully-connected network: (x,y) → (u, v, p)."""

    def __init__(self, n_hidden: int = 5, n_units: int = 64):
        super().__init__()
        sizes = [2] + [n_units] * n_hidden + [3]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._xavier_init()

    def _xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


# ── autograd helper ───────────────────────────────────────────────────────────
def _d(f: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        f, var,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
    )[0]


# ── PDE residuals ─────────────────────────────────────────────────────────────
def pde_residuals(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (continuity, x-momentum, y-momentum) residuals."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)

    out = model(x, y)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    u_x = _d(u, x);  u_y = _d(u, y)
    v_x = _d(v, x);  v_y = _d(v, y)
    p_x = _d(p, x);  p_y = _d(p, y)

    u_xx = _d(u_x, x);  u_yy = _d(u_y, y)
    v_xx = _d(v_x, x);  v_yy = _d(v_y, y)

    nu = 1.0 / Re
    cont  = u_x + v_y
    mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return cont, mom_x, mom_y


# ── boundary-condition loss ───────────────────────────────────────────────────
def bc_loss(model: nn.Module, n: int = N_BOUNDARY) -> torch.Tensor:
    def _loss(x_bc, y_bc, u_ref, v_ref):
        out = model(x_bc, y_bc)
        return ((out[:, 0:1] - u_ref) ** 2 + (out[:, 1:2] - v_ref) ** 2).mean()

    rnd = lambda: torch.rand(n, 1, device=DEVICE)
    zer = lambda: torch.zeros(n, 1, device=DEVICE)
    one = lambda: torch.ones(n, 1, device=DEVICE)

    loss  = _loss(rnd(), zer(), zer(), zer())   # bottom  y=0
    loss += _loss(rnd(), one(), one(), zer())   # top     y=1  (lid)
    loss += _loss(zer(), rnd(), zer(), zer())   # left    x=0
    loss += _loss(one(), rnd(), zer(), zer())   # right   x=1
    return loss


# ── model + optimiser ─────────────────────────────────────────────────────────
model = PINN(n_hidden=8, n_units=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9997)

# ── training loop ─────────────────────────────────────────────────────────────
hist_total, hist_pde, hist_bc = [], [], []

print(f"\nTraining for {EPOCHS} epochs  |  Re={Re}  |  interior pts={N_INTERIOR}")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    # --- PDE loss (interior) ---
    x_f = torch.rand(N_INTERIOR, 1, device=DEVICE)
    y_f = torch.rand(N_INTERIOR, 1, device=DEVICE)
    cont, mx, my = pde_residuals(model, x_f, y_f)
    loss_pde = cont.pow(2).mean() + mx.pow(2).mean() + my.pow(2).mean()

    # --- BC loss ---
    loss_bc = bc_loss(model)

    # --- pressure reference: pin p at (0,0)=0 to remove gauge freedom ---
    x0 = torch.zeros(1, 1, device=DEVICE)
    y0 = torch.zeros(1, 1, device=DEVICE)
    p0 = model(x0, y0)[:, 2:3]
    loss_pref = p0.pow(2).mean()

    loss = loss_pde + BC_WEIGHT * loss_bc + loss_pref

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    hist_total.append(loss.item())
    hist_pde.append(loss_pde.item())
    hist_bc.append(loss_bc.item())

    # --- console + loss-graph update every 100 epochs ---
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:6d}/{EPOCHS}  "
            f"total={loss.item():.3e}  "
            f"pde={loss_pde.item():.3e}  "
            f"bc={loss_bc.item():.3e}"
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        ep = range(1, epoch + 1)
        ax.semilogy(ep, hist_total, "k-",  lw=1.5, label="Total")
        ax.semilogy(ep, hist_pde,   "b--", lw=1.0, label="PDE")
        ax.semilogy(ep, hist_bc,    "r--", lw=1.0, label="BC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss  (epoch {epoch} / {EPOCHS})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_progress.png", dpi=100)
        plt.close()

        # ── L-BFGS fine-tuning phase ──────────────────────────────────────────────────
# Adam gets close; L-BFGS (quasi-Newton) uses curvature to drive PDE residual
# much lower. It needs a FIXED point set and a closure.
print("\nAdam phase done. Starting L-BFGS fine-tuning …")

x_f_fix = torch.rand(N_INTERIOR, 1, device=DEVICE)
y_f_fix = torch.rand(N_INTERIOR, 1, device=DEVICE)

_nb  = N_BOUNDARY
_r = lambda: torch.rand(_nb, 1, device=DEVICE)
_z = lambda: torch.zeros(_nb, 1, device=DEVICE)
_o = lambda: torch.ones(_nb, 1, device=DEVICE)
bc_pts = [
    (_r(), _z(), _z(), _z()),   # bottom  y=0
    (_r(), _o(), _o(), _z()),   # top     y=1  (lid)
    (_z(), _r(), _z(), _z()),   # left    x=0
    (_o(), _r(), _z(), _z()),   # right   x=1
]

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=5000,
    max_eval=5500,
    history_size=50,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe",
)

_it = [0]

def closure():
    optimizer_lbfgs.zero_grad()

    cont, mx, my = pde_residuals(model, x_f_fix, y_f_fix)
    loss_pde = cont.pow(2).mean() + mx.pow(2).mean() + my.pow(2).mean()

    loss_bc = 0.0
    for x_bc, y_bc, u_ref, v_ref in bc_pts:
        out = model(x_bc, y_bc)
        loss_bc = loss_bc + ((out[:, 0:1] - u_ref) ** 2
                             + (out[:, 1:2] - v_ref) ** 2).mean()

    p0 = model(torch.zeros(1, 1, device=DEVICE),
               torch.zeros(1, 1, device=DEVICE))[:, 2:3]
    loss_pref = p0.pow(2).mean()

    loss = loss_pde + BC_WEIGHT * loss_bc + loss_pref
    loss.backward()

    _it[0] += 1
    if _it[0] % 100 == 0:
        print(f"  L-BFGS iter {_it[0]:5d}  total={loss.item():.3e}  "
              f"pde={loss_pde.item():.3e}  bc={float(loss_bc):.3e}")
        hist_total.append(loss.item())
        hist_pde.append(loss_pde.item())
        hist_bc.append(float(loss_bc))
    return loss

model.train()
optimizer_lbfgs.step(closure)
print(f"L-BFGS fine-tuning complete ({_it[0]} iterations).")

print("\nTraining finished. Generating output figures …")


# ── figure 0 — domain collocation points ─────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

_xi = torch.rand(N_INTERIOR, 1).numpy()
_yi = torch.rand(N_INTERIOR, 1).numpy()

_n = N_BOUNDARY
_rnd = lambda: np.random.rand(_n, 1)
_zer = lambda: np.zeros((_n, 1))
_one = lambda: np.ones((_n, 1))

bc_bottom = (np.hstack([_rnd(), _zer()]), "Bottom wall  (u=0, v=0)")
bc_top    = (np.hstack([_rnd(), _one()]), "Top lid      (u=1, v=0)")
bc_left   = (np.hstack([_zer(), _rnd()]), "Left wall    (u=0, v=0)")
bc_right  = (np.hstack([_one(), _rnd()]), "Right wall   (u=0, v=0)")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(_xi, _yi, s=1.2, c="#4C72B0", alpha=0.25,
           label=f"Interior PDE pts ({N_INTERIOR:,})")

colors = ["#DD4949", "#E8963A", "#2CA02C", "#9467BD"]
for (pts, label), col in zip([bc_bottom, bc_top, bc_left, bc_right], colors):
    ax.scatter(pts[:, 0], pts[:, 1], s=10, c=col, zorder=3, label=label)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Collocation Points  (Re={Re})\n"
             f"Interior: {N_INTERIOR:,}   ·   BC per wall: {N_BOUNDARY}")
ax.legend(loc="upper left", markerscale=3, fontsize=8)
plt.tight_layout()
plt.savefig("collocation_points.png", dpi=150)
plt.close()
print("Saved: collocation_points.png")

# ── evaluation grid ───────────────────────────────────────────────────────────
model.eval()
N_GRID = 100
xi = np.linspace(0, 1, N_GRID)
yi = np.linspace(0, 1, N_GRID)
X, Y = np.meshgrid(xi, yi)          # shape (N_GRID, N_GRID)

xt = torch.tensor(X.ravel()[:, None], dtype=torch.float32, device=DEVICE)
yt = torch.tensor(Y.ravel()[:, None], dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    pred = model(xt, yt)

U = pred[:, 0].cpu().numpy().reshape(N_GRID, N_GRID)
V = pred[:, 1].cpu().numpy().reshape(N_GRID, N_GRID)
P = pred[:, 2].cpu().numpy().reshape(N_GRID, N_GRID)
P -= P.mean()          # zero-mean pressure
speed = np.sqrt(U ** 2 + V ** 2)


# ── figure 1 — contours ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
titles  = ["U velocity", "V velocity", "Pressure"]
fields  = [U, V, P]
cmaps   = ["RdBu_r", "RdBu_r", "RdBu_r"]

for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
    cf = ax.contourf(X, Y, field, levels=60, cmap=cmap)
    ax.contour(X, Y, field, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=ax, shrink=0.9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

fig.suptitle(f"Lid-Driven Cavity — PINN — Re = {Re}", fontsize=14)
plt.tight_layout()
plt.savefig("cavity_contours.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cavity_contours.png")


# ── figure 2 — velocity magnitude + streamlines ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5))
cf = ax.contourf(X, Y, speed, levels=60, cmap="viridis")
ax.streamplot(
    X, Y, U, V,
    density=1.8, color="white", linewidth=0.6, arrowsize=0.9,
)
plt.colorbar(cf, ax=ax, label="Speed |u|")
ax.set_title(f"Velocity Magnitude & Streamlines — Re = {Re}")
ax.set_xlabel("x");  ax.set_ylabel("y")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("streamlines.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: streamlines.png")


# ── figure 3 — centerline profiles (+ Ghia 1982 benchmark for Re=100) ─────────
# Ghia et al. (1982) tabulated data — Re = 100
ghia_u_y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                     0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                     0.0625, 0.0547, 0.0000])
ghia_u   = np.array([1.0000,  0.84123,  0.78871,  0.73722,  0.68717,  0.23151,
                     0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
                    -0.06434, -0.04775, -0.04192, -0.03717,  0.00000])

ghia_v_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                     0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                     0.0703, 0.0625, 0.0000])
ghia_v   = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                    -0.22445, -0.24533,  0.05454,  0.17527,  0.17507,  0.16077,
                     0.12003,  0.10945,  0.10090,  0.09233,  0.00000])

mid = N_GRID // 2    # index ≈ 0.5

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# U at x = 0.5  (vertical centre-line)
axes[0].plot(U[:, mid], yi, "b-", lw=2, label="PINN")
axes[0].plot(ghia_u, ghia_u_y, "ko", ms=5, label="Ghia et al. (1982)")
axes[0].axvline(0, color="gray", ls="--", lw=0.8)
axes[0].set_xlabel("U velocity")
axes[0].set_ylabel("y")
axes[0].set_title("U velocity at x = 0.5")
axes[0].legend()
axes[0].grid(True, alpha=0.4)

# V at y = 0.5  (horizontal centre-line)
axes[1].plot(xi, V[mid, :], "r-", lw=2, label="PINN")
axes[1].plot(ghia_v_x, ghia_v, "ko", ms=5, label="Ghia et al. (1982)")
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set_xlabel("x")
axes[1].set_ylabel("V velocity")
axes[1].set_title("V velocity at y = 0.5")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

fig.suptitle(f"Centerline Velocity Profiles — Re = {Re}", fontsize=13)
plt.tight_layout()
plt.savefig("velocity_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: velocity_profiles.png")


# ── figure 4 — final loss history ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(hist_total, "k-",  lw=1.5, label="Total")
ax.semilogy(hist_pde,   "b--", lw=1.0, label="PDE (NS residual)")
ax.semilogy(hist_bc,    "r--", lw=1.0, label="BC")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Full Training Loss History")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("final_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: final_loss.png")

print("\nAll done. Output files:")
print("  loss_progress.png   — loss updated every 100 epochs")
print("  final_loss.png      — complete loss history")
print("  cavity_contours.png — U, V, P contour plots")
print("  streamlines.png     — velocity magnitude + streamlines")
print("  velocity_profiles.png — centerline profiles vs Ghia (1982)")
