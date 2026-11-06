"""Dein kurzes Start-Skript."""
import numpy as np
import torch

from src.models import Pinn2D
from src.problems import NS2DPINN
from src.trainer import Trainer
from src.visualizer import plot_3d_comparison, plot_comparison_and_residuals, plot_loss_history


def generate_kovasznay_batch(n_col: int = 2500, n_data: int = 250, re: float = 40.0, device: str = "cpu"):
    """Erzeugt das Trainings-Dictionary (Batch) und gibt die exakte Lösungsfunktion zurück."""
    nu = 1.0 / re
    lam = 0.5 * re - np.sqrt(0.25 * (re**2) + 4.0 * (np.pi**2))

    def exact_fn(x: np.ndarray, y: np.ndarray):
        u = 1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y)
        v = (lam / (2.0 * np.pi)) * np.exp(lam * x) * np.sin(2.0 * np.pi * y)
        p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
        return u, v, p

    # Kollokationspunkte im Gebiet + Randpunkte für den Data-Loss
    x_col = np.random.uniform(-0.5, 1.0, (n_col, 1))
    y_col = np.random.uniform(-0.5, 1.5, (n_col, 1))

    x_dat = np.random.uniform(-0.5, 1.0, (n_data, 1))
    y_dat = np.random.uniform(-0.5, 1.5, (n_data, 1))
    u_dat, v_dat, _ = exact_fn(x_dat, y_dat)

    batch = {
        "x_col": torch.tensor(x_col, dtype=torch.float32, device=device, requires_grad=True),
        "y_col": torch.tensor(y_col, dtype=torch.float32, device=device, requires_grad=True),
        "x_data": torch.tensor(x_dat, dtype=torch.float32, device=device),
        "y_data": torch.tensor(y_dat, dtype=torch.float32, device=device),
        "uv_data": torch.tensor(np.hstack([u_dat, v_dat]), dtype=torch.float32, device=device),
    }
    return batch, nu, exact_fn


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starte Training auf Device: {device}")

    # 1. Konfiguration & Daten
    config = {"re": 40.0, "hidden_dim": 64, "num_layers": 4, "lr": 1e-3, "epochs": 3000}
    train_batch, nu, exact_fn = generate_kovasznay_batch(re=config["re"], device=device)

    #val_batch als gitter
    x_lin = np.linspace(-0.5, 1.0, 50)
    y_lin = np.linspace(-0.5, 1.5, 50)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    u_true, v_true, p_true = exact_fn(X, Y)
    val_batch = {
        "coords": torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32, device=device),
        "u_true": torch.tensor(np.stack([u_true.ravel(), v_true.ravel(), p_true.ravel()], axis=-1), dtype=torch.float32, device=device),
    }

    # 2. Modell, Problem & Trainer zusammenstecken
    model = Pinn2D(in_dim=2, out_dim=3, hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]).to(device)
    problem = NS2DPINN(model=model, nu=nu, pde_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    trainer = Trainer(problem=problem, optimizer=optimizer, checkpoint_dir="checkpoints", config=config)

    # 3. Training starten
    trainer.fit(train_batch=train_batch, val_batch=val_batch, epochs=config["epochs"], log_every=500)
    plot_loss_history(trainer.history)

    # 4. Bestes Modell aus Checkpoint wiederherstellen & auf feinem Gitter evaluieren
    trainer.load_checkpoint("best_model.pt")
    model.eval()

    x_lin = np.linspace(-0.5, 1.0, 120)
    y_lin = np.linspace(-0.5, 1.5, 120)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    u_true, v_true, p_true = exact_fn(X, Y)

    with torch.no_grad():
        xy_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32, device=device)
        preds = model(xy_tensor).cpu().numpy()

    true_dict = {"u": u_true, "v": v_true, "p": p_true}
    pred_dict = {
        "u": preds[:, 0].reshape(X.shape),
        "v": preds[:, 1].reshape(X.shape),
        "p": preds[:, 2].reshape(X.shape),
    }

    # 5. Ground Truth vs. Prediction vs. Residual plotten
    plot_comparison_and_residuals(X, Y, true_dict, pred_dict)
    plot_3d_comparison(X, Y, true_dict, pred_dict)