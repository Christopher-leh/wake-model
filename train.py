"""Dein kurzes Start-Skript."""

import numpy as np
import torch

from src.models import Pinn2D
from src.problems import NS2DPINN
from src.trainer import Trainer
from src.visualizer import (
    plot_3d_comparison,
    plot_comparison_and_residuals,
    plot_loss_history,
)
from src.data import load_kovasznay, get_wake_2d_data

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starte Training auf Device: {device}")

    # 1. Konfiguration & Daten laden
    config = {
        "dataset": "wake_2d",  # Oder "wake_2d"
        "re": 40.0,
        "pde_weight": 1.0,
        "hidden_dim": 64,
        "num_layers": 4,
        "lr": 1e-3,
        "epochs": 3000,
    }
    # 1.1 Daten laden
    if config["dataset"] == "kovasznay":
        train_batch, val_batch, (X, Y, true_dict) = load_kovasznay(
            re=config["re"], device=device
        )
    elif config["dataset"] == "wake_2d":
        train_batch, val_batch, (X, Y, true_dict) = get_wake_2d_data(
            data_dir="data/wake_2d", z_index=0, n_train=250, n_col=2500, device=device
        )
    else:
        raise ValueError(f"Unbekanntes Dataset: {config['dataset']}")

    # 2. Modell, Problem & Trainer zusammenstecken
    model = Pinn2D(
        in_dim=2,
        out_dim=3,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)
    problem = NS2DPINN(
        model=model, nu=1.0 / config["re"], pde_weight=config["pde_weight"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    trainer = Trainer(
        problem=problem,
        optimizer=optimizer,
        checkpoint_dir="checkpoints",
        config=config,
    )

    # 3. Training starten
    trainer.fit(
        train_batch=train_batch,
        val_batch=val_batch,
        epochs=config["epochs"],
        log_every=500,
    )
    plot_loss_history(trainer.history, save_path="outputs/loss_curve.png")

    # 4. Bestes Modell aus Checkpoint wiederherstellen & auf feinem Gitter evaluieren
    trainer.load_checkpoint("best_model.pt")
    model.eval()

    with torch.no_grad():
        xy_tensor = torch.tensor(
            np.stack([X.ravel(), Y.ravel()], axis=-1),
            dtype=torch.float32,
            device=device,
        )
        preds = model(xy_tensor).cpu().numpy()

    pred_dict = {
        "u": preds[:, 0].reshape(X.shape),
        "v": preds[:, 1].reshape(X.shape),
        "p": preds[:, 2].reshape(X.shape),
    }

    # 5. Ground Truth vs. Prediction vs. Residual plotten
    plot_comparison_and_residuals(
        X, Y, true_dict, pred_dict, save_path="outputs/pinn_evaluation_3x3.png"
    )
    plot_3d_comparison(
        X, Y, true_dict, pred_dict, save_path="outputs/pinn_evaluation_3d.png"
    )
