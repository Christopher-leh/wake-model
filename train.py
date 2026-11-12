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
from src.data import WakeData2D

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starte Training auf Device: {device}")

    # 1. Konfiguration & Daten laden
    config = {
        "dataset": "wake_2d",  # Oder "wake_2d"
        "re": 40.0,
        "pde_weight": 0.0,
        "hidden_dim": 128,
        "num_layers": 4,
        "lr": 1e-3,
        "epochs": 5000,
        "batch_size_col": 256,
        "batch_size_data": None,  # None = alle Datenpunkte auf einmal verwenden
    }

    if config["dataset"] == "wake_2d":
        data = WakeData2D(
            data_dir="data", device=device, z_index=18, n_train=1500, n_col=2500
        )
        col_loader, data_loader = data.get_loaders(
            batch_size_col=config["batch_size_col"],
            batch_size_data=config["batch_size_data"],
        )

    # 2. Modell, Problem & Trainer zusammenstecken
    model = Pinn2D(
        in_dim=2,
        out_dim=3,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    problem = NS2DPINN(
        model=model,
        nu=1.0 / config["re"],
        pde_weight=config["pde_weight"],
        scales=data.scales,
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
        col_loader=col_loader,
        data_loader=data_loader,
        test_batch=data.test_batch,
        epochs=config["epochs"],
        log_every=100,
    )

    plot_loss_history(trainer.history, save_path="outputs/loss_curve.png")

    # 4. Bestes Modell aus Checkpoint wiederherstellen & auf feinem Gitter evaluieren
    trainer.load_checkpoint("best_model.pt", device)
    model.eval()

    with torch.no_grad():
        preds = model(data.test_batch["coords"]).cpu().numpy()

    pred_dict = {
        "u": preds[:, 0].reshape(data.X.shape),
        "v": preds[:, 1].reshape(data.Y.shape),
        "p": preds[:, 2].reshape(data.X.shape),
    }

    # 5. Ground Truth vs. Prediction vs. Residual plotten
    plot_comparison_and_residuals(
        data.X_phys,
        data.Y_phys,
        data.true_dict,
        pred_dict,
        save_path="outputs/pinn_evaluation_3x3.png",
    )
    # plot_3d_comparison(
    #    X, Y, true_dict, pred_dict, save_path="outputs/pinn_evaluation_3d.png"
    # )
