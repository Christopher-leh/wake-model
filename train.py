"""Dein kurzes Start-Skript."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.models import Pinn2D
from src.problems import NS2DPINN, NS2DtimePINN
from src.trainer import Trainer
from src.visualizer import (
    plot_3d_comparison,
    plot_comparison_and_residuals,
    plot_loss_history,
    plot_time_slices,
)
from src.data import WakeData2D, WakeData2Dtime

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starte Training auf Device: {device}")

    # 1. Konfiguration & Daten laden
    config = {
        "dataset": "wake_2dtime",  # Oder "wake_2d"
        "nu": 1.5e-5,
        "pde_weight": 100.0,
        "hidden_dim": 128,
        "num_layers": 4,
        "lr": 1e-3,
        "epochs": 10,
        "batch_size_col": 256,
        "batch_size_data": None,  # None = alle Datenpunkte auf einmal verwenden
        "seed": 42,
        "z_index": 17,
        "n_train": 1500,
        "n_col": 2500,
    }

    rng = np.random.default_rng(config["seed"])
    torch.manual_seed(config["seed"])

    # Ein Ordner pro Lauf: Modell, Config und Plots liegen zusammen, nichts wird überschrieben
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{timestamp}_{config['dataset']}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Ergebnisse werden gespeichert in: {run_dir}")

    if config["dataset"] == "wake_2d":
        data = WakeData2D(
            data_dir="data",
            device=device,
            z_index=config["z_index"],
            n_train=config["n_train"],
            n_col=config["n_col"],
            rng=rng,
        )
        col_loader, data_loader = data.get_loaders(
            batch_size_col=config["batch_size_col"],
            batch_size_data=config["batch_size_data"],
        )
    elif config["dataset"] == "wake_2dtime":
        data = WakeData2Dtime(
            data_dir="data",
            device=device,
            z_index=config["z_index"],
            n_train=config["n_train"],
            n_col=config["n_col"],
            rng=rng,
        )
        col_loader, data_loader = data.get_loaders(
            batch_size_col=config["batch_size_col"],
            batch_size_data=config["batch_size_data"],
        )

    # 2. Modell, Problem & Trainer zusammenstecken
    model = Pinn2D(
        in_dim=data.coords_col.shape[1],
        out_dim=3,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    if config["dataset"] == "wake_2d":
        problem = NS2DPINN(
            model=model,
            nu=config["nu"],
            pde_weight=config["pde_weight"],
            scales=data.scales,
        )
    elif config["dataset"] == "wake_2dtime":
        problem = NS2DtimePINN(
            model=model,
            nu=config["nu"],
            pde_weight=config["pde_weight"],
            scales=data.scales,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    trainer = Trainer(
        problem=problem,
        optimizer=optimizer,
        checkpoint_dir=run_dir,
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

    # 4. Finales Modell, Loss-Verlauf und Plots im Run-Ordner speichern
    trainer.save_checkpoint("model.pt", config["epochs"], trainer.history["total"][-1])
    (run_dir / "history.json").write_text(json.dumps(trainer.history))
    plot_loss_history(trainer.history, save_path=run_dir / "loss_curve.png")
    model.eval()

    if config["dataset"] == "wake_2dtime":
        n_t = data.X.shape[2]
        plot_time_slices(
            model,
            data,
            [0, n_t // 2, n_t - 1],
            device=device,
            save_prefix=run_dir / "pinn_eval",
        )

    elif config["dataset"] == "wake_2d":
        coords_col = data.test_batch["coords"]
        with torch.no_grad():
            with torch.no_grad():
                preds = model(data.test_batch["coords"]).cpu().numpy()

            pred_dict = {
                "u": preds[:, 0].reshape(data.X.shape),
                "v": preds[:, 1].reshape(data.Y.shape),
                "p": preds[:, 2].reshape(data.X.shape),
            }

        plot_comparison_and_residuals(
            data.X_phys,
            data.Y_phys,
            data.true_dict,
            pred_dict,
            save_path=run_dir / "pinn_evaluation_3x3.png",
        )
