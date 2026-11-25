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
        "dataset": "wake_2dtime",  # Oder "wake_2d" (Mean Flow)
        "nu": 0.0,  # macht keinen unterschied
        "data_weight": 1.0,
        "pde_weight_imp": 1.0,  # bei wake_2d: Gewicht des gesamten PDE-Loss
        "pde_weight_cont": 1.0,  # nur wake_2dtime
        "hidden_dim": 128,
        "num_layers": 8,
        "lr": 1e-3,
        "lr_end": 1e-5,
        "epochs": 20000,
        "batch_size_col": 4000,
        "batch_size_data": None,  # None = alle Datenpunkte auf einmal verwenden
        "seed": 42,
        "z_index": 17,
        "n_train": 5000,
        "n_col": 100000,
        "cone_angle": np.pi / 8,  # Beispiel: 45 Grad Kegelwinkel
        "radius": 50,  # Beispiel: Radius des Kegels (920 m in x und etwa 985 m in y.)
        "sampling_near_turbine": True,  # Wenn True, werden die Trainingspunkte in der Nähe der Turbine gesampelt, sonst zufällig im gesamten Bereich
        "monitor": "val_nrmse_uv",  # Validierungsmetrik für das beste Modell
    }

    rng = np.random.default_rng(config["seed"])
    torch.manual_seed(config["seed"])

    # Ein Ordner pro Lauf: Modell, Config und Plots liegen zusammen, nichts wird überschrieben
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{timestamp}_{config['dataset']}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Ergebnisse werden gespeichert in: {run_dir}")

    # 2. Daten + Problem je nach Datensatz
    model_kwargs = dict(out_dim=3, hidden_dim=config["hidden_dim"], num_layers=config["num_layers"])

    if config["dataset"] == "wake_2d":
        data = WakeData2D(
            data_dir="data",
            device=device,
            z_index=config["z_index"],
            n_train=config["n_train"],
            n_col=config["n_col"],
            rng=rng,
        )
        model = Pinn2D(in_dim=2, **model_kwargs).to(device)
        problem = NS2DPINN(
            model=model,
            nu=config["nu"],
            data_weight=config["data_weight"],
            pde_weight=config["pde_weight_imp"],
            scales=data.scales,
        )
    elif config["dataset"] == "wake_2dtime":
        data = WakeData2Dtime(
            data_dir="data",
            z_index=config["z_index"],
            n_train=config["n_train"],
            n_col=config["n_col"],
            device=device,
            rng=rng,
            cone_angle=config["cone_angle"],
            radius=config["radius"],
            sampling_near_turbine=config["sampling_near_turbine"],
        )
        model = Pinn2D(in_dim=3, **model_kwargs).to(device)
        problem = NS2DtimePINN(
            model=model,
            nu=config["nu"],
            data_weight=config["data_weight"],
            pde_weight_imp=config["pde_weight_imp"],
            pde_weight_cont=config["pde_weight_cont"],
            scales=data.scales,
        )
    else:
        raise ValueError(f"Unbekannter Datensatz: {config['dataset']}")

    col_loader, data_loader = data.get_loaders(
        batch_size_col=config["batch_size_col"],
        batch_size_data=config["batch_size_data"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(config["lr_end"] / config["lr"]) ** (1 / config["epochs"])
    )

    trainer = Trainer(
        problem=problem,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=run_dir,
        config=config,
    )

    # 3. Training starten (Validierung wählt das beste Modell)
    trainer.fit(
        col_loader=col_loader,
        data_loader=data_loader,
        val_batch=data.val_batch,
        epochs=config["epochs"],
        log_every=100,
        monitor=config["monitor"],
    )

    # 4. Letztes Modell und Verläufe speichern
    trainer.save_checkpoint("model.pt", trainer.last_epoch, trainer.history["total"][-1])
    (run_dir / "history.json").write_text(json.dumps({
        "train": trainer.history,
        "val": trainer.val_history,
        "lr": trainer.lr_history,
        "last_epoch": trainer.last_epoch,
        "stopped_early": trainer.stopped_early,
    }))
    plot_loss_history(trainer.history, save_path=run_dir / "loss_curve.png")

    # 5. Bestes Modell laden und EINMAL auf dem Test-Batch auswerten
    best = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    test_metrics = problem.evaluate(data.test_batch)
    print(f"\nTEST (bestes Modell, Epoch {best['epoch']}):")
    for kind in ["nrmse", "l2"]:
        print(f"  {kind:5s} " + "  ".join(f"{n} {test_metrics[f'val_{kind}_{n}']:.2%}" for n in "uvp"))
    (run_dir / "test_metrics.json").write_text(json.dumps({
        "test_metrics": test_metrics,
        "best_epoch": best["epoch"],
        "best_val_metric": trainer.best_metric,
        "monitor": config["monitor"],
    }, indent=2))

    # 6. Plots mit dem besten Modell
    if config["dataset"] == "wake_2dtime":
        plot_time_slices(
            model,
            data,
            [data.train_t[0], data.val_t[2], data.test_t[4]],  # je ein Train-, Val-, Test-Zeitschritt
            device=device,
            save_prefix=run_dir / "pinn_eval",
        )

    elif config["dataset"] == "wake_2d":
        with torch.no_grad():
            preds = model(data.test_batch["coords"]).cpu().numpy()
        pred_dict = {name: preds[:, i].reshape(data.X.shape) for i, name in enumerate(["u", "v", "p"])}

        plot_comparison_and_residuals(
            data.X_phys,
            data.Y_phys,
            data.true_dict,
            pred_dict,
            save_path=run_dir / "pinn_evaluation_3x3.png",
        )
