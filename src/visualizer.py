from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(history: Dict[str, List[float]], save_path: str = "loss_curve.png") -> None:
    plt.figure(figsize=(8, 4.5))
    for key, values in history.items():
        plt.semilogy(values, label=f"Loss: {key}", lw=1.8)

    plt.title("Trainings-Konvergenz (Log-Skala)")
    plt.xlabel("Epoche")
    plt.ylabel("MSE Loss (log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss-Kurve gespeichert unter: '{save_path}'")
    plt.show()


def plot_comparison_and_residuals(
        X: np.ndarray,
        Y: np.ndarray,
        true_fields : Dict[str,np.ndarray],
        pred_fields : Dict[str,np.ndarray],
        save_path : str = "pinn_evaluation_3x3.png"
) -> None:

    variables = ["u", "v", "p"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)

    for row_idx, var in enumerate(variables):
        u_true = true_fields[var].copy()
        u_pred = pred_fields[var].copy()

        # Kritisch für Navier-Stokes: Druck p ist nur bis auf eine Konstante eindeutig!
        if var == "p":
            u_true -= np.mean(u_true)
            u_pred -= np.mean(u_pred)

        # Fehler-Feld (Residual / Pointwise Error) berechnen
        err = np.abs(u_pred - u_true)
        rel_l2 = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12)

        # Gemeinsame Farbskala für Ground Truth und Prediction erzwingen!
        vmin, vmax = np.min(u_true), np.max(u_true)

        # 1. Spalte: Ground Truth
        im0 = axes[row_idx, 0].contourf(X, Y, u_true, levels=30, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_title(f"Ground Truth: {var}(x,y)")
        fig.colorbar(im0, ax=axes[row_idx, 0])

        # 2. Spalte: PINN Prediction
        im1 = axes[row_idx, 1].contourf(X, Y, u_pred, levels=30, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row_idx, 1].set_title(f"PINN Prediction: {var}(x,y)")
        fig.colorbar(im1, ax=axes[row_idx, 1])

        # 3. Spalte: Residual / Absoluter Fehler (eigene Farbskala 'magma' oder 'inferno')
        im2 = axes[row_idx, 2].contourf(X, Y, err, levels=30, cmap="inferno")
        axes[row_idx, 2].set_title(f"Absoluter Fehler |{var}_pred - {var}_true| (Rel. L2: {rel_l2:.2%})")
        fig.colorbar(im2, ax=axes[row_idx, 2])

        for col_idx in range(3):
            axes[row_idx, col_idx].set_xlabel("x")
            axes[row_idx, col_idx].set_ylabel("y")

    plt.savefig(save_path, dpi=150)
    print(f"3x3 Evaluierungs-Plot gespeichert unter: '{save_path}'")
    plt.show()


def plot_3d_comparison(
        X: np.ndarray,
        Y: np.ndarray,
        true_fields: Dict[str, np.ndarray],
        pred_fields: Dict[str, np.ndarray],
        save_path: str = "pinn_evaluation_3d.png",
) -> None:
    """Erzeugt einen 3D-Oberflächenvergleich für die Felder u, v und p."""
    variables = ["u", "v", "p"]
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)

    for row_idx, var in enumerate(variables):
        true_field = true_fields[var].copy()
        pred_field = pred_fields[var].copy()

        if var == "p":
            true_field -= np.mean(true_field)
            pred_field -= np.mean(pred_field)

        error = np.abs(pred_field - true_field)
        rel_l2 = np.linalg.norm(pred_field - true_field) / (np.linalg.norm(true_field) + 1e-12)
        vmin, vmax = np.min(true_field), np.max(true_field)

        surfaces = (
            (true_field, "viridis", vmin, vmax, f"Ground Truth: {var}(x,y)"),
            (pred_field, "viridis", vmin, vmax, f"PINN Prediction: {var}(x,y)"),
            (error, "inferno", None, None, f"Absoluter Fehler {var} (Rel. L2: {rel_l2:.2%})"),
        )

        for col_idx, (field, cmap, field_min, field_max, title) in enumerate(surfaces):
            ax = fig.add_subplot(3, 3, row_idx * 3 + col_idx + 1, projection="3d")
            surface = ax.plot_surface(
                X,
                Y,
                field,
                cmap=cmap,
                vmin=field_min,
                vmax=field_max,
                linewidth=0,
                antialiased=True,
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel(var)
            fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)

    fig.savefig(save_path, dpi=150)
    print(f"3D-Evaluierungs-Plot gespeichert unter: '{save_path}'")
    plt.show()


def plot_3d_slice_comparison(
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    z_1d: np.ndarray,
    true_3d: Dict[str, np.ndarray],
    pred_3d: Dict[str, np.ndarray],
    plane: str = "xy",
    slice_coord: float = 0.0,
    save_path: str = "slice_eval.png",
) -> None:
    """
    Schneidet aus 3D-Feldern (Nx, Ny, Nz) eine 2D-Ebene ('xy', 'xz' oder 'yz') bei 'slice_coord'
    heraus und nutzt die bestehende 2D-3x3-Plot-Funktion wieder!
    """
    if plane == "xy":
        idx = int(np.argmin(np.abs(z_1d - slice_coord)))
        X_2d, Y_2d = np.meshgrid(x_1d, y_1d, indexing="ij")
        true_2d = {k: v[:, :, idx] for k, v in true_3d.items()}
        pred_2d = {k: v[:, :, idx] for k, v in pred_3d.items()}
    elif plane == "xz":
        idx = int(np.argmin(np.abs(y_1d - slice_coord)))
        X_2d, Y_2d = np.meshgrid(x_1d, z_1d, indexing="ij")
        true_2d = {k: v[:, idx, :] for k, v in true_3d.items()}
        pred_2d = {k: v[:, idx, :] for k, v in pred_3d.items()}
    elif plane == "yz":
        idx = int(np.argmin(np.abs(x_1d - slice_coord)))
        X_2d, Y_2d = np.meshgrid(y_1d, z_1d, indexing="ij")
        true_2d = {k: v[idx, :, :] for k, v in true_3d.items()}
        pred_2d = {k: v[idx, :, :] for k, v in pred_3d.items()}
    else:
        raise ValueError(f"Unbekannte Schnittebene: {plane}. Wähle 'xy', 'xz' oder 'yz'.")

    plot_comparison_and_residuals(X_2d, Y_2d, true_2d, pred_2d, save_path=save_path)