from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn
from src.physics import NS2D, NS2Dtime


class BaseProblem(ABC):
    """Abstrakte Basisklasse: jedes Problem implementiert 'compute_loss'.
    'evaluate' ist für alle Probleme gleich und steckt deshalb hier."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def compute_loss(
        self,
        coords_col: torch.Tensor,
        pde_terms: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Muss ein Dict mit mindestens dem Key 'total' zurückgeben."""

    def evaluate(self, batch: dict) -> Dict[str, float]:
        """
        Fehler gegen die wahre Lösung (u, v, p) auf einem Validierungs-/Test-Batch.
        - val_l2_*:    relativer L2-Fehler ||pred - true|| / ||true||
        - val_nrmse_*: RMSE / (max(true) - min(true)), wie im Paper (Tabelle 2)
        """
        self.model.eval()
        with torch.no_grad():
            chunks = torch.split(batch["coords"], 100_000, dim=0)
            preds = torch.cat([self.model(chunk) for chunk in chunks], dim=0)
        targets = batch["targets"]
        grid_shape = batch["grid_shape"]

        metrics = {}
        for idx, name in enumerate(["u", "v", "p"]):
            pred_f = preds[:, idx]
            true_f = targets[:, idx]

            # Druck ist nur bis auf eine Konstante bestimmt -> Mittelwert abziehen.
            # Mittel über x, y: beim Mean Flow global, bei Zeitdaten pro Zeitschritt.
            if name == "p":
                pred_f = pred_f.reshape(grid_shape)
                true_f = true_f.reshape(grid_shape)
                pred_f = pred_f - pred_f.mean(dim=(0, 1), keepdim=True)
                true_f = true_f - true_f.mean(dim=(0, 1), keepdim=True)

            diff = pred_f - true_f
            metrics[f"val_l2_{name}"] = (torch.linalg.norm(diff) / torch.linalg.norm(true_f)).item()
            rmse = torch.sqrt(torch.mean(diff**2))
            metrics[f"val_nrmse_{name}"] = (rmse / (true_f.max() - true_f.min())).item()

        metrics["val_l2_mean"] = (metrics["val_l2_u"] + metrics["val_l2_v"] + metrics["val_l2_p"]) / 3
        metrics["val_nrmse_uv"] = (metrics["val_nrmse_u"] + metrics["val_nrmse_v"]) / 2  # ohne p, für Paper-Vergleich
        self.model.train()

        return metrics


class NS2DPINN(BaseProblem):
    """Stationäres 2D-NS für den Mean Flow, Koordinaten [x, y]."""

    def __init__(
        self,
        model: nn.Module,
        nu: float,
        data_weight: float = 1.0,
        pde_weight: float = 1.0,
        scales: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__(model)
        self.nu = nu
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.scales = scales

    def compute_loss(
        self,
        coords_col: torch.Tensor,
        pde_terms: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred_data = self.model(coords_data)
        loss_data = torch.mean((pred_data[:, 0:2] - targets_data) ** 2)

        res_mx, res_my, res_c = NS2D(self.model, coords_col, pde_terms, self.scales, self.nu)
        loss_pde = torch.mean(res_mx**2 + res_my**2 + res_c**2)

        total_loss = self.data_weight * loss_data + self.pde_weight * loss_pde
        return {
            "total": total_loss,
            "data": loss_data.detach(),
            "pde": loss_pde.detach(),
        }


class NS2DtimePINN(BaseProblem):
    """Instationäres 2D-NS mit Turbinenkraft und SGS-Viskosität, Koordinaten [x, y, t]."""

    def __init__(
        self,
        model: nn.Module,
        nu: float,
        data_weight: float = 1.0,
        pde_weight_imp: float = 1.0,
        pde_weight_cont: float = 1.0,
        scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__(model)
        self.nu = nu
        self.data_weight = data_weight
        self.pde_weight_imp = pde_weight_imp
        self.pde_weight_cont = pde_weight_cont
        self.scales = scales

    def compute_loss(
        self,
        coords_col: torch.Tensor,
        pde_terms: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred_data = self.model(coords_data)
        loss_data = torch.mean((pred_data[:, 0:2] - targets_data) ** 2)

        # Nur Daten: PDE-Residuen gar nicht erst berechnen (spart die teuren Ableitungen)
        if self.pde_weight_imp == 0.0 and self.pde_weight_cont == 0.0:
            loss_pde_imp = torch.zeros((), device=coords_col.device)
            loss_pde_cont = torch.zeros((), device=coords_col.device)
        else:
            res_mx, res_my, res_c = NS2Dtime(self.model, coords_col, pde_terms, self.scales, self.nu)
            loss_pde_imp = torch.mean(res_mx**2 + res_my**2)
            loss_pde_cont = torch.mean(res_c**2)

        total_loss = (
            self.data_weight * loss_data
            + self.pde_weight_imp * loss_pde_imp
            + self.pde_weight_cont * loss_pde_cont
        )
        return {
            "total": total_loss,
            "data": loss_data.detach(),
            "pde_imp": loss_pde_imp.detach(),
            "pde_cont": loss_pde_cont.detach(),
        }
