from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn
from src.physics import NS2D


class BaseProblem(ABC):
    """Abstrakte Basisklasse: Erzwingt, dass jedes Problem 'compute_loss' implementiert."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def compute_loss(
        self,
        coords_col: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Muss ein Dict mit mindestens dem Key 'total' zurückgeben."""
        pass


class NS2DPINN(BaseProblem):
    def __init__(
        self,
        model: nn.Module,
        nu: float,
        pde_weight: float = 1.0,
        scales: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__(model)
        self.nu = nu
        self.pde_weight = pde_weight
        self.scales = scales
        self.x_scale, self.y_scale = scales

    def compute_loss(
        self,
        coords_col: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred_data = self.model(coords_data)
        loss_data = torch.mean((pred_data[:, 0:2] - targets_data) ** 2)

        res_mx, res_my, res_c = NS2D(coords_col, self.model, self.scales, self.nu)
        loss_pde = torch.mean(res_mx**2 + res_my**2 + res_c**2)

        total_loss = loss_data + self.pde_weight * loss_pde
        return {
            "total": total_loss,
            "data": loss_data.detach(),
            "pde": loss_pde.detach(),
        }

    def evaluate(self, test_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Berechnet den relativen L2-Testfehler gegen die wahre Lösung (u, v, p)
        auf einem ungesehenen Validierungs-/Testgitter.
        """

        self.model.eval()
        with torch.no_grad():
            preds = self.model(test_batch["coords"])
        targets = test_batch["targets"]

        metrics = {}
        for idx, name in enumerate(["u", "v", "p"]):
            pred_f = preds[:, idx].clone()
            true_f = targets[:, idx].clone()

            if name == "p":
                pred_f -= torch.mean(pred_f)
                true_f -= torch.mean(true_f)

            l2_err = torch.linalg.norm(pred_f - true_f) / torch.linalg.norm(true_f)
            metrics[f"val_l2_{name}"] = l2_err.item()

        metrics["val_l2_mean"] = sum(metrics.values()) / len(metrics)
        self.model.train()

        return metrics
