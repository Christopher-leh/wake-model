from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn


class BaseProblem(ABC):
    """Abstrakte Basisklasse: Erzwingt, dass jedes Problem 'compute_loss' implementiert."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def compute_loss(self, coords_col : torch.Tensor, coords_data: torch.Tensor, targets_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Muss ein Dict mit mindestens dem Key 'total' zurückgeben."""
        pass


class NS2DPINN(BaseProblem):
    def __init__(self, model: nn.Module, nu: float, pde_weight: float = 1.0, scales: tuple[float, float] = (1.0, 1.0)) -> None:
        super().__init__(model)
        self.nu = nu
        self.pde_weight = pde_weight
        self.x_scale, self.y_scale = scales
        

    def _pde_residuals(
        self,
        coords_col: torch.Tensor,
    ) -> torch.Tensor:
        x = coords_col[:, 0:1].detach().requires_grad_(True)
        y = coords_col[:, 1:2].detach().requires_grad_(True)

        out = self.model(torch.cat([x, y], dim=1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # calculate gradients, scaled by the spatial scales(from normalization)

        u_x_norm, u_y_norm = torch.autograd.grad(
            u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
        )
        v_x_norm, v_y_norm = torch.autograd.grad(
                v, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
        )
        p_x_norm, p_y_norm = torch.autograd.grad(
            p, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
        )

        u_xx_norm = torch.autograd.grad(
            u_x_norm, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_yy_norm = torch.autograd.grad(
            u_y_norm, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        v_xx_norm = torch.autograd.grad(
            v_x_norm, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        v_yy_norm = torch.autograd.grad(
            v_y_norm, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        u_x = self.x_scale * u_x_norm  # ∂/∂x = 2 / (x_max - x_min) · ∂/∂x_norm
        u_y = self.y_scale * u_y_norm  # ∂/∂y = 2 / (y_max - y_min) · ∂/∂y_norm

        v_x = self.x_scale * v_x_norm
        v_y = self.y_scale * v_y_norm

        p_x = self.x_scale * p_x_norm
        p_y = self.y_scale * p_y_norm

        u_xx = self.x_scale**2 * u_xx_norm
        u_yy = self.y_scale**2 * u_yy_norm

        v_xx = self.x_scale**2 * v_xx_norm
        v_yy = self.y_scale**2 * v_yy_norm

        res_mom_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_mom_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        res_cont = u_x + v_y

        return res_mom_x, res_mom_y, res_cont

    def compute_loss(self, coords_col: torch.Tensor, coords_data: torch.Tensor, targets_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        pred_data = self.model(coords_data)
        loss_data = torch.mean((pred_data[:, 0:2] - targets_data) ** 2)

        res_mx, res_my, res_c = self._pde_residuals(coords_col)
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
