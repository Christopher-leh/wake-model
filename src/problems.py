from abc import ABC, abstractmethod
from typing import Dict
import torch 
import torch.nn as nn

class BaseProblem(ABC):
    """Abstrakte Basisklasse: Erzwingt, dass jedes Problem 'compute_loss' implementiert."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model 

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Muss ein Dict mit mindestens dem Key 'total' zurückgeben."""
        pass


class NS2DPINN(BaseProblem):
    def __init__(self, model: nn.Module, nu: float, pde_weight: float = 1.0) -> None:
        super().__init__(model)
        self.nu = nu
        self.pde_weight = pde_weight

    def _pde_residuals(self, x: torch.Tensor, y: torch.Tensor):
        out = self.model(torch.cat([x,y], dim = 1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        u_x, u_y = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
        v_x, v_y = torch.autograd.grad(v, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
        p_x, p_y = torch.autograd.grad(p, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        res_mom_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_mom_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        res_cont = u_x + v_y

        return res_mom_x, res_mom_y, res_cont

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_data = self.model(torch.cat([batch["x_data"], batch["y_data"]], dim=1))
        loss_data = torch.mean((pred_data[:, 0:2] - batch["uv_data"]) ** 2)

        res_mx, res_my, res_c = self._pde_residuals(batch["x_col"], batch["y_col"])
        loss_pde = torch.mean(res_mx ** 2 + res_my ** 2 + res_c ** 2)  

        total_loss = loss_data + self.pde_weight * loss_pde 
        return {"total": total_loss , "data": loss_data.detach(), "pde": loss_pde.detach()}

    def evaluate(self, val_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Berechnet den relativen L2-Testfehler gegen die wahre Lösung (u, v, p)
        auf einem ungesehenen Validierungs-/Testgitter.
        """

        self.model.eval()
        preds = self.model(val_batch["coords"])
        targets = val_batch["u_true"]

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
