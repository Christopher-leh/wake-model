from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn
from src.physics import NS2D, NS2Dtime


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
        pde_terms: torch.Tensor,
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
        pde_terms: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred_data = self.model(coords_data)
        loss_data = torch.mean((pred_data[:, 0:2] - targets_data) ** 2)

        res_mx, res_my, res_c = NS2D( self.model,coords_col,pde_terms, self.scales, self.nu)
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



class NS2DtimePINN(BaseProblem):
    def __init__(
        self,
        model: nn.Module,
        nu: float,
        data_weight: float = 1.0,
        pde_weight_imp: float = 1.0,
        pde_weight_cont: float = 1.0,
        scales: tuple[float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__(model)
        self.nu = nu
        self.data_weight = data_weight
        self.pde_weight_imp = pde_weight_imp
        self.pde_weight_cont = pde_weight_cont
        self.scales = scales
        self.x_scale, self.y_scale, self.t_scale = scales

    def compute_loss(
        self,
        coords_col: torch.Tensor,
        pde_terms: torch.Tensor,
        coords_data: torch.Tensor,
        targets_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred_data = self.model(coords_data)
        loss_data = torch.mean((pred_data[:, 0:2] - targets_data) ** 2)

        res_mx, res_my, res_c = NS2Dtime(self.model, coords_col, pde_terms, self.scales, self.nu)
        loss_pde_imp = torch.mean(res_mx**2 + res_my**2)
        loss_pde_cont = torch.mean(res_c**2)

        total_loss = self.data_weight * loss_data + self.pde_weight_imp * loss_pde_imp + self.pde_weight_cont * loss_pde_cont
        return {
            "total": total_loss,
            "data": loss_data.detach(),
            "pde_imp": loss_pde_imp.detach(),
            "pde_cont": loss_pde_cont.detach(),
        }

    def evaluate(self, test_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Berechnet den relativen L2-Testfehler gegen die wahre Lösung (u, v, p)
        auf einem ungesehenen Validierungs-/Testgitter.
        """

        self.model.eval()
        with torch.no_grad():
            chunks = torch.split(test_batch["coords"], 100000, dim=0)
            preds_list = [self.model(chunk) for chunk in chunks]
            preds = torch.cat(preds_list, dim=0)
        targets = test_batch["targets"]
        n_x, n_y, n_t = test_batch["grid_shape"]

        metrics = {}
        for idx, name in enumerate(["u", "v", "p"]):
            pred_f = preds[:, idx].clone()
            true_f = targets[:, idx].clone()
        # pressure here is a function of time, so we need a different handling
        # we have to subtract the mean of the pressure at each time step to get the relative error
        # not the global mean
            if name == "p": 
                pred_f = pred_f.reshape(n_x, n_y, n_t) # (3_612_000,) → (301, 100, 120)
                true_f = true_f.reshape(n_x, n_y, n_t)
                pred_f -= pred_f.mean(dim=(0, 1), keepdim=True) #mean over x,y, so one mean per timestep t(120)
                true_f -= true_f.mean(dim=(0, 1), keepdim=True)

            l2_err = torch.linalg.norm(pred_f - true_f) / torch.linalg.norm(true_f)
            metrics[f"val_l2_{name}"] = l2_err.item()

        metrics["val_l2_mean"] = sum(metrics.values()) / len(metrics)
        self.model.train()

        return metrics
