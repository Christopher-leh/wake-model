"""Die EINE universelle Trainingsschleife inkl. Speichern/Laden."""

from pathlib import Path
from typing import Any, Dict
import torch
from src.problems import BaseProblem
from itertools import cycle
from torch.utils.data import DataLoader
from collections import defaultdict


class Trainer:

    def __init__(
        self,
        problem: BaseProblem,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        checkpoint_dir: str = "checkpoints",
        config: Dict[str, Any] | None = None,
    ) -> None:

        self.problem = problem
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.history: Dict[str, list] = {"total": [], "data": [], "pde": []}
        self.val_history: Dict[str, list] = {"epoch":[]}
        self.lr_history: list = []

    def fit(
        self,
        col_loader: DataLoader,
        data_loader: DataLoader,
        test_batch: Dict[str, torch.Tensor] | None = None,
        epochs: int = 100,
        log_every: int = 200,
    ) -> None:

        self.problem.model.train()

        for epoch in range(1, epochs + 1):
            # defaultdict bricht nicht ab bei fehlendem key
            epoch_losses = defaultdict(float) # float() ergibt 0.0 als standardwert
            
            num_batches = 0

            for (coords_col, pde_terms), (coords_data, targets_data) in zip(
                col_loader, cycle(data_loader)
            ):

                self.optimizer.zero_grad()
                losses = self.problem.compute_loss(
                    coords_col, pde_terms, coords_data, targets_data
                )
                losses["total"].backward()
                self.optimizer.step()

                for k, v in losses.items():
                    epoch_losses[k] += v.item() # hier also auch neue losse ohne defintion oben nutzbar, wegen defaultdict
                num_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= num_batches
                self.history.setdefault(k, []).append(epoch_losses[k])

            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(lr)

            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                test_msg = ""
                if test_batch is not None:
                    test_metrics = self.problem.evaluate(test_batch)
                    self.val_history["epoch"].append(epoch)
                    for k, v in test_metrics.items():
                        self.val_history.setdefault(k, []).append(v)
                    test_msg = " || " + " | ".join(f"{k}: {v:.2%}" for k, v in test_metrics.items())

                train_msg = " | ".join(f"{k}: {v:.2e}" for k, v in epoch_losses.items())
                print(f"Epoch {epoch:04d} | lr: {lr:.1e} | {train_msg}{test_msg}")

    def save_checkpoint(self, filename: str, epoch: int, loss: float) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.problem.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "config": self.config,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            },
            path,
        )

    def load_checkpoint(self, filename: str, device: str) -> Dict[str, Any]:
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        self.problem.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(
            f"Checkpoint '{filename}' geladen (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.2e})"
        )
        return checkpoint
