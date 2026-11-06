"""Die EINE universelle Trainingsschleife inkl. Speichern/Laden."""

from pathlib import Path
from typing import Any, Dict
import torch
from src.problems import BaseProblem


class Trainer:

    def __init__(
        self,
        problem: BaseProblem,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str = "checkpoints",
        config: Dict[str, Any] | None = None
    ) -> None:
        
        self.problem = problem
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.best_loss = float("inf")
        self.history: Dict[str, list] = {"total": [], "data": [], "pde": []}

    def fit(self, train_batch: Dict[str, torch.Tensor], val_batch: Dict[str, torch.Tensor], epochs: int = 100, log_every: int = 200) -> None:

        self.problem.model.train()

        for epoch in range(1, epochs + 1 ):
            self.optimizer.zero_grad()
            losses = self.problem.compute_loss(train_batch)
            losses["total"].backward()
            self.optimizer.step()

            for k, v in losses.items():
                self.history.setdefault(k, []).append(v.item()) 
            
            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                val_msg = ""
                if val_batch is not None:
                    val_metrics = self.problem.evaluate(val_batch)
                    for k, v in val_metrics.items():
                        self.history.setdefault(k, []).append(v)
                    val_msg = " || " + " | ".join(f"{k}: {v:.2%}" for k, v in val_metrics.items())

                    # Speichere bestes Modell anhand des echten Test-Fehlers (falls vorhanden)
                    if val_metrics["val_l2_mean"] < self.best_loss:
                        self.best_loss = val_metrics["val_l2_mean"]
                        self.save_checkpoint("best_model.pt", epoch, self.best_loss)
                else:
                    if losses["total"].item() < self.best_loss:
                        self.best_loss = losses["total"].item()
                        self.save_checkpoint("best_model.pt", epoch, self.best_loss)

                train_msg = " | ".join(f"{k}: {v.item():.2e}" for k, v in losses.items())
                print(f"Epoch {epoch:04d} | {train_msg}{val_msg}")










    def save_checkpoint(self, filename: str, epoch: int, loss : float) -> None:
        path = self.checkpoint_dir / filename 
        torch.save(
           {
                "epoch": epoch,
                "model_state_dict": self.problem.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "config": self.config
           },
           path,
        )

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, weights_only = False)
        self.problem.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint '{filename}' geladen (Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.2e})")
        return checkpoint