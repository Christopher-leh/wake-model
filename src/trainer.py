"""Die EINE universelle Trainingsschleife inkl. Speichern/Laden."""

import signal
from pathlib import Path
from typing import Any, Dict
import torch
from src.problems import BaseProblem
from itertools import cycle
from torch.utils.data import DataLoader
from collections import defaultdict


def format_metrics(metrics: Dict[str, float], monitor: str) -> str:
    """Kurze Konsolenausgabe: nur NRMSE pro Feld + Monitor-Wert.
    Alle Metriken (auch L2) stehen weiterhin vollständig in der History."""
    fields = " ".join(f"{n} {metrics[f'val_nrmse_{n}']:.1%}" for n in "uvp")
    return f"nrmse {fields} | {monitor.removeprefix('val_')} {metrics[monitor]:.2%}"


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
        self.history: Dict[str, list] = {}  # Keys kommen aus compute_loss
        self.val_history: Dict[str, list] = {"epoch":[]}
        self.lr_history: list = []

    def fit(
        self,
        col_loader: DataLoader,
        data_loader: DataLoader,
        val_batch: Dict[str, Any] | None = None,
        epochs: int = 100,
        log_every: int = 200,
        monitor: str = "val_nrmse_uv",
    ) -> None:
        """monitor: Validierungsmetrik (kleiner = besser), nach der 'best_model.pt' gespeichert wird."""

        self.problem.model.train()
        self.best_metric, self.best_epoch = float("inf"), None
        self.last_epoch = 0
        self.stopped_early = False

        # Strg+C: 1x = nach der aktuellen Epoche sauber beenden (Validierung, Speichern, Plots
        # laufen danach normal weiter), 2x = sofortiger harter Abbruch
        def request_stop(signum, frame):
            if self.stopped_early:
                raise KeyboardInterrupt
            self.stopped_early = True
            print("\n[Strg+C] Training wird nach dieser Epoche beendet "
                  "(nochmal Strg+C = sofortiger Abbruch ohne Speichern)")

        old_handler = signal.signal(signal.SIGINT, request_stop)
        try:
            self._fit_loop(col_loader, data_loader, val_batch, epochs, log_every, monitor)
        finally:
            signal.signal(signal.SIGINT, old_handler)

        if self.stopped_early:
            print(f"Training manuell beendet nach Epoch {self.last_epoch}/{epochs}")
        if self.best_epoch is not None:
            print(f"Bestes Modell: Epoch {self.best_epoch}, {monitor} = {self.best_metric:.2%}")

    def _fit_loop(self, col_loader, data_loader, val_batch, epochs, log_every, monitor) -> None:
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
            self.last_epoch = epoch

            # bei manuellem Stopp diese letzte Epoche noch validieren, damit nichts verloren geht
            if epoch % log_every == 0 or epoch == 1 or epoch == epochs or self.stopped_early:
                val_msg = ""
                if val_batch is not None:
                    val_metrics = self.problem.evaluate(val_batch)
                    self.val_history["epoch"].append(epoch)
                    for k, v in val_metrics.items():
                        self.val_history.setdefault(k, []).append(v)
                    val_msg = " || val " + format_metrics(val_metrics, monitor)

                    if val_metrics[monitor] < self.best_metric:
                        self.best_metric = val_metrics[monitor]
                        self.best_epoch = epoch
                        self.save_checkpoint("best_model.pt", epoch, epoch_losses["total"])
                        val_msg += " *"

                train_msg = " ".join(f"{k} {v:.1e}" for k, v in epoch_losses.items())
                print(f"Ep {epoch:5d} | lr {lr:.0e} | {train_msg}{val_msg}")

            if self.stopped_early:
                break

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
