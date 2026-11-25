import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, BatchSampler, RandomSampler
from scipy import ndimage


def normalize_data(values: np.array) -> np.array:
    """
    scale to [-1, 1]
    """
    values_min = values.min()
    values_max = values.max()
    values_norm = 2.0 * (values - values_min) / (values_max - values_min) - 1.0

    return values_norm, values_min, values_max


class BasePINNData:
    """Dimensionsunabhängige Basisklasse für Daten.

    Unterklassen setzen in _load_data per _build_tensors die Trainingstensoren
    und zusätzlich self.val_batch / self.test_batch (per _make_batch).
    """

    def __init__(
        self, device: str = "cpu", rng: np.random.Generator | None = None
    ) -> None:
        self.device = device
        self.scales: tuple[float, ...] = ()
        self.rng = rng or np.random.default_rng()

    def _to_tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=self.device)

    def _build_tensors(
        self,
        coords_col: np.ndarray,  # (N_col, D_in)    z.B. [x, y] oder [x, y, t]
        coords_data: np.ndarray,  # (N_data, D_in)
        targets_data: np.ndarray,  # (N_data, D_obs)  z.B. [u, v]
        pde_terms: np.ndarray | None = None,  # (N_col, D_pde) z.B. [f_x, f_y, visc]
    ) -> None:
        self.coords_col = self._to_tensor(coords_col)
        if pde_terms is not None:
            self.pde_terms = self._to_tensor(pde_terms)
        else:
            # Platzhalter, damit der Loader immer (coords, pde_terms) liefert
            self.pde_terms = torch.zeros(len(coords_col), 0, device=self.device)
        self.coords_data = self._to_tensor(coords_data)
        self.targets_data = self._to_tensor(targets_data)

    def _make_batch(
        self, coords: np.ndarray, targets: np.ndarray, grid_shape: tuple[int, ...]
    ) -> dict:
        """Auswertungs-Batch (Validierung/Test) für problem.evaluate.
        grid_shape: Form des Gitters, z.B. (n_x, n_y) oder (n_x, n_y, n_t)."""
        return {
            "coords": self._to_tensor(coords),
            "targets": self._to_tensor(targets),
            "grid_shape": grid_shape,
        }

    def get_loaders(
        self, batch_size_col: int = 500, batch_size_data: int | None = None
    ) -> tuple[DataLoader, DataLoader]:
        # BatchSampler holt ganze Batches auf einmal per Index -> viel schneller
        # als einzelne Samples, da alle Tensoren schon auf dem Device liegen
        col_dataset = TensorDataset(self.coords_col, self.pde_terms)
        col_loader = DataLoader(
            col_dataset,
            sampler=BatchSampler(RandomSampler(col_dataset), batch_size_col, drop_last=False),
            batch_size=None,
        )

        data_dataset = TensorDataset(self.coords_data, self.targets_data)
        bs_data = batch_size_data or len(data_dataset)
        data_loader = DataLoader(
            data_dataset,
            sampler=BatchSampler(RandomSampler(data_dataset), bs_data, drop_last=False),
            batch_size=None,
        )

        return col_loader, data_loader


class WakeData2D(BasePINNData):
    """Zeitlich gemitteltes Feld (Mean Flow), eine z-Ebene. Koordinaten [x, y]."""

    def __init__(
        self,
        data_dir: str,
        z_index: int,
        n_train: int,
        n_col: int,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(device=device, rng=rng)
        self.data_dir = Path(data_dir)
        self.z_index = z_index
        self.n_train = n_train
        self.n_col = n_col

        self._load_data()

    def _load_data(self) -> None:
        with h5py.File(self.data_dir / "meanvelocity.h5", "r") as data_file:
            field = np.asarray(
                data_file["meanvelocity_0001"]
            )  # z,y,x, (u, v, w) (25, 30, 301, 3)
            x = np.asarray(data_file["xcoor"], dtype=np.float32)
            y = np.asarray(data_file["ycoor"], dtype=np.float32)

        with h5py.File(self.data_dir / "meanpressure.h5", "r") as data_file:
            pressure = np.asarray(data_file["meanpressure_0001"])  # z,y,x (25, 30, 301)

        x_norm, x_min, x_max = normalize_data(x)
        y_norm, y_min, y_max = normalize_data(y)

        x_scale = float(2.0 / (x_max - x_min))
        y_scale = float(2.0 / (y_max - y_min))
        self.scales = (x_scale, y_scale)

        self.X, self.Y = np.meshgrid(x_norm, y_norm, indexing="ij")  # X, Y (301, 30)

        self.X_phys, self.Y_phys = np.meshgrid(
            x, y, indexing="ij"
        )  # zum plotten der physikalischen Lösung

        # without transpose it would be (y, x)
        u_true = field[self.z_index, :, :, 0].T  # u (301, 30)
        v_true = field[self.z_index, :, :, 1].T  # v (301, 30)
        p_true = np.squeeze(pressure[self.z_index]).T  # p (301, 30)
        self.true_dict = {"u": u_true, "v": v_true, "p": p_true}

        x_data = self.rng.integers(0, len(x), size=self.n_train)
        y_data = self.rng.integers(0, len(y), size=self.n_train)

        x_col = self.rng.integers(0, len(x), size=self.n_col)
        y_col = self.rng.integers(0, len(y), size=self.n_col)

        self._build_tensors(
            coords_col=np.column_stack([x_norm[x_col], y_norm[y_col]]),
            coords_data=np.column_stack([x_norm[x_data], y_norm[y_data]]),
            targets_data=np.column_stack(
                [u_true[x_data, y_data], v_true[x_data, y_data]]
            ),
        )

        # Mean Flow: keine Zeit zum Zurückhalten -> Validierung = Test = ganzes Gitter
        full_grid = self._make_batch(
            coords=np.column_stack([self.X.ravel(), self.Y.ravel()]),
            targets=np.column_stack([u_true.ravel(), v_true.ravel(), p_true.ravel()]),
            grid_shape=self.X.shape,
        )
        self.val_batch = full_grid
        self.test_batch = full_grid


# Referenzgrößen (für entdimensionalisierung)
L_REF = 126.0  # (Referenzlänge, hier Rotordurchmesser)
U_REF = 10.0  # (Referenzgeschwindigkeit, hier Windgeschwindigkeit vor der Turbine)
T_REF = L_REF / U_REF  # (Referenzzeit)

# [x,y,t], [u,v,p]
class WakeData2Dtime(BasePINNData):
    def __init__(
        self,
        data_dir: str,
        z_index: int,
        n_train: int,
        n_col: int,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
        cone_angle: float = np.pi / 8,  # Beispiel: 45 Grad Kegelwinkel
        radius: float = 50,  # Beispiel: Radius des Kegels (920 m in x und etwa 985 m in y.)
        sampling_near_turbine: bool = True,  # Wenn True, werden die Trainingspunkte in der Nähe der Turbine gesampelt, sonst zufällig im gesamten Bereich
    ) -> None:
        super().__init__(device=device, rng=rng)
        self.data_dir = Path(data_dir)
        self.z_index = z_index
        self.n_train = n_train
        self.n_col = n_col
        self.dt = 0.5
        self.cone_angle = cone_angle
        self.radius = radius
        self.sampling_near_turbine = sampling_near_turbine

        self._load_data()

    def _load_snapshots(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        """Lädt alle <prefix>_t*.h5, gibt (snapshots, field) zurück.
        field-Shape: (n_t, y, x, n_comp) – nur die Ebene self.z_index."""
        snapshots, fields = [], []
        for path in sorted(self.data_dir.glob(f"{prefix}_t*.h5")):
            with h5py.File(path, "r") as f:
                for key in sorted(k for k in f if k.startswith(f"{prefix}_")):
                    snapshots.append(int(key.split("_")[1]))  # velocity_0013 -> 13
                    fields.append(
                        f[key][self.z_index, :, :, :]
                    )  # nur die Ebene self.z_index
        return np.array(snapshots), np.stack(fields)

    def _grid_batch(self, t_ids, u, v, p) -> dict:
        """Auswertungs-Batch über das ganze x-y-Gitter, nur für die Zeitschritte t_ids."""
        sl = (slice(None), slice(None), t_ids)
        return self._make_batch(
            coords=np.column_stack([self.X[sl].ravel(), self.Y[sl].ravel(), self.T[sl].ravel()]),
            targets=np.column_stack([u[sl].ravel(), v[sl].ravel(), p[sl].ravel()]),
            grid_shape=(self.X.shape[0], self.X.shape[1], len(t_ids)),
        )

    def _load_data(self) -> None:
        t_snap, vel = self._load_snapshots("velocity")  # vel: (n_t, y, x, 3)
        t_snap_p, pres = self._load_snapshots("pressure")  # pres: (n_t, y, x, 1)
        t_snap_f, force = self._load_snapshots("force")
        t_snap_visc, visc = self._load_snapshots("sgsviscosity")

        with h5py.File(self.data_dir / "velocity_t0001.h5", "r") as data_file:
            x = np.asarray(data_file["xcoor"], dtype=np.float32)
            y = np.asarray(data_file["ycoor"], dtype=np.float32)
        t = (t_snap - t_snap[0]) * self.dt
        n_t = len(t)
        t_idx = np.arange(n_t)  # Indexe der Zeitschritte, die wir verwenden

        # zwischenschritte auslassen, für validierung und test
        val_t = t_idx[5::10]  # alle 10 Zeitschritte ab dem 5. für Validierung
        test_t = t_idx[9::10]  # alle 10 Zeitschritte ab dem 9. für Test
        train_t = np.setdiff1d(t_idx, np.concatenate([val_t, test_t]))  # Rest für Training

        with h5py.File(self.data_dir / "force_t0001.h5", "r") as f:
            x_force = np.asarray(f["xcoor"], dtype=np.float32)
            x_force_offset = len(x) - len(x_force)
        force_pad = np.zeros_like(vel)
        force_pad[:, :, x_force_offset:, :] = force

        # Entdimensionalisierung der Daten
        x_dim = x / L_REF
        y_dim = y / L_REF
        t_dim = t / T_REF
        vel_dim = vel / U_REF
        pres_dim = pres / (U_REF**2)
        force_dim = force_pad / (U_REF**2 / L_REF)
        visc_dim = visc / (U_REF * L_REF)

        self.ref = {
            "L_REF": L_REF,
            "T_REF": T_REF,
            "U_REF": U_REF,
        }

        x_norm, x_min, x_max = normalize_data(x_dim)
        y_norm, y_min, y_max = normalize_data(y_dim)
        t_norm, t_min, t_max = normalize_data(t_dim)

        x_scale = float(2.0 / (x_max - x_min))
        y_scale = float(2.0 / (y_max - y_min))
        t_scale = float(2.0 / (t_max - t_min))
        self.scales = (x_scale, y_scale, t_scale)

        self.X, self.Y, self.T = np.meshgrid(
            x_norm, y_norm, t_norm, indexing="ij"
        )  # X, Y, T (301, 30, n_t)

        self.X_phys, self.Y_phys, self.T_phys = np.meshgrid(
            x, y, t, indexing="ij"
        )  # zum plotten der physikalischen Lösung

        # without transpose it would be (y, x)
        u_true = vel_dim[:, :, :, 0].T  # u (nt, 30, 301).T -> (301, 100, nt)
        v_true = vel_dim[:, :, :, 1].T  # v (nt, 30, 301).T -> (301, 100, nt)
        p_true = pres_dim[:, :, :, 0].T  # p (301, 100, nt)
        force_x = force_dim[:, :, :, 0].T
        force_y = force_dim[:, :, :, 1].T
        visc = visc_dim[:, :, :, 0].T

        self.true_dict = {"u": u_true, "v": v_true, "p": p_true}

        # hier überlegen wie daten gesamplet werden, dass sie auch im wake liegen, vielleicht doch wie lidar daten samplen
        # dafür noch funktion schreiben die les daten in lidar daten umwandelt. also strahlen ausgehend von gondeln nach vorne
        # maximum der kraft ist standort der turbine

        # Verwende den sample range, um die Datenpunkte zu wählen
        if self.sampling_near_turbine:
            F_mag = (
                np.mean(force_pad[:, :, :, 0], axis=0) ** 2
                + np.mean(force_pad[:, :, :, 1], axis=0) ** 2
            )  # |F| in (y, x), evtl. über t gemittelt
            mask = F_mag > 0.05 * F_mag.max()  # Schwellwert, z.B. ein paar Prozent
            labels, n = ndimage.label(mask)  # n sollte 15 sein
            centers = ndimage.center_of_mass(F_mag, labels, range(1, n + 1))


            cone_mask = np.zeros_like(F_mag, dtype=bool)
            for center in centers:
                center_y, center_x = int(center[0]), int(center[1])
                for y_f in range(F_mag.shape[0]):
                    for x_f in range(F_mag.shape[1]):
                        dy = center_y - y_f
                        dx = (
                            center_x - x_f
                        )  # Invertiere dx, um die Richtung zu berücksichtigen
                        distance = np.sqrt(dx**2 + dy**2)
                        if (
                            distance <= self.radius
                            and np.abs(np.arctan2(dy, dx)) <= self.cone_angle
                        ):
                            cone_mask[y_f, x_f] = True
            # sample data in cone
            ys, xs = np.where(cone_mask)
            idx = self.rng.choice(len(xs), size=self.n_train, replace=False)
            # -> xs[idx], ys[idx]
            x_data, y_data = xs[idx], ys[idx]
            t_data = self.rng.choice(
                train_t, size=self.n_train
            )

        else:
            x_data = self.rng.integers(0, len(x), size=self.n_train)
            y_data = self.rng.integers(0, len(y), size=self.n_train)
            t_data = self.rng.choice(train_t, size=self.n_train)

        x_col = self.rng.integers(0, len(x), size=self.n_col)
        y_col = self.rng.integers(0, len(y), size=self.n_col)
        t_col = self.rng.integers(0, len(t), size=self.n_col)

        self._build_tensors(
            coords_col=np.column_stack([x_norm[x_col], y_norm[y_col], t_norm[t_col]]),
            coords_data=np.column_stack(
                [x_norm[x_data], y_norm[y_data], t_norm[t_data]]
            ),
            targets_data=np.column_stack(
                [u_true[x_data, y_data, t_data], v_true[x_data, y_data, t_data]]
            ),
            pde_terms=np.column_stack(
                [
                    force_x[x_col, y_col, t_col],
                    force_y[x_col, y_col, t_col],
                    visc[x_col, y_col, t_col],
                ]
            ),
        )

        # Validierung (Modellauswahl) und Test (finaler Vergleich) auf zurückgehaltenen Zeitschritten
        self.val_t, self.test_t, self.train_t = val_t, test_t, train_t
        self.test_batch = self._grid_batch(test_t, u_true, v_true, p_true)
        self.val_batch = self._grid_batch(val_t, u_true, v_true, p_true)
