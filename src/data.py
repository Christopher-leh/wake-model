import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def normalize_data(values: np.array) -> np.array:
    """
    scale to [-1, 1]
    """
    values_min = values.min()
    values_max = values.max()
    values_norm = 2.0 * (values - values_min) / (values_max - values_min) - 1.0

    return values_norm, values_min, values_max


 
class BasePINNData:
    # dimensionsunabhängige Basisklasse für Daten
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.scales: tuple[float, ...] = ()

    def _build_tensors(
        self,
        coords_col: np.ndarray,  # Shape: (N_col, D_in)   -> z.B. (2500, 2) oder (50000, 3)
        coords_data: np.ndarray,  # Shape: (N_data, D_in)  -> z.B. (250, 2)  oder (5000, 3)
        targets_data: np.ndarray,  # Shape: (N_data, D_obs) -> z.B. [u, v]    oder [u, v, w]
        coords_test: np.ndarray,  # Shape: (N_grid, D_in)
        targets_test: np.ndarray,  # Shape: (N_grid, D_out) -> z.B. [u, v, p] oder [u, v, w, p]
    ) -> None:
        self.coords_col = torch.tensor(
            coords_col, dtype=torch.float32, device=self.device
        )
        self.coords_data = torch.tensor(
            coords_data, dtype=torch.float32, device=self.device
        )
        self.targets_data = torch.tensor(
            targets_data, dtype=torch.float32, device=self.device
        )

        self.test_batch = {
            "coords": torch.tensor(
                coords_test, dtype=torch.float32, device=self.device
            ),
            "targets": torch.tensor(
                targets_test, dtype=torch.float32, device=self.device
            ),
        }

    def get_loaders(
        self, batch_size_col: int = 500, batch_size_data: int | None = None
    ) -> tuple[DataLoader, DataLoader]:
        col_dataset = TensorDataset(self.coords_col)
        col_loader = DataLoader(col_dataset, batch_size_col, shuffle=True)

        data_dataset = TensorDataset(self.coords_data, self.targets_data)
        bs_data = batch_size_data or len(data_dataset)
        data_loader = DataLoader(data_dataset, batch_size= bs_data, shuffle=True)

        return col_loader, data_loader


class WakeData2D(BasePINNData):
    def __init__(
        self, data_dir: str, z_index: int, n_train: int, n_col: int, device: str = "cpu"
    ) -> None:
        super().__init__(device=device)
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

        self.X_phys, self.Y_phys = np.meshgrid(x, y, indexing="ij") #zum plotten der physikalischen Lösung

        # without transpose it would be (y, x)
        u_true = field[self.z_index, :, :, 0].T  # u (301, 30)
        v_true = field[self.z_index, :, :, 1].T  # v (301, 30)
        p_true = np.squeeze(pressure[self.z_index]).T  # p (301, 30)
        self.true_dict = {"u": u_true, "v": v_true, "p": p_true}


        x_data = np.random.randint(0, len(x), size=self.n_train)
        y_data = np.random.randint(0, len(y), size=self.n_train)

        x_col = np.random.randint(0, len(x), size=self.n_col)
        y_col = np.random.randint(0, len(y), size=self.n_col)

        self._build_tensors(
            coords_col=np.column_stack([x_norm[x_col], y_norm[y_col]]),
            coords_data=np.column_stack([x_norm[x_data], y_norm[y_data]]),
            targets_data=np.column_stack(
                [u_true[x_data, y_data], v_true[x_data, y_data]]
            ),
            coords_test=np.column_stack([self.X.ravel(), self.Y.ravel()]),
            targets_test=np.column_stack(
                [u_true.ravel(), v_true.ravel(), p_true.ravel()]
            ),
        )

        with h5py.File(self.data_dir / "force.h5", "r") as data_file:
            force_field = np.asarray(
                data_file["force_0001"]
            )  # z,y,x, (u, v, w) (25, 30, 301, 3)
            x_force = np.asarray(data_file["xcoor"], dtype=np.float32)
            y_force = np.asarray(data_file["ycoor"], dtype=np.float32)
            
        


class KovasznayData(BasePINNData):
    def __init__(
        self, n_col: int = 2500, n_data: int = 250, re: float = 40.0, device: str = "cpu"
    ) -> None:
        super().__init__(device=device)
        self.n_col = n_col
        self.n_data = n_data
        self.re = re
        self.nu = 1.0 / re
        self.lam = 0.5 * re - np.sqrt(0.25 * (re**2) + 4.0 * (np.pi**2))
        self.scales = (1.0, 1.0)

        self._load_data()


    def _load_data(self) -> None:

        def exact_fn(x: np.ndarray, y: np.ndarray):
                u = 1.0 - np.exp(self.lam * x) * np.cos(2.0 * np.pi * y)
                v = (self.lam / (2.0 * np.pi)) * np.exp(self.lam * x) * np.sin(2.0 * np.pi * y)
                p = 0.5 * (1.0 - np.exp(2.0 * self.lam * x))
                return u, v, p
        
        # Kollokationspunkte im Gebiet + Randpunkte für den Data-Loss
        x_col = np.random.uniform(-0.5, 1.0, (self.n_col, 1))
        y_col = np.random.uniform(-0.5, 1.5, (self.n_col, 1))
    
        x_dat = np.random.uniform(-0.5, 1.0, (self.n_data, 1))
        y_dat = np.random.uniform(-0.5, 1.5, (self.n_data, 1))
        u_dat, v_dat, _ = exact_fn(x_dat, y_dat)

        x_lin = np.linspace(-0.5, 1.0, 120)
        y_lin = np.linspace(-0.5, 1.5, 120)
        self.X, self.Y = np.meshgrid(x_lin, y_lin, indexing="ij")
        u_true, v_true, p_true = exact_fn(self.X, self.Y)
        self.true_dict = {"u": u_true, "v": v_true, "p": p_true}
    
        self._build_tensors(
            coords_col=np.column_stack([x_col, y_col]),
            coords_data=np.column_stack([x_dat, y_dat]),
            targets_data=np.column_stack([u_dat, v_dat]),
            coords_test=np.column_stack([self.X.ravel(), self.Y.ravel()]),
            targets_test=np.column_stack([u_true.ravel(), v_true.ravel(), p_true.ravel()]),
        )


#[x,y,t], [u,v,p]
class WakeData2Dtime(BasePINNData):
    def __init__(
        self, data_dir: str, z_index: int, n_train: int, n_col: int, device: str = "cpu"
    ) -> None:
        super().__init__(device=device)
        self.data_dir = Path(data_dir)
        self.z_index = z_index
        self.n_train = n_train
        self.n_col = n_col
        self.dt = 0.5

        self._load_data()

    def _load_snapshots(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        """Lädt alle <prefix>_t*.h5, gibt (snapshots, field) zurück.
        field-Shape: (n_t, y, x, n_comp) – nur die Ebene self.z_index."""
        snapshots, fields = [], []
        for path in sorted(self.data_dir.glob(f"{prefix}_t*.h5")):
            with h5py.File(path, "r") as f:
                for key in sorted(k for k in f if k.startswith(f"{prefix}_")):          
                    snapshots.append(int(key.split("_")[1])) #velocity_0013 -> 13     
                    fields.append(f[key][self.z_index, :, :, :])  # nur die Ebene self.z_index
        return np.array(snapshots), np.stack(fields)


    def _load_data(self) -> None:
        t_snap, vel = self._load_snapshots("velocity")    # vel: (n_t, y, x, 3)
        t_snap_p, pres = self._load_snapshots("pressure") # pres: (n_t, y, x, 1)
        t_snap_f, force = self._load_snapshots("force")
        t_snap_visc, visc = self._load_snapshots("sgsviscosity")

        with h5py.File(self.data_dir / "velocity_t0001.h5", "r") as data_file:
                    x = np.asarray(data_file["xcoor"], dtype=np.float32)
                    y = np.asarray(data_file["ycoor"], dtype=np.float32)
        t = (t_snap - t_snap[0]) * self.dt

        with h5py.File(self.data_dir/"force_t0001.hy", "r") as f:
            x_force = np.asarray(f["xcoor"], dtype=np.float32])
            x_force_offset = np.argmin(np.abs(force[x_force, :]))
            


        x_norm, x_min, x_max = normalize_data(x)
        y_norm, y_min, y_max = normalize_data(y)
        t_norm, t_min, t_max = normalize_data(t)


        x_scale = float(2.0 / (x_max - x_min))
        y_scale = float(2.0 / (y_max - y_min))
        t_scale = float(2.0 / (t_max - t_min))
        self.scales = (x_scale, y_scale, t_scale)

        self.X, self.Y, self.T = np.meshgrid(x_norm, y_norm, t_norm, indexing="ij")  # X, Y, T (301, 30, n_t)

        self.X_phys, self.Y_phys, self.T_phys = np.meshgrid(x, y, t, indexing="ij") #zum plotten der physikalischen Lösung

        # without transpose it would be (y, x)
        u_true = vel[:, :, :, 0].T  # u (nt, 30, 301).T -> (301, 100, nt)
        v_true = vel[:, :, :, 1].T  # v (nt, 30, 301).T -> (301, 100, nt)
        p_true = np.squeeze(pres).T  # p (301, 100, nt)
        force_true = np.squeeze(force[:, :, :, :].T)  


        self.true_dict = {"u": u_true, "v": v_true, "p": p_true, "force": force_true}

        x_data = np.random.randint(0, len(x), size=self.n_train)
        y_data = np.random.randint(0, len(y), size=self.n_train)
        t_data = np.random.randint(0, len(t), size=self.n_train)

        x_col = np.random.randint(0, len(x), size=self.n_col)
        y_col = np.random.randint(0, len(y), size=self.n_col)
        t_col = np.random.randint(0, len(t), size=self.n_col)

        self._build_tensors(
            coords_col=np.column_stack([x_norm[x_col], y_norm[y_col], t_norm[t_col]]),
            coords_data=np.column_stack([x_norm[x_data], y_norm[y_data], t_norm[t_data]]),
            targets_data=np.column_stack(
                [u_true[x_data, y_data, t_data], v_true[x_data, y_data, t_data]]
            ),
            coords_test=np.column_stack([self.X.ravel(), self.Y.ravel(), self.T.ravel()]),
            targets_test=np.column_stack(
                [u_true.ravel(), v_true.ravel(), p_true.ravel()]
            ),
        )