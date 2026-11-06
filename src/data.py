import numpy as np
import torch
import h5py
from pathlib import Path

DATA_DIR = Path("data")


def normalize_data(values: np.array) -> np.array:
    """
    scale to [-1, 1]
    """
    values_min = values.min()
    values_max = values.max()
    values_norm = 2.0 * (values - values_min) / (values_max - values_min) - 1.0

    return values_norm, values_min, values_max


def load_kovasznay(
    n_col: int = 2500, n_data: int = 250, re: float = 40.0, device: str = "cpu"
):
    nu = 1.0 / re
    lam = 0.5 * re - np.sqrt(0.25 * (re**2) + 4.0 * (np.pi**2))

    def exact_fn(x: np.ndarray, y: np.ndarray):
        u = 1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y)
        v = (lam / (2.0 * np.pi)) * np.exp(lam * x) * np.sin(2.0 * np.pi * y)
        p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
        return u, v, p

    # Kollokationspunkte im Gebiet + Randpunkte für den Data-Loss
    x_col = np.random.uniform(-0.5, 1.0, (n_col, 1))
    y_col = np.random.uniform(-0.5, 1.5, (n_col, 1))

    x_dat = np.random.uniform(-0.5, 1.0, (n_data, 1))
    y_dat = np.random.uniform(-0.5, 1.5, (n_data, 1))
    u_dat, v_dat, _ = exact_fn(x_dat, y_dat)

    train_batch = {
        "x_col": torch.tensor(
            x_col, dtype=torch.float32, device=device, requires_grad=True
        ),
        "y_col": torch.tensor(
            y_col, dtype=torch.float32, device=device, requires_grad=True
        ),
        "x_data": torch.tensor(x_dat, dtype=torch.float32, device=device),
        "y_data": torch.tensor(y_dat, dtype=torch.float32, device=device),
        "uv_data": torch.tensor(
            np.hstack([u_dat, v_dat]), dtype=torch.float32, device=device
        ),
    }

    x_lin = np.linspace(-0.5, 1.0, 120)
    y_lin = np.linspace(-0.5, 1.5, 120)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    u_true, v_true, p_true = exact_fn(X, Y)

    # Tipp: Du kannst dasselbe Gitter direkt für val_batch UND für den Plot am Ende nutzen!
    val_batch = {
        "coords": torch.tensor(
            np.stack([X.ravel(), Y.ravel()], axis=-1),
            dtype=torch.float32,
            device=device,
        ),
        "uvp_true": torch.tensor(
            np.stack([u_true.ravel(), v_true.ravel(), p_true.ravel()], axis=-1),
            dtype=torch.float32,
            device=device,
        ),
    }
    true_dict = {"u": u_true, "v": v_true, "p": p_true}

    return train_batch, val_batch, (X, Y, true_dict)


def get_wake_2d_data(
    data_dir: str, z_index: int, n_train: int, n_col: int, device: str, 
):
    # 1. HDF5 laden & bei z_index schneiden -> X, Y, u_true, v_true, p_true
    # 2. Zufällige Indizes für train_batch ziehen (n_data Punkte für Data-Loss, n_col für PDE)
    # 3. Gesamtes Gitter in val_batch packen
    # 4. true_dict = {"u": u_true, "v": v_true, "p": p_true}

    with h5py.File(DATA_DIR / "meanvelocity.h5", "r") as data_file:
        field = np.asarray(
            data_file["meanvelocity_0001"]
        )  # z,y,x, (u, v, w) (25, 30, 301, 3)
        coordinates = tuple(
            np.asarray(data_file[name]) for name in ("xcoor", "ycoor")
        )  # x, y, z np_arrays (301, 30, 25)      -> 225.750 points

    with h5py.File(DATA_DIR / "meanpressure.h5", "r") as data_file:
        pressure = np.asarray(data_file["meanpressure_0001"])  # z,y,x (25, 30, 301)

    x = coordinates[0].astype(np.float32)  # x (301,)
    y = coordinates[1].astype(np.float32)  # y (30,)

    x_norm, x_min, x_max = normalize_data(x)
    y_norm, y_min, y_max = normalize_data(y)

    x_scale = float(2.0 / (x_max - x_min))
    y_scale = float(2.0 / (y_max - y_min))

    X_norm, Y_norm = np.meshgrid(x_norm, y_norm, indexing="ij")  # X, Y (30, 301)

    # without transpose it would be (y, x)
    u_true = field[z_index, :, :, 0].T  # u (301, 30)
    v_true = field[z_index, :, :, 1].T  # v (301, 30)
    p_true = np.squeeze(pressure[z_index]).T  # p (301, 30)

    x_data = np.random.randint(0, len(x), size=n_train)
    y_data = np.random.randint(0, len(y), size=n_train)

    x_col = np.random.randint(0, len(x), size=n_col)
    y_col = np.random.randint(0, len(y), size=n_col)

    true_dict = {"u": u_true, "v": v_true, "p": p_true}

    train_batch = {
        "x_col": torch.tensor(x_norm[x_col], device=device, requires_grad=True).reshape(
            -1, 1
        ),
        "y_col": torch.tensor(y_norm[y_col], device=device, requires_grad=True).reshape(
            -1, 1
        ),
        "x_data": torch.tensor(x_norm[x_data], device=device).reshape(-1, 1),
        "y_data": torch.tensor(y_norm[y_data], device=device).reshape(-1, 1),
        # only train on uv data, p is only learned via PDE
        "uv_data": torch.tensor(
            np.stack(
                [
                    u_true[x_data, y_data],
                    v_true[x_data, y_data],
                ],
                axis=-1,
            ),
            dtype=torch.float32,
            device=device,
        ),
        "x_scale": x_scale,
        "y_scale": y_scale,
    }

    val_batch = {
        "coords": torch.tensor(
            np.stack([X_norm.ravel(), Y_norm.ravel()], axis=-1),
            dtype=torch.float32,
            device=device,
        ),
        "uvp_true": torch.tensor(
            np.stack([u_true.ravel(), v_true.ravel(), p_true.ravel()], axis=-1),
            dtype=torch.float32,
            device=device,
        ),
    }

    return train_batch, val_batch, (X_norm, Y_norm, true_dict)
