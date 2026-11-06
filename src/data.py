import numpy as np
import torch



def load_kovasznay(n_col: int = 2500, n_data: int = 250, re: float = 40.0, device: str = "cpu"):
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
        "x_col": torch.tensor(x_col, dtype=torch.float32, device=device, requires_grad=True),
        "y_col": torch.tensor(y_col, dtype=torch.float32, device=device, requires_grad=True),
        "x_data": torch.tensor(x_dat, dtype=torch.float32, device=device),
        "y_data": torch.tensor(y_dat, dtype=torch.float32, device=device),
        "uv_data": torch.tensor(np.hstack([u_dat, v_dat]), dtype=torch.float32, device=device),
    }

    x_lin = np.linspace(-0.5, 1.0, 120)
    y_lin = np.linspace(-0.5, 1.5, 120)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    u_true, v_true, p_true = exact_fn(X, Y)

    # Tipp: Du kannst dasselbe Gitter direkt für val_batch UND für den Plot am Ende nutzen!
    val_batch = {
        "coords": torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32, device=device),
        "u_true": torch.tensor(np.stack([u_true.ravel(), v_true.ravel(), p_true.ravel()], axis=-1), dtype=torch.float32, device=device),
    }
    true_dict = {"u": u_true, "v": v_true, "p": p_true}

    return train_batch, val_batch, (X, Y, true_dict)


def get_wake_2d_data(data_dir: str, z_index: int, n_data: int, n_col: int, device: str):
    # 1. HDF5 laden & bei z_index schneiden -> X, Y, u_true, v_true, p_true
    # 2. Zufällige Indizes für train_batch ziehen (n_data Punkte für Data-Loss, n_col für PDE)
    # 3. Gesamtes Gitter in val_batch packen
    # 4. true_dict = {"u": u_true, "v": v_true, "p": p_true}
    return train_batch, val_batch, (X, Y, true_dict)