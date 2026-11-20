import h5py
import numpy as np
from pathlib import Path

DATA_DIR = Path("..") / "data"

Z = 17
DT = 0.5

AX_T, AX_Z, AX_Y, AX_X = 0, 1, 2, 3  # Achsenindizes für die Daten (time, z, y, x)


def load(prefix):
    snaps, fields = [], []
    for path in sorted(DATA_DIR.glob(f"{prefix}*.h5")):
        with h5py.File(path, "r") as f:
            for key in sorted(k for k in f if k.startswith(f"{prefix}_")):
                snaps.append(int(key.split("_")[1]))
                fields.append(f[key][Z - 2 : Z + 3])
    return np.array(snaps), np.stack(fields)


def d(f, coord, axis):
    """Berechnet die Ableitung der Funktion f an den Koordinaten coord entlang der Achse axis."""
    df = np.gradient(f, coord, axis=axis)
    return df


def rms(a):
    # rms nur über innere Punkte(am rand über  einseitife differenzen)
    return np.sqrt(np.mean(a[1:-1, 2, 1:-1, 1:-1] ** 2))


snaps, vel = load("velocity")
_, pres = load("pressure")
_, force = load("force")
_, nu_t = load("sgsviscosity")


with h5py.File(DATA_DIR / "velocity_t0001.h5", "r") as f:
    x, y, z = (np.asarray(f[k], dtype=np.float64) for k in ("xcoor", "ycoor", "zcoor"))
z = z[Z - 2 : Z + 3]
t = (snaps - snaps[0]) * DT


# TODO: force auf das volle x-Gitter auffüllen (wie in data.py)
with h5py.File(DATA_DIR / "force_t0001.h5", "r") as f:
    x_force = np.asarray(f["xcoor"], dtype=np.float32)
    x_force_offset = len(x) - len(x_force)
force_pad = np.zeros_like(vel)
force_pad[:, :, :, x_force_offset:, :] = force

# TODO: u, v, w, p, f_x, f_y, nu als 4D-Arrays (t, z, y, x) herausziehen
u = vel[:, :, :, :, 0]
v = vel[:, :, :, :, 1]
w = vel[:, :, :, :, 2]
p = pres.squeeze()
f_x = force_pad[:, :, :, :, 0]
f_y = force_pad[:, :, :, :, 1]
nu = nu_t.squeeze()


# --- 2. Ableitungen ---
# TODO: u_t, u_x, u_y, u_z, u_xx, u_yy, u_zz, dasselbe für v
# TODO: w_x, w_z, p_x, p_y, nu_x, nu_y, nu_z


u_x = d(u, x, AX_X)
u_y = d(u, y, AX_Y)
u_z = d(u, z, AX_Z)
u_t = d(u, t, AX_T)
u_xx = d(u_x, x, AX_X)
u_yy = d(u_y, y, AX_Y)
u_zz = d(u_z, z, AX_Z)
v_x = d(v, x, AX_X)
v_y = d(v, y, AX_Y)
v_z = d(v, z, AX_Z)
v_t = d(v, t, AX_T)
v_xx = d(v_x, x, AX_X)
v_yy = d(v_y, y, AX_Y)
v_zz = d(v_z, z, AX_Z)
w_x = d(w, x, AX_X)
w_z = d(w, z, AX_Z)
p_x = d(p, x, AX_X)
p_y = d(p, y, AX_Y)
nu_x = d(nu, x, AX_X)
nu_y = d(nu, y, AX_Y)
nu_z = d(nu, z, AX_Z)

# --- 3. Terme der x-Impulsgleichung einzeln ("Term-Budget") ---
terms_x = {
    "u_t": u_t,
    "konvektion": u * u_x + v * u_y,  # u*u_x + v*u_y
    "p_x": p_x,
    "f_x": f_x,
    "nu*lap2d": nu * (u_xx + u_yy),  # nu*(u_xx + u_yy)
    "nu-gradient": 2 * nu_x * u_x + nu_y * (u_y + v_x),  # 2*nu_x*u_x + nu_y*(u_y + v_x)
    "w*u_z": w * u_z,  # w*u_z
    "z-viskos": nu * u_zz + nu_z * (u_z + w_x),  # nu*u_zz + nu_z*(u_z + w_x)
}

for name, val in terms_x.items():
    print(f"{name:>12}: {rms(val):.3e}")

# --- 4. Residuen der vier Varianten, relativ zur Konvektion ---
ref = rms(terms_x["konvektion"])
R1 = terms_x["u_t"] + terms_x["konvektion"] + terms_x["p_x"] - terms_x["f_x"]  # nu = 0
R2 = (
    terms_x["u_t"]
    + terms_x["konvektion"]
    + terms_x["p_x"]
    - terms_x["f_x"]
    - nu.mean() * (u_xx + u_yy)
)  # nu = konstant (z.B. nu.mean())
R3 = (
    terms_x["u_t"]
    + terms_x["konvektion"]
    + terms_x["p_x"]
    - terms_x["f_x"]
    - terms_x["nu*lap2d"]
    - terms_x["nu-gradient"]
)  # nu = nu_t(x,y,t) inkl. Gradiententerm
R4 = (
    terms_x["u_t"]
    + terms_x["konvektion"]
    + w * u_z
    + terms_x["p_x"]
    - terms_x["f_x"]
    - terms_x["nu*lap2d"]
    - terms_x["nu-gradient"]
    - terms_x["z-viskos"]
)  # R3 + z-Terme
for name, R in [("nu=0", R1), ("nu=konst", R2), ("nu_t", R3), ("nu_t+3D", R4)]:
    print(f"{name:>10}: {rms(R) / ref:.3f}")

    # --- 5. Kontinuität ---
# TODO: rms(u_x + v_y) und rms(u_x + v_y + w_z), jeweils relativ zu rms(u_x)
ref = rms(u_x)
R1 = u_x + v_y
R2 = u_x + v_y + w_z
for name, R in [("2D", R1), ("3D", R2)]:
    print(f"{name:>10}: {rms(R) / ref:.3f}")
