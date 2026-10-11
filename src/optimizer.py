import joblib

model_xD = joblib.load("model_xD.joblib")
model_Qreb = joblib.load("model_Qreb.joblib")

# Search bounds, based on the training data's distribution (data/distillation_purity_dataset.csv)
R_BOUNDS = (0.5, 10.0)
N_BOUNDS = (6.0, 25.0)
XB_BOUNDS = (0.01, 0.05)

# Per-system alpha, lam taken as the midpoint of each system's sampling range
# (see SYSTEMS dict in src/generate_dataset.py)
SYSTEMS = {
    1: {"alpha": 1.75, "lam": 33500},
    2: {"alpha": 2.25, "lam": 30500},
    3: {"alpha": 2.75, "lam": 28500},
    4: {"alpha": 3.25, "lam": 27000},
    5: {"alpha": 3.75, "lam": 26000},
}


def optimize(system_id, z_F, F, N, purity_spec, n_R=50, n_xB=50):
    """
    Find the reflux ratio (R), number of trays (N), and bottoms
    composition (x_B) that minimize predicted reboiler duty (Q_reb),
    subject to predicted distillate purity (x_D) >= purity_spec.

    Fixed inputs:
        system_id    : 1-5, selects alpha and lam from SYSTEMS
        z_F          : feed composition (mole fraction light key)
        F            : feed flow rate
    Decision variables (searched over a grid):
        R  in R_BOUNDS
        N  in N_BOUNDS
        x_B in XB_BOUNDS
    Other parameters:
        purity_spec  : minimum required x_D
        n_R, n_N, n_xB : number of grid points per dimension

    Returns:
        TODO (decided in the next step)
    """
    import numpy as np

    alpha = SYSTEMS[system_id]["alpha"]
    lam = SYSTEMS[system_id]["lam"]

    R_grid = np.linspace(R_BOUNDS[0], R_BOUNDS[1], n_R)
    xB_grid = np.linspace(XB_BOUNDS[0], XB_BOUNDS[1], n_xB)

    R_mesh, xB_mesh = np.meshgrid(R_grid, xB_grid, indexing="ij")
 
    R_flat = R_mesh.ravel()
    N_flat = np.full(R_mesh.ravel().shape, N)
    xB_flat = xB_mesh.ravel()

    n_points = len(R_flat)

    import pandas as pd

    X_xD = pd.DataFrame({
        "R": R_flat,
        "N": N_flat,
        "z_F": np.full(n_points, z_F),
        "alpha": np.full(n_points, alpha),
        "F": np.full(n_points, F),
        "x_B": xB_flat,
    })

    xD_pred = model_xD.predict(X_xD)

    feasible_mask = xD_pred >= purity_spec

    R_feas = R_flat[feasible_mask]
    N_feas = N_flat[feasible_mask]
    xB_feas = xB_flat[feasible_mask]
    n_feas = len(R_feas)

    if n_feas == 0:
        return {"feasible": False, "message": f"No combination met x_D >= {purity_spec} within the search bounds, at N={N}."}

    X_Qreb = pd.DataFrame({
        "R": R_feas,
        "N": N_feas,
        "z_F": np.full(n_feas, z_F),
        "alpha": np.full(n_feas, alpha),
        "F": np.full(n_feas, F),
        "lam": np.full(n_feas, lam),
        "x_D": xD_pred[feasible_mask],
        "x_B": xB_feas,
    })

    Qreb_log_pred = model_Qreb.predict(X_Qreb)
    Qreb_pred = np.exp(Qreb_log_pred)

    if n_feas == 0:
        return {"feasible": False, "message": f"No combination met x_D >= {purity_spec} within the search bounds."}

    best_idx = np.argmin(Qreb_pred)

    result = {
        "feasible": True,
        "R": R_feas[best_idx],
        "N": N_feas[best_idx],
        "x_B": xB_feas[best_idx],
        "x_D_pred": xD_pred[feasible_mask][best_idx],
        "Q_reb_pred": Qreb_pred[best_idx],
    }

    return result

   