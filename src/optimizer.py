import numpy as np
import pandas as pd
import joblib
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent

model_xD   = joblib.load(_MODEL_DIR / "model_xD.joblib")
model_Qreb = joblib.load(_MODEL_DIR / "model_Qreb.joblib")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Absolute R and x_B search bounds — derived from training-data ranges.
R_BOUNDS  = (0.5, 10.0)
XB_BOUNDS = (0.01, 0.05)

# Training data was generated as R = R_factor × R_min, with R_factor
# sampled uniformly in [1.1, 2.5] (see generate_dataset.py).
# The ML surrogate is only reliable within this R_factor band:
#   - R_factor < 1.1  →  R approaches or falls below R_min; physically
#     infeasible and completely outside training support.
#   - R_factor > 2.5  →  beyond training support; surrogate under-predicts
#     purity by ~0.02 mole fraction.
R_FACTOR_BOUNDS = (1.1, 2.5)

# Tolerance applied during independent FUG physical validation.
#
# Set to 0.0 for strict physical validation: a candidate is accepted only if
# the FUG-calculated theoretical tray requirement satisfies FUG_N <= N_actual.
FUG_N_TOL = 0.0

# Per-system (alpha, lam) midpoints — from generate_dataset.py SYSTEMS dict.
SYSTEMS = {
    1: {"alpha": 1.75, "lam": 33500},
    2: {"alpha": 2.25, "lam": 30500},
    3: {"alpha": 2.75, "lam": 28500},
    4: {"alpha": 3.25, "lam": 27000},
    5: {"alpha": 3.75, "lam": 26000},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_R_min(purity_spec, z_F, alpha):
    """
    Minimum reflux ratio for a binary system (Underwood shortcut,
    saturated-liquid feed q = 1).

    Replicates the formula used in distillation_physics.py:

        y_q   = alpha * z_F / (1 + (alpha - 1) * z_F)
        R_min = (purity_spec - y_q) / (y_q - z_F)

    Returns R_min (float) if the specification is thermodynamically
    meaningful, or None if the input is degenerate (y_q ≈ z_F, or
    purity_spec ≤ y_q which implies R_min ≤ 0).
    """
    y_q   = alpha * z_F / (1.0 + (alpha - 1.0) * z_F)
    denom = y_q - z_F
    if abs(denom) < 1e-10:
        return None                   # degenerate feed — alpha ≈ 1
    R_min = (purity_spec - y_q) / denom
    if R_min <= 0.0:
        return None                   # spec ≤ y_q: trivially easy or invalid
    return R_min


def _fug_validate(z_F, purity_spec, x_B, alpha, R, N_actual):
    """
    Independently verify physical feasibility of a recommended design
    using the FUG shortcut equations.  Does NOT use the ML surrogate.

    Procedure:
        1. Compute R_min for the target purity_spec.
        2. If R ≤ R_min the Underwood condition is violated — infeasible.
        3. Compute R_factor = R / R_min and call
           distillation_physics.compute_distillation_design at
           x_D = purity_spec to obtain the required number of trays.
        4. The design is physically feasible iff
               FUG_N ≤ N_actual + FUG_N_TOL
           Under strict validation (FUG_N_TOL = 0.0), this requires
           FUG_N ≤ N_actual, ensuring the column has at least as many
           trays as required by the FUG shortcut model.

    Returns
    -------
    (feasible : bool, fug_N : float or None)
    """
    from distillation_physics import compute_distillation_design

    R_min = _compute_R_min(purity_spec, z_F, alpha)
    if R_min is None or R <= R_min:
        return False, None

    result = compute_distillation_design(
        z_F, purity_spec, x_B, alpha, R_factor=R / R_min
    )
    if result is None:
        return False, None

    fug_N = result["N"]
    return fug_N <= N_actual + FUG_N_TOL, fug_N


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(system_id, z_F, F, N, purity_spec, n_R=50, n_xB=50):
    """
    Find the reflux ratio (R) and bottoms composition (x_B) that minimise
    predicted reboiler duty (Q_reb), subject to predicted distillate purity
    (x_D) ≥ purity_spec, at a fixed number of trays (N).

    Fixed inputs
    ------------
    system_id : 1–5  — selects alpha and lam from SYSTEMS
    z_F       : feed composition (mole fraction light key)
    F         : feed flow rate (kmol/h)
    N         : number of trays (fixed per call)

    Decision variables (searched over a training-supported grid)
    ------------------------------------------------------------
    R   ∈ [max(R_BOUNDS[0], 1.1·R_min),  min(R_BOUNDS[1], 2.5·R_min)]
    x_B ∈ XB_BOUNDS

    where R_min is the Underwood minimum reflux for the requested
    purity_spec.  This keeps every grid point within the R_factor range
    [1.1, 2.5] on which the ML surrogate was trained.

    purity_spec : minimum required x_D
    n_R, n_xB  : grid resolution per dimension (default 50 × 50 = 2 500 pts)

    Physical validation
    -------------------
    After ML screening, candidates are sorted by ascending predicted Q_reb.
    Each is independently verified with the FUG shortcut equations via
    _fug_validate().  The first candidate that passes both the ML purity
    filter and FUG physical validation is returned.

    Returns
    -------
    dict with:
      "feasible"   : bool
      "R"          : float    — optimal reflux ratio          (if feasible)
      "N"          : int/float — trays (the caller-supplied N) (always)
      "x_B"        : float    — optimal bottoms composition   (if feasible)
      "x_D_pred"   : float    — ML-predicted distillate purity(if feasible)
      "Q_reb_pred" : float    — predicted reboiler duty kJ/h  (if feasible)
      "fug_N"      : float    — FUG-computed N at purity_spec (if feasible)
      "message"    : str      — reason for infeasibility      (if infeasible)
    """
    alpha = SYSTEMS[system_id]["alpha"]
    lam   = SYSTEMS[system_id]["lam"]

    # ------------------------------------------------------------------
    # 1. Supported R grid bounds derived from R_min(purity_spec)
    # ------------------------------------------------------------------
    R_min = _compute_R_min(purity_spec, z_F, alpha)
    if R_min is None:
        return {
            "feasible": False,
            "N": N,
            "message": (
                f"Cannot compute R_min for purity_spec={purity_spec}, z_F={z_F}, "
                f"system_id={system_id}.  Verify that purity_spec > z_F and that "
                f"the feed composition is physically valid."
            ),
        }

    R_low  = max(R_BOUNDS[0],  R_FACTOR_BOUNDS[0] * R_min)
    R_high = min(R_BOUNDS[1],  R_FACTOR_BOUNDS[1] * R_min)

    if R_low >= R_high:
        return {
            "feasible": False,
            "N": N,
            "message": (
                f"No valid R search region: lower training-support bound "
                f"R_low={R_low:.4f} ≥ upper bound R_high={R_high:.4f} "
                f"(R_min={R_min:.4f}, system_id={system_id}).  "
                f"The purity specification may be too stringent for this system "
                f"within the trained operating envelope."
            ),
        }

    # ------------------------------------------------------------------
    # 2. Build the R × x_B search grid
    # ------------------------------------------------------------------
    R_grid  = np.linspace(R_low,         R_high,         n_R)
    xB_grid = np.linspace(XB_BOUNDS[0],  XB_BOUNDS[1],   n_xB)

    R_mesh, xB_mesh = np.meshgrid(R_grid, xB_grid, indexing="ij")
    R_flat  = R_mesh.ravel()
    xB_flat = xB_mesh.ravel()
    n_pts   = len(R_flat)

    # ------------------------------------------------------------------
    # 3. ML prediction for x_D across all grid points
    # ------------------------------------------------------------------
    X_xD = pd.DataFrame({
        "R":     R_flat,
        "N":     np.full(n_pts, N),
        "z_F":   np.full(n_pts, z_F),
        "alpha": np.full(n_pts, alpha),
        "F":     np.full(n_pts, F),
        "x_B":   xB_flat,
    })
    xD_pred = model_xD.predict(X_xD)

    # ------------------------------------------------------------------
    # 4. ML purity filter
    # ------------------------------------------------------------------
    ml_mask = xD_pred >= purity_spec
    R_feas  = R_flat[ml_mask]
    xB_feas = xB_flat[ml_mask]
    xD_feas = xD_pred[ml_mask]

    if len(R_feas) == 0:
        return {
            "feasible": False,
            "N": N,
            "message": (
                f"No grid point achieved x_D ≥ {purity_spec} according to the ML "
                f"surrogate (R ∈ [{R_low:.3f}, {R_high:.3f}], "
                f"x_B ∈ {XB_BOUNDS}, N={N})."
            ),
        }

    # ------------------------------------------------------------------
    # 5. ML prediction for Q_reb on the purity-feasible subset
    # ------------------------------------------------------------------
    n_feas = len(R_feas)
    X_Qreb = pd.DataFrame({
        "R":     R_feas,
        "N":     np.full(n_feas, N),
        "z_F":   np.full(n_feas, z_F),
        "alpha": np.full(n_feas, alpha),
        "F":     np.full(n_feas, F),
        "lam":   np.full(n_feas, lam),
        "x_D":   xD_feas,
        "x_B":   xB_feas,
    })
    Qreb_pred = np.exp(model_Qreb.predict(X_Qreb))

    # ------------------------------------------------------------------
    # 6. Independent FUG validation — lowest Q_reb first
    # ------------------------------------------------------------------
    for idx in np.argsort(Qreb_pred):
        fug_ok, fug_N = _fug_validate(
            z_F, purity_spec, xB_feas[idx], alpha, R_feas[idx], N
        )
        if fug_ok:
            return {
                "feasible":   True,
                "R":          float(R_feas[idx]),
                "N":          N,
                "x_B":        float(xB_feas[idx]),
                "x_D_pred":   float(xD_feas[idx]),
                "Q_reb_pred": float(Qreb_pred[idx]),
                "fug_N":      float(fug_N),
            }

    return {
        "feasible": False,
        "N": N,
        "message": (
            f"All {n_feas} ML-feasible candidate(s) failed independent FUG "
            f"physical validation at N={N} (purity_spec={purity_spec}, "
            f"system_id={system_id}).  The ML surrogate may be over-predicting "
            f"purity in this region."
        ),
    }


def predict(system_id, z_F, F, R, N, x_B):
    """
    Predict distillate purity (x_D) and reboiler duty (Q_reb) for a
    single operating point using the trained RF surrogate models.

    Parameters
    ----------
    system_id : 1–5  selects alpha and lam from SYSTEMS
    z_F       : feed composition (mole fraction light key)
    F         : feed flow rate (kmol/h)
    R         : reflux ratio (dimensionless)
    N         : number of trays
    x_B       : bottoms composition (mole fraction light key)

    Returns
    -------
    dict with keys x_D_pred (float) and Q_reb_pred (float, kJ/h).
    """
    alpha = SYSTEMS[system_id]["alpha"]
    lam   = SYSTEMS[system_id]["lam"]

    # x_D model feature order: [R, N, z_F, alpha, F, x_B]
    x_D_pred = float(model_xD.predict(pd.DataFrame([{
        "R": R, "N": N, "z_F": z_F, "alpha": alpha, "F": F, "x_B": x_B,
    }]))[0])

    # Q_reb model feature order: [R, N, z_F, alpha, F, lam, x_D, x_B]
    Q_reb_pred = float(np.exp(model_Qreb.predict(pd.DataFrame([{
        "R": R, "N": N, "z_F": z_F, "alpha": alpha, "F": F,
        "lam": lam, "x_D": x_D_pred, "x_B": x_B,
    }]))[0]))

    return {"x_D_pred": x_D_pred, "Q_reb_pred": Q_reb_pred}


def n_sweep(system_id, z_F, F, purity_spec, N_values):
    """
    Run optimize() at each N in N_values and collect results.

    Parameters
    ----------
    system_id   : 1–5
    z_F         : feed composition (mole fraction light key)
    F           : feed flow rate (kmol/h)
    purity_spec : minimum required x_D
    N_values    : iterable of tray counts to sweep

    Returns
    -------
    list of dicts — one per N value, each the optimize() result dict
    with "N" always present.
    """
    results = []
    for N in N_values:
        res = optimize(system_id, z_F, F, N, purity_spec)
        res["N"] = N          # guarantee "N" is present even if infeasible
        results.append(res)
    return results
