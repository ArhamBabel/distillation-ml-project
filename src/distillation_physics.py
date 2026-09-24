import numpy as np

def compute_distillation_design(z_F, x_D, x_B, alpha, R_factor):
    """
    Given target purities and separation parameters, compute the 
    required reflux ratio (R) and number of trays (N) using the 
    Fenske-Underwood-Gilliland shortcut method.

    Parameters:
        z_F      : feed composition (mole fraction light key)
        x_D      : target distillate purity
        x_B      : target bottoms impurity (light key in bottoms)
        alpha    : relative volatility
        R_factor : multiplier over R_min (e.g. 1.1 - 2.5)

    Returns:
        dict with R, N, z_F, x_D, x_B, alpha (or None if invalid/infeasible)
    """

    # --- Step 0: Basic validity checks ---
    #  check x_B < z_F < x_D
    if not (x_B < z_F < x_D):
        return None
    

    # --- Step 1: Minimum reflux (binary shortcut) ---
    #  compute y_q using the q-line/equilibrium intersection formula
    y_q = (alpha * z_F) / (1 + (alpha - 1) * z_F) 


    # TODO: guard against near-zero denominator (y_q - z_F)
    if abs(y_q - z_F) < 1e-6:
        return None

    #  compute R_min
    R_min = (x_D - y_q) / (y_q - z_F)

    # --- Step 2: Actual reflux ratio ---
    R = R_factor * R_min  

    if R < 0.5:
        return None  
    

    

    # --- Step 3: Minimum trays (Fenske equation) ---
    #  compute N_min
    N_min = np.log((x_D / (1 - x_D)) * ((1 - x_B) / x_B)) / np.log(alpha)

    # --- Step 4: Gilliland correlation (Molokanov form) ---
    #  compute X
    X = (R - R_min) / (R + 1)

    #  guard against X outside (0, 1) - reject row if so
    if not (0 < X < 1):
        return None

    #  compute Y using the Molokanov equation
    Y = 1 - np.exp(((1 + 54.4 * X) / (11 + 117.2 * X)) * ((X - 1) / np.sqrt(X)))

    if abs(1 - Y) < 1e-6:
        return None
    

    #  compute N from N_min and Y
    N = (N_min + Y) /(1 - Y)

    # --- Step 5: Return the consistent row ---
    return {
        "z_F": z_F,
        "x_D": x_D,
        "x_B": x_B,
        "alpha": alpha,
        "R": R,   
        "N": N
    }

def compute_energy(F, R, lam, z_F, x_D, x_B):
    """
    Reboiler duty from a material balance on the column.

    Assumptions (project assumptions, see docs/assumptions.md):
      - constant molar overflow (CMO)
      - saturated-liquid feed (q = 1), so reboiler boilup = V

    Parameters:
    F     : feed flow rate (kmol/h)
    R     : reflux ratio L/D (dimensionless)
    lam   : molar latent heat of vaporization (kJ/kmol)
    z_F   : feed composition, mole fraction of light key (-)
    x_D   : distillate purity, mole fraction of light key (-)
    x_B   : light key in bottoms, mole fraction (-)

Returns:
    Q_reb : reboiler duty (kJ/h)
    """
    #  distillate flow D from the overall + component balance
    D = F * (z_F - x_B) / (x_D - x_B)

    #  vapor rate V in the rectifying section
    V = (R+1) * D 

    #  reboiler duty Q_reb
    Q_reb = V * lam

    return Q_reb

# --- Quick manual test (fill in after implementing above) ---
if __name__ == "__main__":
    result = compute_distillation_design(
        z_F=0.5, x_D=0.98, x_B=0.02, alpha=2.5, R_factor=1.5
    )
    print(result)