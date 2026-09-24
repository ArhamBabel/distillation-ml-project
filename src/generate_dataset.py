import numpy as np
from scipy.stats import qmc
import pandas as pd
from distillation_physics import compute_distillation_design, compute_energy

TARGET_ROWS = 5000
SEED = 42

# LHS-sampled variables (5 dimensions now — alpha and lam moved to system-based sampling)
bounds = {
    "z_F":      (0.2, 0.8),
    "x_D":      (0.90, 0.995),
    "x_B":      (0.01, 0.05),
    "R_factor": (1.1, 2.5),
    "F":        (100, 500),
}

# Representative chemical systems — ties alpha and lam together realistically
# (Project assumption: general tendency, not derived from specific real components)
SYSTEMS = {
    1: {"alpha_range": (1.5, 2.0), "lam_range": (32000, 35000)},
    2: {"alpha_range": (2.0, 2.5), "lam_range": (29000, 32000)},
    3: {"alpha_range": (2.5, 3.0), "lam_range": (27000, 30000)},
    4: {"alpha_range": (3.0, 3.5), "lam_range": (26000, 28000)},
    5: {"alpha_range": (3.5, 4.0), "lam_range": (25000, 27000)},
}


def generate_dataset(target_rows=TARGET_ROWS, seed=SEED):
    sampler = qmc.LatinHypercube(d=5, seed=seed)  # created ONCE, outside the loop
    rng = np.random.default_rng(seed)              # for system_id + alpha/lam sampling

    l_bounds = [bounds[k][0] for k in bounds]
    u_bounds = [bounds[k][1] for k in bounds]

    rows = []
    total_attempted = 0
    batch_size = target_rows  # how many samples to draw per round

    while len(rows) < target_rows:
        raw_samples = sampler.random(n=batch_size)          # values in [0,1]^5
        scaled = qmc.scale(raw_samples, l_bounds, u_bounds)  # stretched to real ranges

        for sample in scaled:
            z_F, x_D, x_B, R_factor, F = sample
            total_attempted += 1

            # --- System-based alpha/lam sampling ---
            system_id = rng.choice(list(SYSTEMS.keys()))
            alpha_low, alpha_high = SYSTEMS[system_id]["alpha_range"]
            lam_low, lam_high = SYSTEMS[system_id]["lam_range"]
            alpha = rng.uniform(alpha_low, alpha_high)
            lam = rng.uniform(lam_low, lam_high)

            # --- Purity layer (Fenske-Underwood-Gilliland) ---
            result = compute_distillation_design(z_F, x_D, x_B, alpha, R_factor)

            if result is not None:
                # --- Energy layer ---
                Q_reb = compute_energy(F, result["R"], lam, z_F, x_D, x_B)

                result["system_id"] = system_id
                result["F"] = F
                result["lam"] = lam
                result["Q_reb"] = Q_reb
                rows.append(result)

            if len(rows) >= target_rows:
                break

    df = pd.DataFrame(rows[:target_rows])

    rejection_rate = 1 - (len(df) / total_attempted)
    print(f"Total sampled: {total_attempted}, Valid rows: {len(df)}, Rejection rate: {rejection_rate:.2%}")

    return df


if __name__ == "__main__":
    df = generate_dataset(target_rows=5000)
    print(df.shape)
    print(df.head())
    print(df.describe())

    df.to_csv("../data/distillation_purity_dataset.csv", index=False)
    print("Saved to data/distillation_purity_dataset.csv")