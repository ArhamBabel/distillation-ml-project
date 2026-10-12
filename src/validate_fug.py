from distillation_physics import compute_distillation_design

def r_mismatch(R_factor, z_F, x_D_guess, x_B, alpha, target_R):
    result = compute_distillation_design(z_F, x_D_guess, x_B, alpha, R_factor)
    if result is None:
        return None
    return result["R"] - target_R


from scipy.optimize import brentq

z_F = 0.5
x_B = 0.046734693877551026
alpha = 1.75
target_R = 4.571428571428571
target_N = 16
x_D_guess = 0.9509042634594431

R_factor_solution = brentq(
    r_mismatch, 1.01, 10.0,
    args=(z_F, x_D_guess, x_B, alpha, target_R)
)

fug_result = compute_distillation_design(z_F, x_D_guess, x_B, alpha, R_factor_solution)

print(f"R_factor found: {R_factor_solution:.4f}")
print(f"FUG's R: {fug_result['R']:.4f}  (target was {target_R:.4f})")
print(f"FUG's N: {fug_result['N']:.4f}  (optimizer assumed N={target_N})")