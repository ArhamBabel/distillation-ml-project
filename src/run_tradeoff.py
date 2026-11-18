from pathlib import Path
from optimizer import n_sweep
import matplotlib.pyplot as plt


# Same standard test case and N values as the original
N_VALUES = [8, 10, 12, 14, 15, 16, 17, 18, 20, 25]

results = n_sweep(system_id=1, z_F=0.5, F=300, purity_spec=0.95, N_values=N_VALUES)

for res in results:
    print(res)

feasible_N = [r["N"] for r in results if r["feasible"]]
feasible_Q = [r["Q_reb_pred"] for r in results if r["feasible"]]

plt.figure(figsize=(7, 5))
plt.plot(feasible_N, feasible_Q, marker="o")
plt.xlabel("Number of trays (N)")
plt.ylabel("Minimum predicted reboiler duty, Q_reb (kJ/h)")
plt.title("Energy\u2013trays trade-off at x_D >= 0.95 (system 1, z_F=0.5, F=300)")
plt.grid(True)
plt.tight_layout()

_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
plt.savefig(_DOCS_DIR / "qreb_vs_N_tradeoff.png")
plt.show()
