# Project Assumptions Log — Distillation ML Project

This document tracks all chemical engineering assumptions, simplifications, and design
decisions made during synthetic dataset generation. Each entry is classified as one of:

- **Established Principle** — supported by standard chemical engineering theory
- **Project Assumption** — a simplification introduced specifically for this project
- **Empirical Relationship** — derived from data (not yet applicable at this stage)
- **Modeling Choice** — a decision about how the ML problem is framed, not a physical claim

---

## 1. Purity Layer — Variable Set

| Variable   | Role             | Range         | Meaning                                            |
|------------|------------------|---------------|-----------------------------------------------------|
| z_F        | Sampled input    | 0.20 – 0.80   | Feed composition (light key mole fraction)          |
| x_D        | Sampled target   | 0.90 – 0.995  | Target distillate purity                            |
| x_B        | Sampled target   | 0.01 – 0.05   | Target bottoms impurity (light key in bottoms)      |
| alpha      | Sampled input    | 1.5 – 4.0     | Relative volatility (separation difficulty)         |
| R_factor   | Sampled input    | 1.1 – 2.5     | Multiplier over minimum reflux (design heuristic)   |
| R          | Computed output  | ≥ 0.5 (floor) | Actual reflux ratio (R_factor × R_min)              |
| N          | Computed output  | —             | Actual number of trays (via Gilliland correlation)  |

---

## 2. Why Fenske-Underwood-Gilliland (FUG) Over a Custom Empirical Formula

An initial simplified empirical formula (exponential decay toward a purity ceiling) was
considered but rejected. FUG was chosen instead because it is an established, citable,
textbook shortcut distillation design method — already covered in coursework — giving a
scientifically defensible relationship instead of hand-tuned constants.

**Direction-of-computation note (important):** FUG is conventionally a *design* method —
given a target purity, it computes the required trays and reflux. Our ML use case needs the
reverse (given R, N → predict achieved purity). Rather than numerically inverting the
correlation, x_D and x_B are treated as *sampled design targets*, and R and N are *computed
outputs* of the design equations. This produces physically consistent (R, N, z_F, x_D)
tuples. The ML model will later learn the operational direction (R, N, z_F → x_D) from this
data — a standard and valid approach for simulation-based synthetic datasets.
**Classification: Modeling Choice.**

---

## 3. Governing Equations

### 3.1 Vapor-Liquid Equilibrium (Established Principle)

y_q = (alpha * z_F) / (1 + (alpha - 1) * z_F)

Evaluates the binary equilibrium curve at the feed composition. Valid under the assumption
of saturated liquid feed (q = 1), which makes the feed q-line vertical at x = z_F.

### 3.2 Minimum Reflux Ratio (Established Principle — binary shortcut form)

R_min = (x_D - y_q) / (y_q - z_F)

### 3.3 Fenske Equation — Minimum Trays at Total Reflux (Established Principle)

N_min = ln[ (x_D / (1 - x_D)) * ((1 - x_B) / x_B) ] / ln(alpha)

### 3.4 Gilliland Correlation — Molokanov Algebraic Form (Established Empirical Correlation)

X = (R - R_min) / (R + 1)
Y = 1 - exp[ ((1 + 54.4X) / (11 + 117.2X)) * ((X - 1) / sqrt(X)) ]
N = (N_min + Y) / (1 - Y)

R is computed as `R_factor * R_min`, where R_factor (1.1–2.5) is a sampled design
heuristic multiplier.

---

## 4. Assumption Classification Summary

| Element                                      | Classification                     | Note |
|-----------------------------------------------|-------------------------------------|------|
| VLE relation (y vs x, relative volatility)    | Established Principle               | Standard binary VLE relation |
| Fenske equation                               | Established Principle               | Textbook shortcut method |
| Minimum reflux formula                        | Established Principle               | Binary shortcut, equivalent to Underwood for binary systems |
| Gilliland (Molokanov) correlation             | Established Empirical Correlation   | Published, citable |
| Saturated liquid feed (q = 1)                 | Project Assumption                  | Simplification; real feeds vary in thermal condition |
| Relative volatility sampled 1.5–4.0           | Project Assumption                  | Representative range, not tied to a specific real component pair |
| R ≥ 0.5 realism floor                         | Project Assumption                  | See Section 5 |
| (R, N, z_F) → x_D used for ML training        | Modeling Choice                     | Data generated in design direction; ML learns operational direction |

---

## 5. Realism Constraint: R ≥ 0.5

**Industry context (Established Principle):** industrial reflux ratios typically range
from ~1.0 to 5, occasionally higher for difficult separations. R < 1 is rare but not
physically impossible for very easy separations (high relative volatility).

**The specific threshold (Project Assumption):** 0.5 is not an industry-standard cutoff.
It was introduced to exclude mathematically valid but operationally meaningless near-zero
reflux edge cases (e.g., R = 0.006 observed before this constraint was added), without
being overly restrictive.

> Documentation phrasing: "Industrial reflux ratios typically range from 1–5. We introduced
> R ≥ 0.5 as a project-defined realism constraint to exclude mathematically valid but
> practically unrepresentative near-zero reflux cases, rather than citing this exact value
> as an industry standard."

---

## 6. Sampling Strategy

- **Method:** Latin Hypercube Sampling (LHS) across the 5-dimensional input space
  (z_F, x_D, x_B, alpha, R_factor), chosen over plain random uniform sampling to ensure
  even coverage and avoid clustering/gaps.
- **Reproducibility:** fixed random seed (42) used throughout.
- **Infeasible rows:** combinations violating x_B < z_F < x_D, numerical guards on the
  Gilliland correlation domain (0 < X < 1), or the R ≥ 0.5 floor are rejected; sampling
  continues (reject-and-resample) until the exact target row count is reached.
- **Result (2,000 target rows):** 2,000 valid rows generated, 2,182 total sampled,
  rejection rate 8.34% (post R ≥ 0.5 floor).

---

## 7. Not Yet Covered

The energy/throughput layer (feed flow rate F, feed thermal condition q → reboiler duty /
energy consumption) has not yet been designed. This will be added as a new section once
those decisions are made.

---

## 8. Energy Layer — Variable Set and Equations

### 8.1 Variable Set

| Variable | Role              | Range                     | Meaning                                  |
|----------|-------------------|---------------------------|--------------------------------------------|
| F        | Sampled input      | 100 – 500 kmol/hr          | Feed flow rate                             |
| lam      | Sampled input      | 25,000 – 35,000 kJ/kmol    | Latent heat of vaporization                |
| V        | Computed intermediate | —                       | Vapor flow in rectifying section           |
| Q_reb    | Computed output    | —                          | Reboiler duty (energy consumption)         |

### 8.2 Governing Equations

**Vapor flow (Established Principle — standard shortcut distillation relation):**

V = F * (R + 1)

Vapor flow in the rectifying section relates directly to reflux ratio and feed rate — more
reflux means more internal liquid/vapor recycling, requiring more vapor to be generated.

**Reboiler duty (Established Principle — standard simplified energy balance):**

Q_reb = V * lam

The heat required at the reboiler is the vapor flow multiplied by the latent heat needed to
vaporize it. This is a simplified energy balance — it does not account for sensible heat,
column heat losses, or feed thermal condition effects on the energy balance directly.

### 8.3 Assumption Classification

| Element                                   | Classification         | Note |
|--------------------------------------------|--------------------------|------|
| V = F(R+1) relation                        | Established Principle    | Standard shortcut distillation vapor flow relation |
| Q_reb = V × lam                            | Established Principle    | Standard simplified reboiler energy balance |
| lam sampled 25,000–35,000 kJ/kmol           | Project Assumption        | Representative range for common organic mixtures, not tied to a specific real compound |
| F sampled 100–500 kmol/hr                   | Project Assumption        | Represents a moderate industrial-scale column |
| Energy balance excludes sensible heat, feed thermal effects, column losses | Project Assumption (simplification) | A full rigorous energy balance was judged out of scope for this project (see Section 12 of project methodology — optimization framework) |

### 8.4 Known Limitation

Q_reb shows a wide range in the generated dataset (~4.8 million to ~264 million, roughly a
55x spread), driven mainly by the wide range of R from the purity layer. This is physically
expected (very high-reflux designs do consume dramatically more energy) but may warrant a
log-transform of Q_reb before ML model training, to be assessed during EDA.

### 8.5 Open Question for Future Consideration

`lam` (latent heat) is currently a sampled input used to generate Q_reb, but it is not
clearly an "operating condition" an operator would set — it is a property of the chemical
mixture being separated. Whether to include `lam` as an ML input feature (alongside R, N,
z_F, F) or treat it as a fixed/implicit property will be decided during ML feature
selection in Phase 3.

---

## 9. Chemical System Coupling (Revision to Section 8)

**Issue identified:** In the original implementation, `alpha` (relative volatility) and `lam`
(latent heat of vaporization) were sampled independently. In reality, both are properties of
the same physical mixture being separated and should be correlated, not independent.

**Revision:** Introduced 5 representative "chemical systems," each defined by a coupled
(alpha range, lam range) pair. Each row is assigned a system_id (1–5), and alpha/lam are
sampled within that system's ranges — making them realistically correlated rather than
arbitrary.

| System | alpha range   | lam range (kJ/kmol) | Character                      |
|--------|---------------|-----------------------|----------------------------------|
| 1      | 1.5 – 2.0     | 32,000 – 35,000       | Hard separation, high latent heat |
| 2      | 2.0 – 2.5     | 29,000 – 32,000       | Moderately hard                  |
| 3      | 2.5 – 3.0     | 27,000 – 30,000       | Medium                            |
| 4      | 3.0 – 3.5     | 26,000 – 28,000       | Moderately easy                   |
| 5      | 3.5 – 4.0     | 25,000 – 27,000       | Easy separation, lower latent heat|

**Classification: Project Assumption.** The general tendency (easier separations often
correlate with lower latent heat in many common organic series) is a loose real-world
pattern, not a strict physical law. The specific system boundaries and ranges are not
derived from real named chemical component data — they are illustrative, representative
values chosen to introduce realistic coupling without overclaiming specific chemical
identities.

**Validation:** Post-implementation, the correlation between alpha and lam in the generated
dataset is -0.92, confirming the intended coupling.

**Known limitation for Phase 3:** alpha and lam are now very strongly correlated (-0.92),
bordering on redundant as separate ML input features. This will be revisited during feature
selection in the ML modeling phase.

---

## 10. Dataset Size Revision

**Initial size:** 2,000 rows.
**Revised size:** 5,000 rows.

**Reason:** During Phase 3 ML modeling, Random Forest performance for predicting x_D was
found to improve meaningfully with more data (R² rose from 0.608 to 0.704 when moving from
2,000 to 5,000 rows, with hyperparameters held constant), confirming dataset size was a
genuine limiting factor. The dataset was regenerated at 5,000 rows as the new primary
dataset for all subsequent phases (EDA, modeling, optimization, dashboard).