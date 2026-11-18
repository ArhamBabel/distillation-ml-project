"""
Distillation Column Optimization Dashboard
AI-Based Distillation Column Optimization using Machine Learning

BITS Pilani — B.E. Chemical Engineering Semester Project
Arham Babel (2024A1PS1295P)

Launch:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve imports: add the project's src/ directory to sys.path so that
# optimizer.py (and its model files) can be imported regardless of the
# working directory from which Streamlit is launched.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

# ---------------------------------------------------------------------------
# Cache the heavy model-loading import so it only happens once.
# Importing optimizer.py triggers joblib.load() for two ~75 MB models.
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_optimizer_module():
    import optimizer as opt
    return opt

opt = _load_optimizer_module()

# Pull references for convenience
predict = opt.predict
optimize = opt.optimize
n_sweep = opt.n_sweep
SYSTEMS = opt.SYSTEMS
R_BOUNDS = opt.R_BOUNDS
N_BOUNDS = opt.N_BOUNDS
XB_BOUNDS = opt.XB_BOUNDS

# ---------------------------------------------------------------------------
# System labels for the dropdown
# ---------------------------------------------------------------------------
SYSTEM_LABELS = {
    1: "System 1 — Hard separation (α ≈ 1.75, λ ≈ 33 500 kJ/kmol)",
    2: "System 2 — Moderately hard (α ≈ 2.25, λ ≈ 30 500 kJ/kmol)",
    3: "System 3 — Medium (α ≈ 2.75, λ ≈ 28 500 kJ/kmol)",
    4: "System 4 — Moderately easy (α ≈ 3.25, λ ≈ 27 000 kJ/kmol)",
    5: "System 5 — Easy separation (α ≈ 3.75, λ ≈ 26 000 kJ/kmol)",
}

# Training-data ranges (from the actual dataset, for user guidance)
TRAINING_RANGES = {
    "z_F": (0.20, 0.80),
    "R": (0.50, 17.90),
    "N": (6.0, 44.0),
    "x_B": (0.01, 0.05),
    "F": (100.0, 500.0),
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Distillation Column Optimization",
    page_icon="🧪",
    layout="wide",
)

# Custom styling to ensure metric values with units fit without truncation
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        white-space: normal !important;
        word-break: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧪 AI-Based Distillation Column Optimization")
st.markdown(
    "Machine-learning surrogate models trained on synthetic data generated "
    "using the **Fenske–Underwood–Gilliland (FUG)** shortcut design method. "
    "The optimizer finds operating conditions that **minimize reboiler energy** "
    "while satisfying a **minimum purity specification**."
)

st.info(
    "**⚠️ Synthetic Data Disclaimer:** All predictions are based on a synthetic "
    "dataset (5 000 rows, seed 42) generated from textbook FUG equations — "
    "not from real plant data. Per-system α and λ are fixed at the midpoint "
    "of each system's sampling range (project assumption). "
    "Do not use these results for real engineering decisions.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Helper: format Q_reb for display
# ---------------------------------------------------------------------------
def _fmt_qreb(val):
    """Format Q_reb with thousands separators and unit."""
    return f"{val:,.0f} kJ/h"


def _check_training_range(label, value, lo, hi):
    """Return a warning string if value is outside the training range."""
    if value < lo or value > hi:
        return f"⚠️ {label} = {value:.4f} is outside the training range [{lo}, {hi}]. Predictions may be unreliable."
    return None


# ===================================================================
# TABS
# ===================================================================
tab_predict, tab_optimize = st.tabs(["📊 Predict", "⚙️ Optimize"])


# ===================================================================
# TAB 1: PREDICT
# ===================================================================
with tab_predict:
    st.header("Single-Point Prediction")
    st.markdown(
        "Enter operating conditions below. The trained Random Forest models "
        "predict the distillate purity (**x_D**) and reboiler duty (**Q_reb**)."
    )

    # --- Inputs in sidebar-style columns ---
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Operating Conditions")

        p_sys_label = st.selectbox(
            "Chemical system",
            options=list(SYSTEM_LABELS.values()),
            index=0,
            key="pred_sys",
        )
        p_system_id = [k for k, v in SYSTEM_LABELS.items() if v == p_sys_label][0]

        p_z_F = st.slider(
            "Feed composition (z_F)",
            min_value=0.20, max_value=0.80, value=0.50, step=0.01,
            key="pred_zf",
        )
        p_F = st.slider(
            "Feed flow rate, F (kmol/h)",
            min_value=100.0, max_value=500.0, value=300.0, step=10.0,
            key="pred_F",
        )
        p_R = st.slider(
            "Reflux ratio (R)",
            min_value=0.5, max_value=10.0, value=3.0, step=0.1,
            key="pred_R",
        )
        p_N = st.slider(
            "Number of trays (N)",
            min_value=6.0, max_value=25.0, value=16.0, step=1.0,
            key="pred_N",
        )
        p_x_B = st.slider(
            "Bottoms composition (x_B)",
            min_value=0.01, max_value=0.05, value=0.03, step=0.005,
            key="pred_xB",
        )

        predict_btn = st.button("🔍 Predict", type="primary", key="btn_predict")

    with col_out:
        st.subheader("Prediction Results")

        if predict_btn:
            # Check training-range warnings
            warnings = []
            for label, val, lo, hi in [
                ("R", p_R, *TRAINING_RANGES["R"]),
                ("N", p_N, *TRAINING_RANGES["N"]),
                ("z_F", p_z_F, *TRAINING_RANGES["z_F"]),
                ("x_B", p_x_B, *TRAINING_RANGES["x_B"]),
                ("F", p_F, *TRAINING_RANGES["F"]),
            ]:
                w = _check_training_range(label, val, lo, hi)
                if w:
                    warnings.append(w)

            if warnings:
                for w in warnings:
                    st.warning(w)

            # Run prediction
            result = predict(
                system_id=p_system_id,
                z_F=p_z_F,
                F=p_F,
                R=p_R,
                N=p_N,
                x_B=p_x_B,
            )

            m1, m2 = st.columns([1, 1.2])
            m1.metric("Predicted Distillate Purity (x_D)", f"{result['x_D_pred']:.4f}")
            m2.metric(
                "Predicted Reboiler Duty (Q_reb)",
                _fmt_qreb(result["Q_reb_pred"]),
                help=f"{result['Q_reb_pred'] / 1e6:.2f} × 10⁶ kJ/h",
            )

            st.markdown("---")
            st.markdown("**Selected conditions:**")
            cond_data = {
                "Parameter": ["System", "z_F", "F (kmol/h)", "R", "N", "x_B",
                              "α (midpoint)", "λ (midpoint, kJ/kmol)"],
                "Value": [
                    str(p_system_id),
                    f"{p_z_F:.2f}",
                    f"{p_F:.0f}",
                    f"{p_R:.2f}",
                    f"{p_N:.0f}",
                    f"{p_x_B:.4f}",
                    f"{SYSTEMS[p_system_id]['alpha']:.2f}",
                    f"{SYSTEMS[p_system_id]['lam']:,.0f}",
                ],
            }
            st.table(cond_data)

        else:
            st.markdown("*Click **Predict** to see results.*")

    # Model limitations note
    with st.expander("ℹ️ Model Limitations"):
        st.markdown(
            "- **x_D model** (distillate purity): Random Forest, R² = **0.7035** on the test set. "
            "This moderate accuracy reflects the complexity of predicting purity from operating "
            "conditions — the underlying FUG relationship is multi-variable and nonlinear.\n"
            "- **log(Q_reb) model** (reboiler duty): Random Forest, R² = **0.9686** on the test set. "
            "Predictions are made in log-space and exponentiated.\n"
            "- Both models were trained on 5 000 synthetic rows (80/20 split, seed 42) with "
            "`n_estimators=500, min_samples_leaf=2`.\n"
            "- Per-system α and λ use **midpoint** values from each system's sampling range — "
            "a project assumption, not derived from specific real chemical components."
        )


# ===================================================================
# TAB 2: OPTIMIZE
# ===================================================================
with tab_optimize:
    st.header("Energy Optimization")
    st.markdown(
        "Find the reflux ratio (**R**) and bottoms composition (**x_B**) that "
        "**minimize predicted reboiler duty** (Q_reb) while meeting the required "
        "distillate purity specification (x_D ≥ target). "
        "The number of trays (**N**) is treated as a fixed design parameter."
    )

    # --- Two sub-sections: Single-N optimize and N-sweep ---
    opt_section, sweep_section = st.tabs(["🎯 Single N Optimization", "📈 Tray Trade-off Analysis"])

    # ---------------------------------------------------------------
    # SECTION A: Single-N Optimization
    # ---------------------------------------------------------------
    with opt_section:
        col_oin, col_oout = st.columns([1, 1], gap="large")

        with col_oin:
            st.subheader("Fixed Conditions")

            o_sys_label = st.selectbox(
                "Chemical system",
                options=list(SYSTEM_LABELS.values()),
                index=0,
                key="opt_sys",
            )
            o_system_id = [k for k, v in SYSTEM_LABELS.items() if v == o_sys_label][0]

            o_z_F = st.slider(
                "Feed composition (z_F)",
                min_value=0.20, max_value=0.80, value=0.50, step=0.01,
                key="opt_zf",
            )
            o_F = st.slider(
                "Feed flow rate, F (kmol/h)",
                min_value=100.0, max_value=500.0, value=300.0, step=10.0,
                key="opt_F",
            )
            o_N = st.slider(
                "Number of trays (N)",
                min_value=6.0, max_value=25.0, value=16.0, step=1.0,
                key="opt_N",
            )
            o_purity = st.slider(
                "Minimum purity specification (x_D ≥)",
                min_value=0.90, max_value=0.995, value=0.95, step=0.005,
                key="opt_purity",
            )

            optimize_btn = st.button("⚡ Optimize", type="primary", key="btn_optimize")

        with col_oout:
            st.subheader("Optimization Results")

            if optimize_btn:
                res = optimize(
                    system_id=o_system_id,
                    z_F=o_z_F,
                    F=o_F,
                    N=o_N,
                    purity_spec=o_purity,
                )

                if res["feasible"]:
                    st.success(f"✅ Feasible solution found at N = {o_N:.0f}")

                    m1, m2 = st.columns(2)
                    m1.metric("Optimal Reflux Ratio (R)", f"{float(res['R']):.4f}")
                    m2.metric("Optimal Bottoms Comp. (x_B)", f"{float(res['x_B']):.4f}")

                    m3, m4 = st.columns([1, 1.2])
                    m3.metric("Predicted Distillate Purity (x_D)", f"{float(res['x_D_pred']):.4f}")
                    m4.metric(
                        "Predicted Reboiler Duty (Q_reb)",
                        _fmt_qreb(float(res["Q_reb_pred"])),
                        help=f"{float(res['Q_reb_pred']) / 1e6:.2f} × 10⁶ kJ/h",
                    )

                    st.markdown("---")
                    st.caption(
                        f"Search grid: R ∈ [{R_BOUNDS[0]}, {R_BOUNDS[1]}] (50 pts) × "
                        f"x_B ∈ [{XB_BOUNDS[0]}, {XB_BOUNDS[1]}] (50 pts) = 2 500 candidates."
                    )
                else:
                    st.error(
                        f"❌ **Infeasible:** {res['message']}\n\n"
                        f"No combination of R and x_B within the search bounds achieves "
                        f"x_D ≥ {o_purity:.3f} at N = {o_N:.0f} trays. "
                        f"Try increasing N or relaxing the purity target."
                    )
            else:
                st.markdown("*Click **Optimize** to find the best operating point.*")

    # ---------------------------------------------------------------
    # SECTION B: Tray Trade-off (N-sweep)
    # ---------------------------------------------------------------
    with sweep_section:
        st.subheader("Energy vs. Number of Trays")
        st.markdown(
            "Sweep across a range of tray counts to see how the minimum achievable "
            "reboiler duty changes. More trays generally allow a lower reflux ratio "
            "for the same purity — reducing energy — but with diminishing returns. "
            "This is the classic **capital vs. operating cost trade-off** in distillation."
        )

        col_sin, col_sout = st.columns([1, 2], gap="large")

        with col_sin:
            s_sys_label = st.selectbox(
                "Chemical system",
                options=list(SYSTEM_LABELS.values()),
                index=0,
                key="sweep_sys",
            )
            s_system_id = [k for k, v in SYSTEM_LABELS.items() if v == s_sys_label][0]

            s_z_F = st.slider(
                "Feed composition (z_F)",
                min_value=0.20, max_value=0.80, value=0.50, step=0.01,
                key="sweep_zf",
            )
            s_F = st.slider(
                "Feed flow rate, F (kmol/h)",
                min_value=100.0, max_value=500.0, value=300.0, step=10.0,
                key="sweep_F",
            )
            s_purity = st.slider(
                "Minimum purity specification (x_D ≥)",
                min_value=0.90, max_value=0.995, value=0.95, step=0.005,
                key="sweep_purity",
            )

            s_n_min = st.number_input(
                "Minimum N", min_value=6, max_value=24, value=8, step=1, key="sweep_nmin",
            )
            s_n_max = st.number_input(
                "Maximum N", min_value=7, max_value=25, value=25, step=1, key="sweep_nmax",
            )
            s_n_step = st.number_input(
                "Step", min_value=1, max_value=5, value=1, step=1, key="sweep_nstep",
            )

            sweep_btn = st.button("📈 Run Sweep", type="primary", key="btn_sweep")

        with col_sout:
            if sweep_btn:
                if s_n_min >= s_n_max:
                    st.error("Minimum N must be less than Maximum N.")
                else:
                    n_list = list(range(int(s_n_min), int(s_n_max) + 1, int(s_n_step)))

                    with st.spinner("Running N-sweep..."):
                        sweep_results = n_sweep(
                            system_id=s_system_id,
                            z_F=s_z_F,
                            F=s_F,
                            purity_spec=s_purity,
                            N_values=n_list,
                        )

                    # Build a results table
                    import pandas as pd
                    table_rows = []
                    for r in sweep_results:
                        if r["feasible"]:
                            table_rows.append({
                                "N (trays)": int(r["N"]),
                                "Feasible": "✅ Yes",
                                "Optimal R": f"{float(r['R']):.4f}",
                                "Optimal x_B": f"{float(r['x_B']):.4f}",
                                "Predicted x_D": f"{float(r['x_D_pred']):.4f}",
                                "Q_reb (kJ/h)": f"{float(r['Q_reb_pred']):,.0f}",
                                "Q_reb (×10⁶)": f"{float(r['Q_reb_pred']) / 1e6:.2f}",
                            })
                        else:
                            table_rows.append({
                                "N (trays)": int(r["N"]),
                                "Feasible": "❌ No",
                                "Optimal R": "—",
                                "Optimal x_B": "—",
                                "Predicted x_D": "—",
                                "Q_reb (kJ/h)": "—",
                                "Q_reb (×10⁶)": "—",
                            })

                    df_table = pd.DataFrame(table_rows)
                    st.dataframe(df_table, use_container_width=True, hide_index=True)

                    # Chart: only feasible points
                    feasible = [r for r in sweep_results if r["feasible"]]
                    if len(feasible) >= 1:
                        chart_df = pd.DataFrame({
                            "N (trays)": [int(r["N"]) for r in feasible],
                            "Q_reb (kJ/h)": [float(r["Q_reb_pred"]) for r in feasible],
                        })

                        st.markdown("#### Energy–Trays Trade-off")
                        st.line_chart(
                            chart_df.set_index("N (trays)"),
                            y="Q_reb (kJ/h)",
                            use_container_width=True,
                        )

                        n_infeasible = len(sweep_results) - len(feasible)
                        if n_infeasible > 0:
                            st.caption(
                                f"{n_infeasible} tray count(s) were infeasible "
                                f"(no R/x_B combination met x_D ≥ {s_purity:.3f}) "
                                f"and are excluded from the chart."
                            )
                    else:
                        st.warning(
                            "No feasible solutions found across the entire N range. "
                            "Try a lower purity specification or a different chemical system."
                        )
            else:
                st.markdown("*Configure the sweep range and click **Run Sweep**.*")

    # Optimization methodology note
    with st.expander("ℹ️ Optimization Methodology"):
        st.markdown(
            "**Objective:** Minimize predicted reboiler duty Q_reb (kJ/h).\n\n"
            "**Constraint:** Predicted distillate purity x_D ≥ purity specification.\n\n"
            "**Decision variables:** Reflux ratio R and bottoms composition x_B, "
            "searched over a 50 × 50 grid within the bounds "
            f"R ∈ [{R_BOUNDS[0]}, {R_BOUNDS[1]}], "
            f"x_B ∈ [{XB_BOUNDS[0]}, {XB_BOUNDS[1]}].\n\n"
            "**Fixed inputs:** Chemical system (sets α, λ at midpoint), feed composition z_F, "
            "feed flow rate F, and number of trays N.\n\n"
            "**Method:** Grid search was chosen over gradient-based methods because "
            "the Random Forest prediction surface is piecewise constant (zero gradient "
            "almost everywhere). The grid is bounded within the training data range "
            "to avoid extrapolation.\n\n"
            "**Why N is fixed per call:** More trays always reduce energy for a given purity "
            "(the capital-vs-operating-cost trade-off is not modeled). The tray trade-off "
            "analysis sweeps N externally to visualize this relationship."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AI-Based Distillation Column Optimization using Machine Learning · "
    "BITS Pilani · B.E. Chemical Engineering · Arham Babel (2024A1PS1295P) · "
    "Models: Random Forest surrogates trained on FUG-generated synthetic data."
)
