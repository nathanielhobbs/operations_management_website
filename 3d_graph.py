# 3d_graph.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

st.set_page_config(layout="wide")

# -----------------------------
# Session state (dynamic constraints)
# -----------------------------
if "constraints" not in st.session_state:
    # Each constraint row: {"a1": float, "a2": float, "op": str, "b": float}
    st.session_state.constraints = [{"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0}]

OPS = ["<", "≤", "=", "≥", ">"]

def add_constraint():
    st.session_state.constraints.append({"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0})

def remove_constraint(i: int):
    if len(st.session_state.constraints) > 1:
        st.session_state.constraints.pop(i)

# -----------------------------
# Objective Function UI
# -----------------------------
st.subheader("Objective Function")

# widen the last column so the label fits
colA, colB, colC, colD = st.columns([1.2, 1, 1, 1.6])

with colA:
    sense = st.radio(
        "Objective direction",
        ["Maximize", "Minimize"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

with colB:
    c1 = st.number_input("c₁ (coefficient of x₁)", value=2.0, step=0.5, format="%.2f")

with colC:
    c2 = st.number_input("c₂ (coefficient of x₂)", value=3.0, step=0.5, format="%.2f")

with colD:
    show_obj = st.checkbox(
        "Show Objective Surface",
        value=True,
        key="show_obj_cb"
    )

st.markdown(f"**Objective:** Z = {c1}·x₁ + {c2}·x₂")

st.divider()


# -----------------------------
# Constraints (Subject to)
# -----------------------------
st.subheader("subject to")
st.caption("Enter each constraint as a₁·x₁ + a₂·x₂ (operation) b")

for i, con in enumerate(st.session_state.constraints):
    c_a1, c_a2, c_op, c_b, c_act = st.columns([1, 1, 0.9, 1, 1.2])

    with c_a1:
        st.session_state.constraints[i]["a1"] = st.number_input(
            "a₁ (coefficient of x₁)", value=float(con["a1"]),
            step=0.5, format="%.2f", key=f"a1_{i}"
        )
    with c_a2:
        st.session_state.constraints[i]["a2"] = st.number_input(
            "a₂ (coefficient of x₂)", value=float(con["a2"]),
            step=0.5, format="%.2f", key=f"a2_{i}"
        )
    with c_op:
        st.session_state.constraints[i]["op"] = st.selectbox(
            "Operator", OPS, index=OPS.index(con["op"]), key=f"op_{i}"
        )
    with c_b:
        st.session_state.constraints[i]["b"] = st.number_input(
            "b (RHS)", value=float(con["b"]),
            step=0.5, format="%.2f", key=f"b_{i}"
        )
    with c_act:
        # fake label so button aligns with inputs that have labels
        st.markdown("<div style='height:1.9em'></div>", unsafe_allow_html=True)
        col_add, col_remove = st.columns(2)
        with col_add:
            st.button("Add constraint", key=f"add_{i}", use_container_width=True, on_click=add_constraint)
        with col_remove:
            st.button("Remove", key=f"remove_{i}", use_container_width=True,
                    on_click=lambda idx=i: remove_constraint(idx))

# Non-negativity in constraint section
nonneg = st.checkbox("Enforce non-negativity (x₁ ≥ 0, x₂ ≥ 0)", value=True)

# -----------------------------
# Helper: convert ops to standard LP matrices
# -----------------------------
def to_standard_matrices():
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in st.session_state.constraints:
        a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]
        if op in ("<", "≤"):
            A_ub.append([a1, a2]); b_ub.append(b)
        elif op in (">", "≥"):
            # multiply by -1 to convert ≥ to ≤
            A_ub.append([-a1, -a2]); b_ub.append(-b)
        else:  # "="
            A_eq.append([a1, a2]); b_eq.append(b)
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    return A_ub, b_ub, A_eq, b_eq

# -----------------------------
# Solve LP
# -----------------------------
A_ub, b_ub, A_eq, b_eq = to_standard_matrices()
c = np.array([c1, c2])
c_obj = -c if sense == "Maximize" else c
bounds = [(0, None), (0, None)] if nonneg else [(None, None), (None, None)]

res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# -----------------------------
# Create grid for plotting
# -----------------------------
x1 = np.linspace(0, 10, 200)
x2 = np.linspace(0, 10, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = c1 * X1 + c2 * X2

# Feasible mask using the actual ops
mask = np.ones_like(X1, dtype=bool)
for con in st.session_state.constraints:
    a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]
    expr = a1 * X1 + a2 * X2
    if op in ("<", "≤"):
        cond = expr <= b
    elif op in (">", "≥"):
        cond = expr >= b
    else:  # "="
        cond = np.isclose(expr, b, atol=1e-6)
    mask &= cond

# -----------------------------
# Build Plotly Figure (3D)
# -----------------------------
fig = go.Figure()

# Feasible region as a 2D contour "shadow" at z=0 (for context)
fig.add_trace(go.Contour(
    x=x1, y=x2, z=mask.astype(int),
    showscale=False, opacity=0.35,
    colorscale=[[0, "white"], [1, "green"]],
    contours=dict(showlines=False),
    name="Feasible Region (shadow)"
))

# Objective surface
if st.session_state.get("show_obj_cb", True):
    fig.add_trace(go.Surface(
        x=x1, y=x2, z=Z,
        opacity=0.5,
        colorscale="Viridis",
        name="Objective Surface",
        showscale=True  # the colorbar you saw comes from this trace
    ))

# Optimal solution
if res.success:
    x_opt = res.x
    z_opt = c1 * x_opt[0] + c2 * x_opt[1]
    fig.add_trace(go.Scatter3d(
        x=[x_opt[0]], y=[x_opt[1]], z=[z_opt],
        mode="markers+text",
        text=[f"({x_opt[0]:.2f}, {x_opt[1]:.2f}, {z_opt:.2f})"],
        textposition="top right",
        marker=dict(size=6, color="red"),
        name="Optimal Solution"
    ))
    # Show objective according to sense
    Z_display = z_opt if sense == "Minimize" else z_opt
    st.success(f"Optimal solution: x₁ = {x_opt[0]:.2f}, x₂ = {x_opt[1]:.2f},  Z* = {Z_display:.2f}")
else:
    st.error(f"Solve failed: {res.message}")

# Layout
fig.update_layout(
    scene=dict(
        xaxis_title="x₁",
        yaxis_title="x₂",
        zaxis_title="Objective value (Z)"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=700
)

st.plotly_chart(fig, use_container_width=True)
