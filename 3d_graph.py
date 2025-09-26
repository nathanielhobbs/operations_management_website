import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

st.set_page_config(layout="wide")

# Session state (dynamic constraints)
if "constraints" not in st.session_state:
    # Each constraint row: {"a1": float, "a2": float, "op": str, "b": float}
    st.session_state.constraints = [{"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0}]

OPS = ["<", "≤", "=", "≥", ">"]  # dropdown operators

def add_constraint():
    st.session_state.constraints.append({"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0})

def remove_constraint(i: int):
    if len(st.session_state.constraints) > 1:
        st.session_state.constraints.pop(i)

# Objective Function
st.subheader("Objective Function")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    sense = st.radio("", ["Maximize", "Minimize"], horizontal=True, index=0)

# Mini headings directly above each field
with colB:
    c1 = st.number_input("c₁ (coefficient of x₁)", value=2.0, step=0.5, format="%.2f")
with colC:
    c2 = st.number_input("c₂ (coefficient of x₂)", value=3.0, step=0.5, format="%.2f")

# Objective function preview
st.markdown(f"**Objective:** Z = {c1}·x₁ + {c2}·x₂")

st.divider()

# Constraints (Subject to)
st.subheader("subject to")
st.caption("Enter each constraint as a₁·x₁ + a₂·x₂ (operation) b")

for i, con in enumerate(st.session_state.constraints):
    c_a1, c_a2, c_op, c_b, c_act = st.columns([1, 1, 0.9, 1, 1.2])

    with c_a1:
        st.session_state.constraints[i]["a1"] = st.number_input(
            f"a₁ (coefficient of x₁)", value=float(con["a1"]),
            step=0.5, format="%.2f", key=f"a1_{i}"
        )
    with c_a2:
        st.session_state.constraints[i]["a2"] = st.number_input(
            f"a₂ (coefficient of x₂)", value=float(con["a2"]),
            step=0.5, format="%.2f", key=f"a2_{i}"
        )
    with c_op:
        st.session_state.constraints[i]["op"] = st.selectbox(
            f"Operator", ["<","≤","=","≥",">"],
            index=["<","≤","=","≥",">"].index(con["op"]), key=f"op_{i}"
        )
    with c_b:
        st.session_state.constraints[i]["b"] = st.number_input(
            f"b (RHS)", value=float(con["b"]),
            step=0.5, format="%.2f", key=f"b_{i}"
        )
    with c_act:
        # fake label so button aligns with inputs that have labels
        st.markdown("<div style='height:1.9em'></div>", unsafe_allow_html=True)
        if i == 0:
            st.button("Add constraint", key=f"add_{i}", use_container_width=True, on_click=add_constraint)
        else:
            st.button("Remove", key=f"remove_{i}", use_container_width=True,
                      on_click=lambda idx=i: remove_constraint(idx))

# Non-negativity in constraint section
nonneg = st.checkbox("Enforce non-negativity (x₁ ≥ 0, x₂ ≥ 0)", value=True)

# Full model preview
with st.expander("Model Preview"):
    opt_word = "maximize" if sense == "Maximize" else "minimize"
    st.write(f"{opt_word}  Z = {c1}·x₁ + {c2}·x₂")
    st.write("subject to:")
    for con in st.session_state.constraints:
        a1 = con["a1"]; a2 = con["a2"]; op = con["op"]; b = con["b"]
        # Show +/− properly for the a2 term in plain text
        term2 = f"+ {abs(a2)}·x₂" if a2 >= 0 else f"- {abs(a2)}·x₂"
        st.write(f"  {a1}·x₁ {term2} {op} {b}")
    if nonneg:
        st.write("  x₁ ≥ 0, x₂ ≥ 0")

st.divider()

# Action buttons
spacer_left, col1, col2, spacer_right = st.columns([1, 2, 2, 1])

with col1:
    st.button("Compute Feasible Solutions", type="primary", use_container_width=True)

with col2:
    st.button("Solve for Optimal Solution", use_container_width=True)

# -----------------------------
# Solve LP
# -----------------------------
A = np.array([[a1, a2] for (a1, a2, _) in constraints])
b = np.array([rhs for (_, _, rhs) in constraints])

c = np.array([c1, c2])
if maximize:
    res = linprog(-c, A_ub=A, b_ub=b, bounds=(0, None))
else:
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))

# -----------------------------
# Create grid
# -----------------------------
x1 = np.linspace(0, 10, 100)
x2 = np.linspace(0, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = c1 * X1 + c2 * X2

mask = np.ones_like(X1, dtype=bool)
for (a1, a2, rhs) in constraints:
    mask &= (a1 * X1 + a2 * X2 <= rhs)

# -----------------------------
# Build Plotly Figure
# -----------------------------
fig = go.Figure()

# Feasible region as 2D contour "shadow"
fig.add_trace(go.Contour(
    x=x1, y=x2, z=mask.astype(int),
    showscale=False, opacity=0.4,
    colorscale=[[0, "white"], [1, "green"]],
    contours=dict(showlines=False),
    name="Feasible Region"
))

# Objective plane
if show_obj:
    fig.add_trace(go.Surface(
        x=x1, y=x2, z=Z,
        opacity=0.5, colorscale="Viridis",
        name="Objective Plane"
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
    st.success(f"Optimal solution: x1 = {x_opt[0]:.2f}, x2 = {x_opt[1]:.2f}, objective = {z_opt:.2f}")
else:
    st.error("No feasible solution found.")

# Layout
fig.update_layout(
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="Objective value (z)"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

