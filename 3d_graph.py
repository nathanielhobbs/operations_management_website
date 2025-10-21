import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

st.set_page_config(layout="wide")

# Session state (dynamic constraints)

if "constraints" not in st.session_state:
    # Each constraint row: {"a1": float, "a2": float, "op": str, "b": float}
    st.session_state.constraints = [{"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0}]

OPS = ["<", "≤", "=", "≥", ">"]

def fmt(num):
    """Return an int if whole number, else a rounded float."""
    return int(num) if float(num).is_integer() else round(num, 2)

def num_input_no_step(label: str, default: float | int, key: str) -> float:
    s = st.text_input(label, value=str(default), key=key)
    try:
        return float(s.strip())
    except ValueError:
        st.error(f"Enter a valid number for {label}.")
        st.stop()

def add_constraint():
    st.session_state.constraints.append({"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0})

def remove_constraint(i: int):
    if len(st.session_state.constraints) > 1:
        st.session_state.constraints.pop(i)

# Objective Function UI
st.subheader("Objective Function")

# Objective Function UI with x₁/x₂ labels next to inputs
colA, colB, colPlus, colC, colD = st.columns([1.2, 1.4, 0.2, 1.4, 1.6])

with colA:
    sense = st.radio(
        "Objective direction",
        ["Maximize", "Minimize"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

# c1 · x1
with colB:
    _c1_l, _c1_r = st.columns([1, 0.25])
    with _c1_l:
        c1 = num_input_no_step("c₁ (coefficient of x₁)", 2.0, "c1")
    with _c1_r:
        st.markdown("<div style='margin-top:1.9rem;'>x₁</div>", unsafe_allow_html=True)

with colPlus:
    st.markdown("<div style='margin-top:1.9rem;'>+</div>", unsafe_allow_html=True)

# c2 · x2
with colC:
    _c2_l, _c2_r = st.columns([1, 0.25])
    with _c2_l:
        c2 = num_input_no_step("c₂ (coefficient of x₂)", 3.0, "c2")
    with _c2_r:
        st.markdown("<div style='margin-top:1.9rem;'>x₂</div>", unsafe_allow_html=True)

with colD:
    show_obj = st.checkbox("Show Objective Surface", value=True, key="show_obj_cb")

st.markdown(
    f"<p style='font-size:1.4rem; font-weight:600;'>Objective: Z = {c1}·x₁ + {c2}·x₂</p>",
    unsafe_allow_html=True,
)

st.divider()

# Constraints (Subject to)
st.subheader("subject to")
st.caption("Enter each constraint as a₁·x₁ + a₂·x₂ (operation) b")

for i, con in enumerate(st.session_state.constraints):
    # Layout: a1 [x1] + a2 [x2] (op) b  [buttons]
    c_a1, c_x1, c_plus, c_a2, c_x2, c_op, c_b, c_act = st.columns(
        [1.1, 0.25, 0.2, 1.1, 0.25, 0.7, 1.1, 1.4]
    )

    with c_a1:
        st.session_state.constraints[i]["a1"] = num_input_no_step(
            "a₁ (coefficient of x₁)", con["a1"], f"a1_{i}"
        )
    with c_x1:
        st.markdown("<div style='margin-top:1.9rem;'>x₁</div>", unsafe_allow_html=True)

    with c_plus:
        st.markdown("<div style='margin-top:1.9rem;'>+</div>", unsafe_allow_html=True)

    with c_a2:
        st.session_state.constraints[i]["a2"] = num_input_no_step(
            "a₂ (coefficient of x₂)", con["a2"], f"a2_{i}"
        )
    with c_x2:
        st.markdown("<div style='margin-top:1.9rem;'>x₂</div>", unsafe_allow_html=True)

    with c_op:
        st.session_state.constraints[i]["op"] = st.selectbox(
            "Operator", OPS, index=OPS.index(con["op"]), key=f"op_{i}"
        )

    with c_b:
        st.session_state.constraints[i]["b"] = num_input_no_step(
            "b (RHS)", con["b"], f"b_{i}"
        )

    with c_act:
        st.markdown("<div style='height:1.9em'></div>", unsafe_allow_html=True)
        col_add, col_remove = st.columns(2)
        with col_add:
            st.button("Add constraint", key=f"add_{i}", use_container_width=True, on_click=add_constraint)
        with col_remove:
            st.button("Remove", key=f"remove_{i}", use_container_width=True,
                      on_click=lambda idx=i: remove_constraint(idx))

# Non-negativity in constraint section
nonneg = st.checkbox("Enforce non-negativity (x₁ ≥ 0, x₂ ≥ 0)", value=True)

# Converting ops to standard LP matrices
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

# Solve LP
A_ub, b_ub, A_eq, b_eq = to_standard_matrices()
c = np.array([c1, c2])
c_obj = -c if sense == "Maximize" else c
bounds = [(0, None), (0, None)] if nonneg else [(None, None), (None, None)]

res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# Create grid for plotting
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

def constraint_segment(a1, a2, b, xmin, xmax, ymin, ymax):
    """
    Return up to two endpoints (x,y) of the line a1*x + a2*y = b clipped to the box.
    Returns None if the line doesn't intersect the box.
    """
    pts = []

    # Intersections with x boundaries → solve for y
    if a2 != 0:
        y_at_xmin = (b - a1 * xmin) / a2
        if ymin <= y_at_xmin <= ymax:
            pts.append((xmin, y_at_xmin))
        y_at_xmax = (b - a1 * xmax) / a2
        if ymin <= y_at_xmax <= ymax:
            pts.append((xmax, y_at_xmax))

    # Intersections with y boundaries → solve for x
    if a1 != 0:
        x_at_ymin = (b - a2 * ymin) / a1
        if xmin <= x_at_ymin <= xmax:
            pts.append((x_at_ymin, ymin))
        x_at_ymax = (b - a2 * ymax) / a1
        if xmin <= x_at_ymax <= xmax:
            pts.append((x_at_ymax, ymax))

    # Deduplicate and keep at most two extreme points
    if not pts:
        return None
    # unique-ish
    uniq = []
    for p in pts:
        if not any(abs(p[0]-q[0]) < 1e-9 and abs(p[1]-q[1]) < 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    # pick the pair with max distance for stability
    maxd, best = -1, None
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            dx = uniq[i][0]-uniq[j][0]; dy = uniq[i][1]-uniq[j][1]
            d2 = dx*dx + dy*dy
            if d2 > maxd:
                maxd = d2; best = (uniq[i], uniq[j])
    return best

# Build Plotly Figure
fig = go.Figure()

# == Base grids ==
xmin, xmax = x1.min(), x1.max()
ymin, ymax = x2.min(), x2.max()

# === Objective surface (base plane) ===
fig.add_trace(go.Surface(
    x=x1, y=x2, z=Z,
    colorscale="Greys",
    opacity=0.35,
    showscale=False,
    name="Objective Plane"
))

# === Feasible region overlay (colored, optional protrusion) ===
lift = 0.0  # set to e.g. 0.2 to make it 'protrude' visually above the plane
Z_feas = np.where(mask, Z + lift, np.nan)

fig.add_trace(go.Surface(
    x=x1, y=x2, z=Z_feas,
    colorscale="Blues",      # distinct from base plane
    opacity=0.85,
    showscale=False,
    name="Feasible Region"
))

# === Constraint boundary lines (on the plane) ===
# Each constraint boundary a1*x + a2*y = b rendered as a 3D line at z = c1*x + c2*y
constraint_colors = [
    "#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"
]
for idx, con in enumerate(st.session_state.constraints):
    a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]

    seg = constraint_segment(a1, a2, b, xmin, xmax, ymin, ymax)
    if seg is None:
        continue

    (xA, yA), (xB, yB) = seg
    zA = c1 * xA + c2 * yA
    zB = c1 * xB + c2 * yB

    # style inequality boundaries dashed, equality solid
    dash = "solid" if op == "=" else "dash"

    fig.add_trace(go.Scatter3d(
        x=[xA, xB],
        y=[yA, yB],
        z=[zA, zB],
        mode="lines",
        line=dict(width=5, color=constraint_colors[idx % len(constraint_colors)], dash=dash),
        name=f"Constraint {idx+1}: {a1:.2f}·x₁ + {a2:.2f}·x₂ {op} {b:.2f}",
        hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
    ))

# === Optimal solution marker (unchanged, but placed after surfaces so it sits on top) ===
if res.success:
    x_opt = res.x
    z_opt = c1 * x_opt[0] + c2 * x_opt[1]
    fig.add_trace(go.Scatter3d(
        x=[x_opt[0]], y=[x_opt[1]], z=[z_opt + lift],
        mode="markers+text",
        text=[f"({fmt(x_opt[0])}, {fmt(x_opt[1])}, {fmt(z_opt)})"],
        textposition="top right",
        marker=dict(size=6, color="red"),
        name="Optimal Solution"
    ))
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
    height=700,
    legend=dict(itemsizing="constant")
)

st.plotly_chart(fig, use_container_width=True)