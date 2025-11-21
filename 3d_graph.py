import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import html
import json 

st.set_page_config(layout="wide")

# Session state (dynamic constraints)
if "constraints" not in st.session_state:
    st.session_state.constraints = [{"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0, "enabled": True}]
else:
    # Migration: add enabled flag if missing
    for con in st.session_state.constraints:
        if "enabled" not in con:
            con["enabled"] = True

# Z level storage
if "z_levels" not in st.session_state:
    st.session_state.z_levels = []

OPS = ["<", "≤", "=", "≥", ">"]

def fmt(num):
    """Return an int if whole number, else a rounded float."""
    return int(num) if float(num).is_integer() else round(num, 2)

def num_input_no_step(label: str, default: float | int, key: str, disabled: bool = False) -> float:
    # If the widget already has a value in session_state, let Streamlit use that.
    # Otherwise, provide the initial default.
    if key in st.session_state:
        s = st.text_input(label, key=key, disabled=disabled)
    else:
        s = st.text_input(label, value=str(default), key=key, disabled=disabled)

    try:
        return float(str(s).strip())
    except ValueError:
        # If disabled, keep default without erroring
        if disabled:
            return float(default)
        st.error(f"Enter a valid number for {label}.")
        st.stop()

def add_constraint():
    st.session_state.constraints.append(
        {"a1": 1.0, "a2": 1.0, "op": "≤", "b": 10.0, "enabled": True}
    )

def remove_constraint(i: int):
    if len(st.session_state.constraints) > 1:
        st.session_state.constraints.pop(i)


# ---- Preset 1 (edit values here if you want different numbers) ----
PRESET_1 = {
    "c1": 2.0,
    "c2": 3.0,
    # List of (a1, a2, b) for each constraint
    "constraints": [
        (1.0, 2.0, 20.0),
        (3.0, 1.0, 18.0),
        (1.0, 1.0, 12.0),
    ],
}

def apply_preset1():
    # Objective coefficients
    st.session_state.c1 = PRESET_1["c1"]
    st.session_state.c2 = PRESET_1["c2"]
    st.session_state.c1_input = PRESET_1["c1"]
    st.session_state.c2_input = PRESET_1["c2"]

    # Number of constraints
    n = len(PRESET_1["constraints"])
    st.session_state.constraints = n
    st.session_state.constraints_input = n

    # Sidebar constraint fields (a1_i, a2_i, b_i)
    for i, (a1, a2, b) in enumerate(PRESET_1["constraints"]):
        st.session_state[f"a1_{i}"] = float(a1)
        st.session_state[f"a2_{i}"] = float(a2)
        st.session_state[f"b_{i}"]  = float(b)

def apply_preset_from_data(data: dict):
    """Apply preset values from a dict loaded from a file."""
    # Objective coefficients
    st.session_state.c1 = float(data["c1"])
    st.session_state.c2 = float(data["c2"])
    st.session_state.c1_input = float(data["c1"])
    st.session_state.c2_input = float(data["c2"])

    # Number of constraints
    constraints_list = data["constraints"]
    n = len(constraints_list)
    st.session_state.constraints = n
    st.session_state.constraints_input = n

    # Sidebar constraint fields (a1_i, a2_i, b_i)
    for i, (a1, a2, b) in enumerate(constraints_list):
        st.session_state[f"a1_{i}"] = float(a1)
        st.session_state[f"a2_{i}"] = float(a2)
        st.session_state[f"b_{i}"]  = float(b)


# state flag for showing the uploader
if "show_preset_upload" not in st.session_state:
    st.session_state.show_preset_upload = False

def handle_preset_from_file_button():
    """
    First click: show uploader.
    Once a file is uploaded: second click applies preset and hides uploader.
    """
    if not st.session_state.show_preset_upload:
        # first click -> just show the uploader
        st.session_state.show_preset_upload = True
    else:
        uploaded = st.session_state.get("preset_file")
        if uploaded is not None:
            data = json.load(uploaded)
            apply_preset_from_data(data)
            # hide uploader again after applying
            st.session_state.show_preset_upload = False


# -----------------------------
# Hide/Show Toggle ("Disappear" button)
# -----------------------------
# if "hide_all" not in st.session_state:
#     st.session_state.hide_all = False

# def toggle_visibility():
#     st.session_state.hide_all = not st.session_state.hide_all

# # Always show this button
# st.button("Disappear / Reappear", on_click=toggle_visibility)


# Z helpers 
def _parse_single_z(text: str) -> float:
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace("Z=", "").replace("z=", "").replace("Z", "").replace("z", "")
    s = s.strip()
    return float(s)

def remove_z(i: int):
    if 0 <= i < len(st.session_state.z_levels):
        st.session_state.z_levels.pop(i)

def _add_z_from_state():
    try:
        z_val = _parse_single_z(st.session_state.get("z_input", ""))
        if not np.isfinite(z_val):
            st.warning("Please enter a finite number for Z."); return
        if not any(abs(z_val - z) < 1e-12 for z in st.session_state.z_levels):
            st.session_state.z_levels.append(z_val)
            st.success(f"Added Z = {fmt(z_val)}")
        else:
            st.info("That Z value is already in the list.")
    except Exception:
        st.warning("Enter a numeric Z (e.g., 5000).")
    finally:
        st.session_state["z_input"] = ""

def remove_z(i: int):
    if 0 <= i < len(st.session_state.z_levels):
        st.session_state.z_levels.pop(i)

# Objective Function UI
st.subheader("Objective Function")

# Objective Function UI with x₁/x₂ labels next to inputs
colA, colB, colPlus, colC, colD = st.columns([1.2, 1.4, 0.2, 1.4, 1.6])

with colA:
    if "sense" in st.session_state:
        sense = st.radio(
            "Objective direction",
            ["Maximize", "Minimize"],
            horizontal=True,
            key="sense",
            label_visibility="collapsed",
        )
    else:
        sense = st.radio(
            "Objective direction",
            ["Maximize", "Minimize"],
            horizontal=True,
            index=0,
            key="sense",
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
    show_obj = st.checkbox("Show in 3D", value=True, key="show_obj_cb")

st.markdown(
    f"<p style='font-size:1.4rem; font-weight:600;'>Objective: Z = {c1}·x₁ + {c2}·x₂</p>",
    unsafe_allow_html=True,
)

# --- Z level input (optional) ---
def _parse_z_levels(s: str) -> list[float]:
    vals = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            pass
    # de-dupe, keep order
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

# ==== Objective level lines (Z) ====
st.subheader("Objective level lines")

# Header row: "Z =" input on left, "Add Z value" button on right
left, mid, right = st.columns([0.15, 0.65, 0.20])

with left:
    st.markdown("<div style='margin-top:0.6rem;'>Z =</div>", unsafe_allow_html=True)

with mid:
    if "z_input" in st.session_state:
        st.text_input(
            "Z value",
            key="z_input",
            label_visibility="collapsed",
        )
    else:
        st.text_input(
            "Z value",
            value="",
            key="z_input",
            label_visibility="collapsed",
        )

with right:
    st.button("Add Z value", key="add_z_btn", use_container_width=True, on_click=_add_z_from_state)

# Existing Z values list
if st.session_state.z_levels:
    st.markdown("**Current Z values**")
    for idx, z in enumerate(st.session_state.z_levels):
        col_txt, col_rm = st.columns([0.85, 0.15])
        with col_txt:
            st.markdown(f"- **Z = {fmt(z)}**")
        with col_rm:
            st.button("Remove", key=f"delz_{idx}", use_container_width=True,
                      on_click=lambda i=idx: remove_z(i))

# Expose list for plotting branches
z_levels = st.session_state.z_levels

st.divider()

# Constraints (Subject to)
st.subheader("subject to")
st.caption("Enter each constraint as a₁·x₁ + a₂·x₂ (operation) b")
# Top-level Add button (right aligned)
_sp, top_right = st.columns([3, 1])
with top_right:
    st.button("Add constraint", use_container_width=True, on_click=add_constraint)

for i, con in enumerate(st.session_state.constraints):
    # Row enabled toggle
    enabled_key = f"enabled_{i}"
    
    row_enabled = st.session_state.constraints[i]["enabled"]

    # Layout: a1 [x1] + a2 [x2] (op) b  [buttons]
    c_a1, c_x1, c_plus, c_a2, c_x2, c_op, c_b, c_act = st.columns(
        [1.1, 0.25, 0.2, 1.1, 0.25, 0.7, 1.1, 1.4]
    )

    with c_a1:
        st.session_state.constraints[i]["a1"] = num_input_no_step(
            "a₁ (coefficient of x₁)", con["a1"], f"a1_{i}", disabled=not row_enabled
        )
    with c_x1:
        st.markdown("<div style='margin-top:1.9rem;'>x₁</div>", unsafe_allow_html=True)

    with c_plus:
        st.markdown("<div style='margin-top:1.9rem;'>+</div>", unsafe_allow_html=True)

    with c_a2:
        st.session_state.constraints[i]["a2"] = num_input_no_step(
            "a₂ (coefficient of x₂)", con["a2"], f"a2_{i}", disabled=not row_enabled
        )
    with c_x2:
        st.markdown("<div style='margin-top:1.9rem;'>x₂</div>", unsafe_allow_html=True)

    with c_op:
        st.session_state.constraints[i]["op"] = st.selectbox(
            "Operator", OPS, index=OPS.index(con["op"]), key=f"op_{i}", disabled=not row_enabled
        )

    with c_b:
        st.session_state.constraints[i]["b"] = num_input_no_step(
            "b (RHS)", con["b"], f"b_{i}", disabled=not row_enabled
        )

    with c_act:
        st.markdown("<div style='height:1.9em'></div>", unsafe_allow_html=True)
        enable_col, remove_col = st.columns([1, 1])

        with enable_col:
            st.session_state.constraints[i]["enabled"] = st.checkbox(
                "Enable", value=con["enabled"], key=f"enabled_{i}"
            )

        with remove_col:
            st.button("Remove", key=f"remove_{i}", use_container_width=True,
                    on_click=lambda idx=i: remove_constraint(idx))



# Non-negativity in constraint section
if "nonneg" in st.session_state:
    nonneg = st.checkbox(
        "Enforce non-negativity (x₁ ≥ 0, x₂ ≥ 0)",
        key="nonneg",
    )
else:
    nonneg = st.checkbox(
        "Enforce non-negativity (x₁ ≥ 0, x₂ ≥ 0)",
        value=True,
        key="nonneg",
    )

# Converting ops to standard LP matrices
def to_standard_matrices():
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in st.session_state.constraints:
        if not con.get("enabled", True):
            continue
        a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]
        if op in ("<", "≤"):
            A_ub.append([a1, a2]); b_ub.append(b)
        elif op in (">", "≥"):
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

def compute_axis_max(constraints, safety: float = 1.1) -> float:
    """
    Max axis extent based on max over constraints of |b| / min_positive(|a1|, |a2|),
    padded by `safety`. Skips disabled constraints and zero coefficients.
    """
    candidates = []
    for con in constraints:
        if not con.get("enabled", True):
            continue
        a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"])
        coeffs_pos = [abs(a) for a in (a1, a2) if abs(a) > 0.0]
        if not coeffs_pos:
            continue
        min_coeff = min(coeffs_pos)
        val = abs(b) / min_coeff
        if np.isfinite(val):
            candidates.append(val)

    base = max(candidates) if candidates else 100.0  # sensible fallback
    return float(base * safety)

# Dynamic scaling based on constraints
axis_max = compute_axis_max(st.session_state.constraints, safety=1.1)

# Keep lower bound at 0 for your current nonneg modeling;
# if you ever turn off nonneg and want a little room below 0:
# xmin = 0.0 if nonneg else -0.1 * axis_max
xmin, ymin = 0.0, 0.0
xmax = ymax = axis_max

x1 = np.linspace(xmin, xmax, 500)
x2 = np.linspace(ymin, ymax, 500)
X1, X2 = np.meshgrid(x1, x2)
Z = c1 * X1 + c2 * X2

# Feasible mask using the actual ops
mask = np.ones_like(X1, dtype=bool)
for con in st.session_state.constraints:
    if not con.get("enabled", True):
        continue
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

# will hold current optimal solution (if any)
x_opt = None
z_opt = None

# Build Plotly Figure (3D if show_obj, else 2D)
if show_obj:
    
    # 3D MODE
    fig = go.Figure()

    # Base objective surface
    fig.add_trace(go.Surface(
        x=x1, y=x2, z=Z,
        colorscale="Greys",
        opacity=0.35,
        showscale=False,
        name="Objective Plane"
    ))

    
    feasible_points = np.column_stack((X1.flatten(), X2.flatten()))
    feasible_points = feasible_points[mask.flatten().astype(bool)]

    if len(feasible_points) > 2:
        hull = ConvexHull(feasible_points)
        hull_vertices = feasible_points[hull.vertices]

        # Filled feasible region (green translucent area)
        # fig.add_trace(go.Mesh3d(
        #     x=hull_vertices[:, 0],
        #     y=hull_vertices[:, 1],
        #     z=np.zeros_like(hull_vertices[:, 0]),
        #     color='green',
        #     opacity=0.4,
        #     name="Feasible Region",
        #     alphahull=0,
        #     showscale=False
        # ))

        # Outline of feasible region (dark green border)
        # fig.add_trace(go.Scatter3d(
        #     x=np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
        #     y=np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
        #     z=np.zeros(len(hull_vertices) + 1),
        #     mode='lines',
        #     line=dict(color='darkgreen', width=5),
        #     name="Feasible Boundary"
        # ))

    # --- Projected feasible boundary onto the objective plane ---
        z_plane = c1 * hull_vertices[:, 0] + c2 * hull_vertices[:, 1]
        fig.add_trace(go.Scatter3d(
            x=np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
            y=np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
            z=np.append(z_plane, z_plane[0]),
            mode='lines',
            line=dict(color='limegreen', width=5, dash='dot'),
            name="Feasible Boundary (on Plane)"
        ))



    # Feasible region overlay
    lift = 0.15
    Z_feas = np.where(mask, Z + lift, np.nan)
    fig.add_trace(go.Surface(
        x=x1, y=x2, z=Z_feas,
        colorscale="Blues",
        opacity=0.85,
        showscale=False,
        name="Feasible Region"
    ))

    # Constraint boundary lines
    constraint_colors = [
        "#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"
    ]
    for idx, con in enumerate(st.session_state.constraints):
        if not con.get("enabled", True):
            continue
        a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]

        seg = constraint_segment(a1, a2, b, xmin, xmax, ymin, ymax)
        if seg is None:
            continue

        (xA, yA), (xB, yB) = seg
        zA = c1 * xA + c2 * yA
        zB = c1 * xB + c2 * yB
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
    
    # Objective level lines (3D)
    if z_levels:
        for z0 in z_levels:
            seg = constraint_segment(c1, c2, z0, xmin, xmax, ymin, ymax)
            if seg is None:
                continue
            (xA, yA), (xB, yB) = seg
            fig.add_trace(go.Scatter3d(
                x=[xA, xB],
                y=[yA, yB],
                z=[z0, z0],
                mode="lines+text",
                line=dict(width=6, color="black"),
                text=[f"Z = {fmt(z0)}", ""],
                textposition="top center",
                name=f"Z = {fmt(z0)}",
                hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>Z=%{z:.0f}<extra></extra>",
            ))

    # Optimal solution marker
    if res.success:
        x_opt = res.x
        z_opt = c1 * x_opt[0] + c2 * x_opt[1]
        fig.add_trace(go.Scatter3d(
            x=[x_opt[0]], y=[x_opt[1]], z=[z_opt + lift],
            mode="markers+text",
            text=[f"({fmt(x_opt[0])}, {fmt(x_opt[1])}, {fmt(z_opt)})"],
            textposition="top center",
            marker=dict(size=4, color="red"),
            name="Optimal Solution"
        ))
    else:
        st.error(f"Solve failed (status {res.status}): {res.message}")

    # 3D camera so it looks clearly 3D (not top-down)
    camera = dict(
        eye=dict(x=1.6, y=1.6, z=0.9)  # tweak to taste
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="x₁",
            yaxis_title="x₂",
            zaxis_title="Objective value (Z)",
            camera=camera,
        ),
        template="plotly_dark",                  
        paper_bgcolor="rgba(0, 0, 0, 1)",         # dark background
        plot_bgcolor="rgba(0, 0, 0, 1)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
        legend=dict(itemsizing="constant")
    )

    ## fig.update_layout({"uirevision": "foo"}, overwrite=True)

else:
    # 2D MODE
    fig = go.Figure()

    # Feasible region shading (2D) — FIXED
    feas_numeric = np.where(mask, 1.0, np.nan)
    fig.add_trace(go.Heatmap(
        x=x1, y=x2, z=feas_numeric,   # ← was feas_numeric.T
        colorscale="Blues",
        showscale=False,
        opacity=0.6,
        name="Feasible Region"
    ))

    # Constraint boundary lines (2D projection)
    constraint_colors = [
        "#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"
    ]
    for idx, con in enumerate(st.session_state.constraints):
        if not con.get("enabled", True):
            continue
        a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]

        seg = constraint_segment(a1, a2, b, xmin, xmax, ymin, ymax)
        if seg is None:
            continue

        (xA, yA), (xB, yB) = seg
        dash = "solid" if op == "=" else "dash"

        fig.add_trace(go.Scatter(
            x=[xA, xB],
            y=[yA, yB],
            mode="lines",
            line=dict(width=3, color=constraint_colors[idx % len(constraint_colors)], dash=dash),
            name=f"Constraint {idx+1}: {a1:.2f}·x₁ + {a2:.2f}·x₂ {op} {b:.2f}",
            hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<extra></extra>",
        ))

    # Objective level lines (2D)
    if z_levels:
        for z0 in z_levels:
            seg = constraint_segment(c1, c2, z0, xmin, xmax, ymin, ymax)
            if seg is None:
                continue
            (xA, yA), (xB, yB) = seg
            midx, midy = (xA + xB) / 2.0, (yA + yB) / 2.0
            fig.add_trace(go.Scatter(
                x=[xA, xB],
                y=[yA, yB],
                mode="lines",
                line=dict(width=4, color="black"),
                name=f"Z = {fmt(z0)}",
                hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<extra></extra>",
            ))
            fig.add_annotation(
                x=midx, y=midy, text=f"Z = {fmt(z0)}",
                showarrow=False, font=dict(size=12), yshift=8
            )

    # Optimal solution marker (2D)
    if res.success:
        x_opt = res.x
        z_opt = c1 * x_opt[0] + c2 * x_opt[1]
        fig.add_trace(go.Scatter(
            x=[x_opt[0]], y=[x_opt[1]],
            mode="markers+text",
            text=[f"({fmt(x_opt[0])}, {fmt(x_opt[1])})"],
            textposition="top center",
            marker=dict(size=8, color="red", symbol="x"),
            name="Optimal Solution"
        ))

    else:
        st.error(f"Solve failed (status {res.status}): {res.message}")

    fig.update_layout(
        xaxis_title="x₁",
        yaxis_title="x₂",
        template="plotly_dark",             # dark
        paper_bgcolor="rgba(0, 0, 0, 1)",
        plot_bgcolor="rgba(0, 0, 0, 1)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
        legend=dict(itemsizing="constant"),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"toImageButtonOptions": {"format": "png", "filename": "lp_graph"}}
)