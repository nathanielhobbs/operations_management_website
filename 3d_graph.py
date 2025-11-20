import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import html
import json 

st.title("3D Linear Programming Visualizer (Interactive)")


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
if "hide_all" not in st.session_state:
    st.session_state.hide_all = False

def toggle_visibility():
    st.session_state.hide_all = not st.session_state.hide_all

# Always show this button
st.button("Disappear / Reappear", on_click=toggle_visibility)


# -----------------------------
# Center Screen Inputs
# -----------------------------

if not st.session_state.hide_all:

    # Callback functions to update c1 and c2
    def update_c1():
        st.session_state.c1 = st.session_state.c1_input
        st.session_state.show_c1_input = False  # Optionally hide input after change

    def update_c2():
        st.session_state.c2 = st.session_state.c2_input
        st.session_state.show_c2_input = False  # Optionally hide input after change

    def update_constraints():
        st.session_state.constraints = st.session_state.constraints_input
        st.session_state.show_constraints_input = False

    # Initialize session state variables
    if "c1" not in st.session_state:
        st.session_state.c1 = 1
    if "c2" not in st.session_state:
        st.session_state.c2 = 1
    if "constraints" not in st.session_state:
        st.session_state.constraints = 1
    if 'show_c1_input' not in st.session_state:
        st.session_state.show_c1_input = False
    if 'show_c2_input' not in st.session_state:
        st.session_state.show_c2_input = False
    if 'show_constraints_input' not in st.session_state:
        st.session_state.show_constraints_input = False
    



    # Functions to toggle input visibility
    def toggle_c1_input():
        st.session_state.show_c1_input = not st.session_state.show_c1_input

    def toggle_c2_input():
        st.session_state.show_c2_input = not st.session_state.show_c2_input

    def toggle_constraints_input():
        st.session_state.show_constraints_input = not st.session_state.show_constraints_input

    # Display buttons
    flex = st.container(horizontal=True)
    flex.button("Value of c1", on_click=toggle_c1_input)
    flex.button("Value of c2", on_click=toggle_c2_input)
    flex.button("Number of Constraints", on_click=toggle_constraints_input)
    flex.button("Preset 1", on_click=apply_preset1)
    flex.button("Preset from File", on_click=handle_preset_from_file_button)

    # show uploader only when requested
    if st.session_state.show_preset_upload:
        st.file_uploader(
            "Upload preset file (JSON)",
            type=["json"],
            key="preset_file"
        )

    # Display text boxes based on state
    if st.session_state.show_c1_input:
        st.number_input(
            "Set c1",
            value=st.session_state.c1,
            key="c1_input",
            on_change=lambda: (st.session_state.update({"c1": st.session_state.c1_input, "show_c1_input": False}))
        )

    if st.session_state.show_c2_input:
        st.number_input(
            "Set c2",
            value=st.session_state.c2,
            key="c2_input",
            on_change=lambda: (st.session_state.update({"c2": st.session_state.c2_input, "show_c2_input": False}))
        )

    if st.session_state.show_constraints_input:
        st.number_input(
            "Set Number of Constraints",
            value=st.session_state.constraints,
            key="constraints_input",
            on_change=lambda: (st.session_state.update({"constraints": st.session_state.constraints_input, "show_constraints_input": False}))
        )

# Values for calculation
c1 = st.session_state.c1
c2 = st.session_state.c2
n_constraints = st.session_state.constraints

dex = st.container(horizontal=True)
show_obj = dex.checkbox("Show Objective Plane", value=True)
maximize = dex.checkbox("Maximize objective", value=True)
minimize = dex.checkbox("Minimize objective", value=True)

constraints = []
for i in range(int(n_constraints)):
    a1 = st.sidebar.number_input(f"Constraint {i+1}: coeff of x1", value=1.0, key=f"a1_{i}")
    a2 = st.sidebar.number_input(f"Constraint {i+1}: coeff of x2", value=1.0, key=f"a2_{i}")
    b = st.sidebar.number_input(f"Constraint {i+1}: RHS (b)", value=10.0, key=f"b_{i}")
    constraints.append((a1, a2, b))

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

feasible_points = np.column_stack((X1.flatten(), X2.flatten()))
feasible_points = feasible_points[mask.flatten().astype(bool)]

if len(feasible_points) > 2:
    hull = ConvexHull(feasible_points)
    hull_vertices = feasible_points[hull.vertices]

    # Filled feasible region (green translucent area)
    fig.add_trace(go.Mesh3d(
        x=hull_vertices[:, 0],
        y=hull_vertices[:, 1],
        z=np.zeros_like(hull_vertices[:, 0]),
        color='green',
        opacity=0.4,
        name="Feasible Region",
        alphahull=0,
        showscale=False
    ))

    # Outline of feasible region (dark green border)
    fig.add_trace(go.Scatter3d(
        x=np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
        y=np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
        z=np.zeros(len(hull_vertices) + 1),
        mode='lines',
        line=dict(color='darkgreen', width=5),
        name="Feasible Boundary"
    ))

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

    # ----------------------------------------
    # Draw each constraint line in its own color
    # ----------------------------------------
    constraint_colors = ["red", "blue", "orange", "purple",
                         "cyan", "magenta", "yellow", "white"]
    tol = 1e-3  # tolerance for "binding" check
    x1_line_base = np.linspace(0, 10, 200)

    for i, (a1, a2, rhs) in enumerate(constraints, start=1):
        # boundary: a1*x1 + a2*x2 = rhs  =>  x2 = (rhs - a1*x1)/a2
        if a2 == 0:
            # vertical constraint in x1 – skip or handle separately
            continue

        x1_line = x1_line_base.copy()
        x2_line = (rhs - a1 * x1_line) / a2

        # keep only nonnegative x2
        mask = x2_line >= 0
        x1_line = x1_line[mask]
        x2_line = x2_line[mask]
        if len(x1_line) == 0:
            continue

        # z on ground plane and on objective plane
        z_ground = np.zeros_like(x1_line)
        z_on_plane = c1 * x1_line + c2 * x2_line

        # is this constraint binding at the optimum?
        lhs_opt = a1 * x_opt[0] + a2 * x_opt[1]
        is_binding = abs(lhs_opt - rhs) <= tol

        color = constraint_colors[(i - 1) % len(constraint_colors)]
        width = 7 if is_binding else 3  # thicker if binding

        # line on ground (z = 0)
        fig.add_trace(go.Scatter3d(
            x=x1_line,
            y=x2_line,
            z=z_ground,
            mode="lines",
            name=f"Constraint {i}",
            line=dict(color=color, width=width),
            showlegend=True
        ))

        # same line projected onto the objective plane (no extra legend)
        fig.add_trace(go.Scatter3d(
            x=x1_line,
            y=x2_line,
            z=z_on_plane,
            mode="lines",
            line=dict(color=color, width=width, dash="dot"),
            showlegend=False
        ))

if not st.session_state.hide_all:
    if res.success:
            st.success(f"Optimal solution: x1 = {x_opt[0]:.2f}, x2 = {x_opt[1]:.2f}, objective = {z_opt:.2f}")
    else:
        st.error("No feasible solution found.")

st.latex(f"Objective Function: Z = (c1)x1 + (c2)x2 = ({st.session_state.c1})x1 + ({st.session_state.c2})x2 = {st.session_state.c1*x_opt[0]:.2f} + {st.session_state.c2*x_opt[1]:.2f} = {st.session_state.c1*x_opt[0] + st.session_state.c2*x_opt[1]:.2f}")

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="Objective value (z)",
        zaxis=dict(range=[0, np.max(Z)], showspikes=False),
        xaxis=dict(showspikes=False),
        yaxis=dict(showspikes=False),
    ),
    # hovermode=False,            # Disable crosshair drawing
    hoverlabel=dict(namelength=-1),
    margin=dict(l=0, r=0, t=40, b=0),
    height=700,
    title="Feasible Region and Objective Plane"
)

# stop reorient when toggling the graph features
# minimize = negative objective


st.plotly_chart(fig, use_container_width=True)

