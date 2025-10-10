import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

st.title("3D Linear Programming Visualizer (Interactive)")


# -----------------------------
# Sidebar Inputs
# -----------------------------

# st.sidebar.header("Objective Function")
# c1 = st.sidebar.number_input("Coefficient for x1 (c1)", value=2.0)
# c2 = st.sidebar.number_input("Coefficient for x2 (c2)", value=3.0)


# -----------------------------
# Center Screen Inputs
# -----------------------------

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
    st.session_state.constraints = 0
if 'show_c1_input' not in st.session_state:
    st.session_state.show_c1_input = False
if 'show_c2_input' not in st.session_state:
    st.session_state.show_c2_input = False
if 'show_constraints_input' not in st.session_state:
    st.session_state.show_constraints_input = False


# Functions to show input for c1 and c2
def show_c1_input():
    st.session_state.show_c1_input = True

def show_c2_input():
    st.session_state.show_c2_input = True

def show_constraints_input():
    st.session_state.show_constraints_input = True

flex = st.container(horizontal=True)
flex.button("Value of c1", on_click=show_c1_input)
flex.button("Value of c2", on_click=show_c2_input)
flex.button("Number of Constraints", on_click=show_constraints_input)

# Show input boxes if corresponding button was clicked, with callbacks
if st.session_state.show_c1_input:
    st.number_input(
        "Set c1",
        value=st.session_state.c1,
        key="c1_input",
        on_change=update_c1
    )
    #Optionally, hide input after change
    st.session_state.show_c1_input = False

if st.session_state.show_c2_input:
    st.number_input(
        "Set c2",
        value=st.session_state.c2,
        key="c2_input",
        on_change=update_c2
    )
    # Optionally, hide input after change
    st.session_state.show_c2_input = False

if st.session_state.show_constraints_input:
    st.number_input(
        "Set Number of Constraints",
        value=st.session_state.constraints,
        key="constraints_input",
        on_change=update_constraints
    )

# Values for calculation
c1 = st.session_state.c1
c2 = st.session_state.c2
n_constraints = st.session_state.constraints

dex = st.container(horizontal=True)
show_obj = dex.checkbox("Show Objective Plane", value=True)
maximize = dex.checkbox("Maximize objective", value=True)
minimize = dex.checkbox("Minimize objective", value=True)

st.sidebar.header("Constraints (Ax ≤ b)")
n_constraints = st.sidebar.number_input("Number of constraints", min_value=1, max_value=5, value=2)

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

st.latex(f"Objective Function: Z = (c1)x1 + (c2)x2 = ({st.session_state.c1})x1 + ({st.session_state.c2})x2 = {st.session_state.c1*x_opt[0]:.2f} + {st.session_state.c2*x_opt[1]:.2f} = {st.session_state.c1*x_opt[0] + st.session_state.c2*x_opt[1]:.2f}")


# # -----------------------------
# # Center Screen Inputs
# # -----------------------------

# # Callback functions to update c1 and c2
# def update_c1():
#     st.session_state.c1 = st.session_state.c1_input
#     st.session_state.show_c1_input = False  # Optionally hide input after change

# def update_c2():
#     st.session_state.c2 = st.session_state.c2_input
#     st.session_state.show_c2_input = False  # Optionally hide input after change

# # Initialize session state variables
# if "c1" not in st.session_state:
#     st.session_state.c1 = 1
# if "c2" not in st.session_state:
#     st.session_state.c2 = 1
# if 'show_c1_input' not in st.session_state:
#     st.session_state.show_c1_input = False
# if 'show_c2_input' not in st.session_state:
#     st.session_state.show_c2_input = False


# # Functions to show input for c1 and c2
# def show_c1_input():
#     st.session_state.show_c1_input = True

# def show_c2_input():
#     st.session_state.show_c2_input = True

# flex = st.container(horizontal=True)
# flex.button("Value of c1", on_click=show_c1_input)
# flex.button("Value of c2", on_click=show_c2_input)
# #flex.button()

# # Show input boxes if corresponding button was clicked, with callbacks
# if st.session_state.show_c1_input:
#     st.number_input(
#         "Set c1",
#         value=st.session_state.c1,
#         key="c1_input",
#         on_change=update_c1
#     )
#     #Optionally, hide input after change
#     st.session_state.show_c1_input = False

# if st.session_state.show_c2_input:
#     st.number_input(
#         "Set c2",
#         value=st.session_state.c2,
#         key="c2_input",
#         on_change=update_c2
#     )
#     # Optionally, hide input after change
#     st.session_state.show_c2_input = False
    
    
# st.latex(f"Objective Function: Z = (c1)x1 + (c2)x2 = ({st.session_state.c1})x1 + ({st.session_state.c2})x2 = {st.session_state.c1*x_opt[0]:.2f} + {st.session_state.c2*x_opt[1]:.2f} = {st.session_state.c1*x_opt[0] + st.session_state.c2*x_opt[1]:.2f}")


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

