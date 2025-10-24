import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
import html

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

# --- Recovery: if my_value is accidentally a type object, reset it ---
if "my_value" in st.session_state and isinstance(st.session_state.my_value, type):
    # remove the bad value so initialization below will set a correct one
    del st.session_state["my_value"]

# Initialize stored value (safe)
if "my_value" not in st.session_state:
    st.session_state.my_value = 0.0  # numeric default

# Use a unique key for the hidden widget to avoid collisions
_hidden_key = "_editable_temp_val_unique_key"

# Ensure the hidden key exists in session_state as a string representation
if _hidden_key not in st.session_state:
    st.session_state[_hidden_key] = str(st.session_state.my_value)

# Helper: sanitize incoming text so we don't accidentally accept docstrings or "class 'int'"
def sanitize_input_text(s: str):
    if s is None:
        return None
    s = s.strip()
    # Reject obviously-non-numeric inputs (docstrings often contain '->' or '-> integer' or 'class')
    if ("->" in s) or ("class " in s) or s.lower().startswith("<class"):
        return None
    # Accept normal numeric strings (including leading +, -, decimals)
    return s

# HTML+JS editable box (click to edit). Note we html-escape the value when inserting to HTML.
escaped_value = html.escape(str(st.session_state.my_value))
components.html(f"""
    <div id="editableBox" 
         style="
            border:1px solid #ccc;
            width:80px;
            height:40px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:5px;
            background-color:#f9f9f9;
            cursor:pointer;
            font-size:18px;
            user-select:none;
        "
        onclick="this.style.display='none';
                 const inp = document.getElementById('inputBox');
                 inp.style.display='block';
                 inp.focus();
                 inp.select();">
        {escaped_value}
    </div>

    <input id="inputBox" type="number" step='any' 
           value="{escaped_value}" 
           style="
               display:none;
               width:80px;
               height:40px;
               text-align:center;
               font-size:18px;
               border-radius:5px;
               border:1px solid #ccc;
           "
           onblur="
               // When leaving input, post the numeric value to Streamlit and restore the display
               const payload = {{type: 'streamlit:setNumber', value: this.value}};
               window.parent.postMessage(payload, '*');

               this.style.display='none';
               const box = document.getElementById('editableBox');
               box.innerText = this.value === '' ? '0' : this.value;
               box.style.display='flex';
           ">
    <script>
        // Listen for messages (e.g. from parent) - not strictly necessary here, but kept for completeness
        window.addEventListener('message', (event) => {{
            if (!event.data) return;
            // When we receive streamlit:setNumber, forward to Streamlit's internal handler
            if (event.data.type === 'streamlit:setNumber') {{
                const newVal = event.data.value;
                // This special message pattern tells Streamlit to set the component value for the hidden widget.
                window.parent.postMessage({{ isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: newVal }}, '*');
            }}
        }});
    </script>
""", height=80)

# Hidden text_input that captures the value coming from JS.
# IMPORTANT: key is unique and unlikely to collide with anything else.
temp_val = st.text_input("hidden temp input", value=st.session_state[_hidden_key], key=_hidden_key, label_visibility="collapsed")

# If it changed, sanitize and update session_state.my_value
if temp_val != str(st.session_state.my_value):
    sanitized = sanitize_input_text(temp_val)
    if sanitized is not None:
        try:
            # parse as float (allows decimals); convert to int if it looks like an int
            parsed = float(sanitized)
            if parsed.is_integer():
                parsed = int(parsed)
            st.session_state.my_value = parsed
            # keep the hidden widget string in sync
            st.session_state[_hidden_key] = str(parsed)
        except Exception:
            # parsing failed; ignore (don't overwrite my_value)
            pass
    else:
        # Received something suspicious (docstring or 'class int'), ignore and restore the hidden widget
        st.session_state[_hidden_key] = str(st.session_state.my_value)

# UI: show stored value and result
st.write(f"**Stored value:** {st.session_state.my_value!s}")
# Use a safe numeric test before computing
if isinstance(st.session_state.my_value, (int, float)):
    st.write(f"**Result (×2):** {st.session_state.my_value * 2}")
else:
    st.write("**Result (×2):** Invalid (not numeric)")

# Optional: a debug / recovery button the user can press during development if something still goes wrong
if st.button("Reset stored value to 0"):
    st.session_state.my_value = 0
    st.session_state[_hidden_key] = "0"
    st.experimental_rerun()


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

# for i in range(int(n_constraints)):
#     a1 = st.sidebar.number_input(f"Constraint {i+1}: coeff of x1", value=1.0, key=f"a1_{i}")
#     a2 = st.sidebar.number_input(f"Constraint {i+1}: coeff of x2", value=1.0, key=f"a2_{i}")
#     b = st.sidebar.number_input(f"Constraint {i+1}: RHS (b)", value=10.0, key=f"b_{i}")
#     st.latex(f"Subjective Function:  = ({st.session_state.c1})x1 + ({st.session_state.c2})x2 = {st.session_state.c1*x_opt[0]:.2f} + {st.session_state.c2*x_opt[1]:.2f} = {st.session_state.c1*x_opt[0] + st.session_state.c2*x_opt[1]:.2f}")
#     constraints.append((a1, a2, b))


# # Initialize stored value
# if "my_value" not in st.session_state:
#     st.session_state.my_value = int(0)  # default value

# # Function to update session_state from JS input
# def update_value():
#     new_val = st.session_state.temp_val
#     st.session_state.my_value = new_val


# #st.write("DEBUG TYPE:", type(st.session_state.my_value))


# if not isinstance(st.session_state.my_value, (int, float)):
#     st.session_state.my_value = 0.0

# # HTML + JS interactive box
# components.html(f"""
#     <div id="editableBox" 
#          style="
#             border:1px solid #ccc;
#             width:80px;
#             height:40px;
#             display:flex;
#             align-items:center;
#             justify-content:center;
#             border-radius:5px;
#             background-color:#f9f9f9;
#             cursor:pointer;
#             font-size:18px;
#         "
#         onclick="this.style.display='none';
#                  document.getElementById('inputBox').style.display='block';
#                  document.getElementById('inputBox').focus();">
#         {st.session_state.my_value}
#     </div>

#     <input id="inputBox" type="number" step="any" 
#            value="{st.session_state.my_value}" 
#            style="
#                display:none;
#                width:80px;
#                height:40px;
#                text-align:center;
#                font-size:18px;
#                border-radius:5px;
#                border:1px solid #ccc;
#            "
#            onblur="
#                window.parent.postMessage(
#                    {type: 'streamlit:setNumber', value: this.value},
#                    '*'
#                );
#                this.style.display='none';
#                document.getElementById('editableBox').innerHTML = this.value;
#                document.getElementById('editableBox').style.display='flex';
#            ">
#     <script>
#         // Send the new number to Streamlit when it changes
#         window.addEventListener('message', (event) => {{
#             if (event.data.type === 'streamlit:setNumber') {{
#                 const newVal = event.data.value;
#                 window.parent.postMessage({{ isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: newVal }}, '*');
#             }}
#         }});
#     </script>
# """, height=70)

# # Capture the new value from JS
# temp_val = st.text_input("Hidden temp input", value=st.session_state.my_value, key="temp_val", label_visibility="collapsed")
# if temp_val != str(st.session_state.my_value):
#     try:
#         st.session_state.my_value = float(temp_val)
#     except ValueError:
#         pass

# st.write(f"**Stored value:** {st.session_state.my_value}")
# st.write(f"**Result (×2):** {st.session_state.my_value * 2}")














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

