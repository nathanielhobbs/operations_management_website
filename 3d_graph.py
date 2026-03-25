import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import html
import json 
from types import SimpleNamespace
from pyomo.environ import (
    ConcreteModel, Var, Objective, ConstraintList,
    NonNegativeReals, Reals, NonNegativeIntegers, Integers,
    maximize, minimize, SolverFactory, Suffix, value
)
from pyomo.opt import SolverStatus, TerminationCondition

LE_OPS = {"<", "<=", "≤", "â‰¤"}
GE_OPS = {">", ">=", "≥", "â‰¥"}
EQ_OPS = {"="}


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

def int_input_no_step(label: str, default: int | float, key: str, disabled: bool = False) -> int:
    if key in st.session_state:
        s = st.text_input(label, key=key, disabled=disabled)
    else:
        s = st.text_input(label, value=str(int(round(float(default)))), key=key, disabled=disabled)

    try:
        v = float(str(s).strip())
        if not v.is_integer():
            if disabled:
                return int(round(float(default)))
            st.error(f"{label} must be an integer.")
            st.stop()
        return int(v)
    except ValueError:
        if disabled:
            return int(round(float(default)))
        st.error(f"Enter a valid integer for {label}.")
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

st.markdown("**Decision variable type (Integer Programming)**")
iv_col1, iv_col2 = st.columns(2)
with iv_col1:
    x1_is_int = st.toggle("x1 is integer", value=False, key="x1_is_int")
with iv_col2:
    x2_is_int = st.toggle("x2 is integer", value=False, key="x2_is_int")

if x1_is_int or x2_is_int:
    st.caption("Integer mode on: selected decision variable(s) are enforced as integers in the solver.")

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
        c1 = num_input_no_step("c1", 2.0, "c1")
    with _c1_r:
        st.markdown("<div style='margin-top:1.9rem;'>x₁</div>", unsafe_allow_html=True)

with colPlus:
    st.markdown("<div style='margin-top:1.9rem;'>+</div>", unsafe_allow_html=True)

# c2 · x2
with colC:
    _c2_l, _c2_r = st.columns([1, 0.25])
    with _c2_l:
        c2 = num_input_no_step("c2", 3.0, "c2")
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
    "a1", con["a1"], f"a1_{i}", disabled=not row_enabled
)
    with c_x1:
        st.markdown("<div style='margin-top:1.9rem;'>x₁</div>", unsafe_allow_html=True)

    with c_plus:
        st.markdown("<div style='margin-top:1.9rem;'>+</div>", unsafe_allow_html=True)

    with c_a2:
        st.session_state.constraints[i]["a2"] = num_input_no_step(
    "a2", con["a2"], f"a2_{i}", disabled=not row_enabled
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

#--------------------------------------------------------------------------------------------------
# # Converting ops to standard LP matrices
# def to_standard_matrices():
#     A_ub, b_ub, A_eq, b_eq = [], [], [], []
#     for con in st.session_state.constraints:
#         if not con.get("enabled", True):
#             continue
#         a1 = float(con["a1"]); a2 = float(con["a2"]); b = float(con["b"]); op = con["op"]
#         if op in ("<", "≤"):
#             A_ub.append([a1, a2]); b_ub.append(b)
#         elif op in (">", "≥"):
#             A_ub.append([-a1, -a2]); b_ub.append(-b)
#         else:  # "="
#             A_eq.append([a1, a2]); b_eq.append(b)
#     A_ub = np.array(A_ub) if A_ub else None
#     b_ub = np.array(b_ub) if b_ub else None
#     A_eq = np.array(A_eq) if A_eq else None
#     b_eq = np.array(b_eq) if b_eq else None
#     return A_ub, b_ub, A_eq, b_eq

# # Solve LP
# A_ub, b_ub, A_eq, b_eq = to_standard_matrices()
# c = np.array([c1, c2])
# c_obj = -c if sense == "Maximize" else c
# bounds = [(0, None), (0, None)] if nonneg else [(None, None), (None, None)]

# res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
#---------------------------------------------------------------------------------------------------------



#OLD SOLVER CODE:

## -------------------------------------------------------------------------------------------------------------

# def solve_lp_with_pyomo(c1, c2, sense, nonneg, constraints, x1_integer=False, x2_integer=False, write_ranges=False, ranges_file="sens.txt"):
#     m = ConcreteModel()

#     if x1_integer:
#         x1_domain = NonNegativeIntegers if nonneg else Integers
#     else:
#         x1_domain = NonNegativeReals if nonneg else Reals

#     if x2_integer:
#         x2_domain = NonNegativeIntegers if nonneg else Integers
#     else:
#         x2_domain = NonNegativeReals if nonneg else Reals

#     m.x1 = Var(domain=x1_domain)
#     m.x2 = Var(domain=x2_domain)


#     m.obj = Objective(
#         expr=c1 * m.x1 + c2 * m.x2,
#         sense=maximize if sense == "Maximize" else minimize
#     )

#     m.cons = ConstraintList()
#     con_meta = []  # store index + expression metadata for reporting

#     for i, con in enumerate(constraints):
#         if not con.get("enabled", True):
#             continue

#         a1 = float(con["a1"])
#         a2 = float(con["a2"])
#         b = float(con["b"])
#         op = con["op"]

#         lhs = a1 * m.x1 + a2 * m.x2
#         if op in LE_OPS:
#             m.cons.add(lhs <= b)
#         elif op in GE_OPS:
#             m.cons.add(lhs >= b)
#         elif op in EQ_OPS:
#             m.cons.add(lhs == b)
#         else:
#             return SimpleNamespace(success=False, status=-1, message=f"Unsupported operator: {op}", x=None), None

#         con_meta.append({"idx": i + 1, "a1": a1, "a2": a2, "b": b, "op": op, "con_obj": m.cons[len(m.cons)]})

#     # Sensitivity suffixes
#     m.dual = Suffix(direction=Suffix.IMPORT)  # shadow prices
#     m.rc = Suffix(direction=Suffix.IMPORT)    # reduced costs

#     opt = SolverFactory("glpk")
#     if not opt.available(exception_flag=False):
#         return SimpleNamespace(success=False, status=-1, message="GLPK solver not available (glpsol not found).", x=None), None

#     if write_ranges:
#         opt.options["ranges"] = ranges_file

#     py_res = opt.solve(
#         m,
#         tee=False,
#         keepfiles=bool(write_ranges),
#         symbolic_solver_labels=bool(write_ranges)
#     )

#     ok = (
#         py_res.solver.status == SolverStatus.ok and
#         py_res.solver.termination_condition == TerminationCondition.optimal
#     )

#     if not ok:
#         msg = f"{py_res.solver.status} / {py_res.solver.termination_condition}"
#         return SimpleNamespace(success=False, status=-1, message=msg, x=None), None

#     x_opt = np.array([float(value(m.x1)), float(value(m.x2))])

#     # Build sensitivity report
#     sens = {
#         "reduced_costs": {
#             "x1": float(m.rc.get(m.x1, np.nan)),
#             "x2": float(m.rc.get(m.x2, np.nan)),
#         },
#         "constraints": []
#     }

#     for row in con_meta:
#         c = row["con_obj"]
#         lhs_val = float(value(c.body))
#         lb = None if c.lower is None else float(value(c.lower))
#         ub = None if c.upper is None else float(value(c.upper))

#         sens["constraints"].append({
#             "constraint_index": row["idx"],
#             "expr": f"{row['a1']}*x1 + {row['a2']}*x2 {row['op']} {row['b']}",
#             "shadow_price": float(m.dual.get(c, 0.0)),
#             "lhs_value": lhs_val,
#             "lower_bound": lb,
#             "upper_bound": ub,
#             "slack_to_upper": (ub - lhs_val) if ub is not None else None,
#             "slack_to_lower": (lhs_val - lb) if lb is not None else None,
#         })

#     # Match your existing downstream usage: res.success, res.x, res.status, res.message
#     return SimpleNamespace(success=True, status=0, message="Optimal", x=x_opt), sens


## ----------------------------------------------------------------------------------------------------


## NEW SOLVER CODE:

##-------------------------------------------------------------------------------------------------

def normalize_solution_value(val, must_be_integer, tol=1e-6):
    v = float(val)
    if must_be_integer and abs(v - round(v)) <= tol:
        return float(int(round(v)))
    return v


def build_pyomo_model(c1, c2, sense, nonneg, constraints, x1_integer=False, x2_integer=False, with_suffixes=False):
    m = ConcreteModel()

    if x1_integer:
        x1_domain = NonNegativeIntegers if nonneg else Integers
    else:
        x1_domain = NonNegativeReals if nonneg else Reals

    if x2_integer:
        x2_domain = NonNegativeIntegers if nonneg else Integers
    else:
        x2_domain = NonNegativeReals if nonneg else Reals

    m.x1 = Var(domain=x1_domain)
    m.x2 = Var(domain=x2_domain)

    m.obj = Objective(
        expr=c1 * m.x1 + c2 * m.x2,
        sense=maximize if sense == "Maximize" else minimize
    )

    m.cons = ConstraintList()
    con_meta = []

    for i, con in enumerate(constraints):
        if not con.get("enabled", True):
            continue

        a1 = float(con["a1"])
        a2 = float(con["a2"])
        b = float(con["b"])
        op = con["op"]

        lhs = a1 * m.x1 + a2 * m.x2
        if op in LE_OPS:
            m.cons.add(lhs <= b)
        elif op in GE_OPS:
            m.cons.add(lhs >= b)
        elif op in EQ_OPS:
            m.cons.add(lhs == b)
        else:
            raise ValueError(f"Unsupported operator: {op}")

        con_meta.append({
            "idx": i + 1,
            "a1": a1,
            "a2": a2,
            "b": b,
            "op": op,
            "con_obj": m.cons[len(m.cons)]
        })

    if with_suffixes:
        m.dual = Suffix(direction=Suffix.IMPORT)
        m.rc = Suffix(direction=Suffix.IMPORT)

    return m, con_meta


def extract_sensitivity_report(m, con_meta):
    sens = {
        "reduced_costs": {
            "x1": float(m.rc.get(m.x1, np.nan)),
            "x2": float(m.rc.get(m.x2, np.nan)),
        },
        "constraints": []
    }

    for row in con_meta:
        c = row["con_obj"]
        lhs_val = float(value(c.body))
        lb = None if c.lower is None else float(value(c.lower))
        ub = None if c.upper is None else float(value(c.upper))

        sens["constraints"].append({
            "constraint_index": row["idx"],
            "expr": f"{row['a1']}*x1 + {row['a2']}*x2 {row['op']} {row['b']}",
            "shadow_price": float(m.dual.get(c, 0.0)),
            "lhs_value": lhs_val,
            "lower_bound": lb,
            "upper_bound": ub,
            "slack_to_upper": (ub - lhs_val) if ub is not None else None,
            "slack_to_lower": (lhs_val - lb) if lb is not None else None,
        })

    return sens


def solve_lp_with_pyomo(c1, c2, sense, nonneg, constraints, x1_integer=False, x2_integer=False, write_ranges=False, ranges_file="sens.txt"):
    mip_mode = x1_integer or x2_integer

    try:
        m, con_meta = build_pyomo_model(
            c1, c2, sense, nonneg, constraints,
            x1_integer=x1_integer,
            x2_integer=x2_integer,
            with_suffixes=not mip_mode
        )
    except ValueError as e:
        return SimpleNamespace(success=False, status=-1, message=str(e), x=None), None

    opt = SolverFactory("glpk")
    if not opt.available(exception_flag=False):
        return SimpleNamespace(success=False, status=-1, message="GLPK solver not available (glpsol not found).", x=None), None

    if write_ranges and not mip_mode:
        opt.options["ranges"] = ranges_file

    py_res = opt.solve(
        m,
        tee=False,
        keepfiles=bool(write_ranges and not mip_mode),
        symbolic_solver_labels=bool(write_ranges and not mip_mode)
    )

    ok = (
        py_res.solver.status == SolverStatus.ok and
        py_res.solver.termination_condition == TerminationCondition.optimal
    )

    if not ok:
        msg = f"{py_res.solver.status} / {py_res.solver.termination_condition}"
        return SimpleNamespace(success=False, status=-1, message=msg, x=None), None

    x_opt = np.array([
        normalize_solution_value(value(m.x1), x1_integer),
        normalize_solution_value(value(m.x2), x2_integer),
    ], dtype=float)

    if not mip_mode:
        sens = extract_sensitivity_report(m, con_meta)
        sens["source"] = "Integer-free LP"
        return SimpleNamespace(success=True, status=0, message="Optimal", x=x_opt), sens

    # For integer / mixed-integer solves:
    # classical LP sensitivity is not defined, so solve the LP relaxation separately.
    try:
        relax_m, relax_meta = build_pyomo_model(
            c1, c2, sense, nonneg, constraints,
            x1_integer=False,
            x2_integer=False,
            with_suffixes=True
        )
    except ValueError as e:
        return SimpleNamespace(success=True, status=0, message="Optimal", x=x_opt), {
            "source": "Unavailable",
            "note": f"Integer solution found, but LP-relaxation sensitivity could not be built: {e}",
            "reduced_costs": None,
            "constraints": []
        }

    relax_opt = SolverFactory("glpk")
    if relax_opt.available(exception_flag=False):
        if write_ranges:
            relax_opt.options["ranges"] = ranges_file

        relax_res = relax_opt.solve(
            relax_m,
            tee=False,
            keepfiles=bool(write_ranges),
            symbolic_solver_labels=bool(write_ranges)
        )

        relax_ok = (
            relax_res.solver.status == SolverStatus.ok and
            relax_res.solver.termination_condition == TerminationCondition.optimal
        )

        if relax_ok:
            sens = extract_sensitivity_report(relax_m, relax_meta)
            sens["source"] = "LP relaxation"
            sens["note"] = (
                "This model includes integer restrictions, so classical reduced costs and shadow prices "
                "are not defined for the integer solution. The values below come from the continuous LP relaxation."
            )
            return SimpleNamespace(success=True, status=0, message="Optimal", x=x_opt), sens

    sens = {
        "source": "Unavailable",
        "note": (
            "This model includes integer restrictions, so classical reduced costs and shadow prices "
            "are not defined for the integer solution."
        ),
        "reduced_costs": None,
        "constraints": []
    }
    return SimpleNamespace(success=True, status=0, message="Optimal", x=x_opt), sens


##------------------------------------------------------------------------------------------------------------------------------------------------------


# ---- call it where you currently do linprog ----
res, sensitivity = solve_lp_with_pyomo(
    c1=c1,
    c2=c2,
    sense=sense,
    nonneg=nonneg,
    constraints=st.session_state.constraints,
    x1_integer=x1_is_int,
    x2_integer=x2_is_int,
    write_ranges=True,
    ranges_file="sens.txt"
)

#__________________________________________________________________________________________________


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


def integer_overlay_from_mask(x_vals, y_vals, mask_arr, x1_int_flag, x2_int_flag):
    line_segments = []  # tuples: ("x"|"y", fixed_value, start, end)
    dot_points = []     # tuples: (x, y)

    if not (x1_int_flag or x2_int_flag):
        return line_segments, dot_points

    x_ints = range(int(np.ceil(x_vals.min())), int(np.floor(x_vals.max())) + 1)
    y_ints = range(int(np.ceil(y_vals.min())), int(np.floor(y_vals.max())) + 1)

    if x1_int_flag and not x2_int_flag:
        for xv in x_ints:
            j = int(np.argmin(np.abs(x_vals - xv)))
            y_feas = y_vals[mask_arr[:, j]]
            if len(y_feas) > 0:
                line_segments.append(("x", xv, float(np.min(y_feas)), float(np.max(y_feas))))

    elif x2_int_flag and not x1_int_flag:
        for yv in y_ints:
            i = int(np.argmin(np.abs(y_vals - yv)))
            x_feas = x_vals[mask_arr[i, :]]
            if len(x_feas) > 0:
                line_segments.append(("y", yv, float(np.min(x_feas)), float(np.max(x_feas))))

    else:  # both integer -> dots
        for xv in x_ints:
            j = int(np.argmin(np.abs(x_vals - xv)))
            for yv in y_ints:
                i = int(np.argmin(np.abs(y_vals - yv)))
                if mask_arr[i, j]:
                    dot_points.append((float(xv), float(yv)))

    return line_segments, dot_points


ip_line_segments, ip_dot_points = integer_overlay_from_mask(x1, x2, mask, x1_is_int, x2_is_int)


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

    if x1_is_int or x2_is_int:
        if x1_is_int and x2_is_int and ip_dot_points:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in ip_dot_points],
                y=[p[1] for p in ip_dot_points],
                z=[0.0 for _ in ip_dot_points],
                mode="markers",
                marker=dict(size=4, color="yellow"),
                name="Integer Feasible Points"
            ))
    else:
        for axis_kind, fixed, lo, hi in ip_line_segments:
            if axis_kind == "x":
                fig.add_trace(go.Scatter3d(
                    x=[fixed, fixed], y=[lo, hi], z=[0.0, 0.0],
                    mode="lines", line=dict(color="yellow", width=4),
                    name="Integer Feasible Line"
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[lo, hi], y=[fixed, fixed], z=[0.0, 0.0],
                    mode="lines", line=dict(color="yellow", width=4),
                    name="Integer Feasible Line"
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
        fig.add_trace(go.Scatter3d(
            x=np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
            y=np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
            z=np.zeros(len(hull_vertices) + 1),
            mode='lines',
            line=dict(color='darkgreen', width=5),
            name="Feasible Boundary"
        ))

    # --- Projected feasible boundary onto the objective plane ---
        # z_plane = c1 * hull_vertices[:, 0] + c2 * hull_vertices[:, 1]
        # fig.add_trace(go.Scatter3d(
        #     x=np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
        #     y=np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
        #     z=np.append(z_plane, z_plane[0]),
        #     mode='lines',
        #     line=dict(color='limegreen', width=5, dash='dot'),
        #     name="Feasible Boundary (on Plane)"
        # ))



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

    
    if res.success:
        x_opt = res.x
        z_opt = c1 * x_opt[0] + c2 * x_opt[1]

        # Optimal solution marker
        fig.add_trace(go.Scatter3d(
            x=[x_opt[0]], y=[x_opt[1]], z=[z_opt + lift],
            mode="markers+text",
            text=[f"({fmt(x_opt[0])}, {fmt(x_opt[1])}, {fmt(z_opt)})"],
            textposition="top center",
            marker=dict(size=4, color="red"),
            name="Optimal Solution"
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

            lhs_opt = a1 * x_opt[0] + a2 * x_opt[1]
            is_binding = abs(lhs_opt - b) <= 1e-3 # hard coded tolerance

            width = 20 if is_binding else 5

            fig.add_trace(go.Scatter3d(
                x=[xA, xB],
                y=[yA, yB],
                z=[zA, zB],
                mode="lines",
                line=dict(width=20, color=constraint_colors[idx % len(constraint_colors)], dash=dash),
                name=f"Constraint {idx+1}: {a1:.2f}·x₁ + {a2:.2f}·x₂ {op} {b:.2f}",
                hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
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
            # TODO: this "should" remove the mouse overlay lines, but it doesn't seem to be working
            # zaxis=dict(range=[0, np.max(Z)], showspikes=False),
            # xaxis=dict(showspikes=False),
            # yaxis=dict(showspikes=False)

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

    if x1_is_int or x2_is_int:
        if x1_is_int and x2_is_int and ip_dot_points:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in ip_dot_points],
                y=[p[1] for p in ip_dot_points],
                mode="markers",
                marker=dict(size=7, color="yellow", symbol="circle"),
                name="Integer Feasible Points"
            ))
    else:
        for axis_kind, fixed, lo, hi in ip_line_segments:
            if axis_kind == "x":
                fig.add_trace(go.Scatter(
                    x=[fixed, fixed], y=[lo, hi],
                    mode="lines", line=dict(color="yellow", width=3),
                    name="Integer Feasible Line"
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[lo, hi], y=[fixed, fixed],
                    mode="lines", line=dict(color="yellow", width=3),
                    name="Integer Feasible Line"
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


#___________________________________________________________________________

if sensitivity is not None:
    title = "Sensitivity analysis (Pyomo + GLPK)"
    if sensitivity.get("source") == "LP relaxation":
        title += " - LP relaxation"

    st.subheader(title)

    if sensitivity.get("note"):
        st.caption(sensitivity["note"])

    if sensitivity.get("reduced_costs") is not None:
        st.write("Reduced costs:", sensitivity["reduced_costs"])

    if sensitivity.get("constraints"):
        st.dataframe(sensitivity["constraints"], use_container_width=True)

