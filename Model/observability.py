import sympy as sp
import json
import numpy as np
from model_theta1_in_Y import A, C, X

# ----------------------------
# Load numerical values (JSON)
# ----------------------------
with open("Model/numerical_values.json", "r") as f:
    values_json = json.load(f)

# ----------------------------
# Map JSON values to the correct SymPy symbols / functions
# ----------------------------
# Get all state functions from the model
state_funcs = [f for f in X]

# Build a substitution dictionary for states
state_subs = {}
for f in state_funcs:
    name = str(f.func)  # function name, e.g., 'omega1'
    if name in values_json:
        state_subs[f] = values_json[name]  # substitute numeric value

# Build a substitution dictionary for parameters
param_subs = {sp.Symbol(k): v for k, v in values_json.items()
              if k not in state_subs.values()}

# Merge state and parameter substitutions
subs_all = {}
subs_all.update(param_subs)
subs_all.update(state_subs)

# ----------------------------
# Substitute numeric values into A and C
# ----------------------------
A_num = A.subs(subs_all).evalf()
C_num = C.subs(subs_all).evalf()

# Convert to NumPy arrays
A_np = np.array(A_num.tolist(), dtype=float)
C_np = np.array(C_num.tolist(), dtype=float)

# ----------------------------
# Build the observability matrix
# ----------------------------
n = len(X)
OO_blocks = [C_np]
A_power = A_np.copy()

for i in range(1, n):
    OO_blocks.append(C_np @ A_power)
    A_power = A_power @ A_np

OO_np = np.vstack(OO_blocks)

# ----------------------------
# Compute the numeric rank
# ----------------------------
rank = np.linalg.matrix_rank(OO_np, tol=1e-8)

# ----------------------------
# Display results
# ----------------------------
print("--- Observability Analysis ---")
print("Shape of A:", A_np.shape)
print("Shape of C:", C_np.shape)
print("Shape of Observability matrix OO:", OO_np.shape)
print("Rank of OO:", rank)
