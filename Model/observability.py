import sympy as sp
import json
import numpy as np
from model_theta1_in_Y import A, C, X, KL

# -------------------------------------------------
# Load numerical values
# -------------------------------------------------
with open("Model/numerical_values.json", "r") as f:
    values = json.load(f)

# -------------------------------------------------
# Define numeric symbols (NO time dependence)
# -------------------------------------------------
theta1, omega1, Tm1 = sp.symbols('theta1 omega1 Tm1')
theta2, omega2, Tm2 = sp.symbols('theta2 omega2 Tm2')
N = sp.symbols('N')

P01, P02 = sp.symbols('P01 P02')

Pr1, Pr2 = sp.symbols('Pr1 Pr2')
KL_sym = sp.symbols('KL')

# -------------------------------------------------
# Build substitution dictionary
# -------------------------------------------------
subs = {}

# --- States ---
subs.update({
    sp.Function('theta1')(sp.Symbol('t')): values['theta1'],
    sp.Function('omega1')(sp.Symbol('t')): values['omega1'],
    sp.Function('Tm1')(sp.Symbol('t')): values['Tm1'],
    sp.Function('theta2')(sp.Symbol('t')): values['theta2'],
    sp.Function('omega2')(sp.Symbol('t')): values['omega2'],
    sp.Function('Tm2')(sp.Symbol('t')): values['Tm2'],
    sp.Function('N')(sp.Symbol('t')): values['N'],
})

# --- Inputs ---
subs.update({
    sp.Function('P01')(sp.Symbol('t')): values['P01'],
    sp.Function('P02')(sp.Symbol('t')): values['P02'],
})

# --- Parameters ---
for k, v in values.items():
    subs[sp.Symbol(k)] = v

# -------------------------------------------------
# Substitute numeric values into A and C
# -------------------------------------------------
A_num = A.subs(subs).evalf()
C_num = C.subs(subs).evalf()

A_np = np.array(A_num.tolist(), dtype=float)
C_np = np.array(C_num.tolist(), dtype=float)

# -------------------------------------------------
# Derived quantities (numerical substitution)
# -------------------------------------------------
F12 = KL * (values['theta1'] - values['theta2'])
Pc1 = values['P01'] + values['N'] * values['Pr1']
Pc2 = values['P02'] + values['N'] * values['Pr2']

# -------------------------------------------------
# Observability matrix
# -------------------------------------------------
n = len(X)
OO_blocks = [C_np]
A_power = A_np.copy()

for _ in range(1, n):
    OO_blocks.append(C_np @ A_power)
    A_power = A_power @ A_np

OO = np.vstack(OO_blocks)

rank = np.linalg.matrix_rank(OO, tol=1e-8)

# -------------------------------------------------
# Results
# -------------------------------------------------
print("\n--- Observability Analysis ---")
print("A shape:", A_np.shape)
print("C shape:", C_np.shape)
print("OO shape:", OO.shape)
print("Rank of OO:", rank)
