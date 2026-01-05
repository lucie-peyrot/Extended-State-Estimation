import sympy as sp

# ----------------------------
# Time symbol
# ----------------------------
t = sp.symbols('t')

# ----------------------------
# Parameters
# ----------------------------
J1, J2, D1, D2, KL, Ks, ω0 = sp.symbols('J1 J2 D1 D2 KL Ks ω0')
α1, α2, β1, β2 = sp.symbols('α1 α2 β1 β2')
Pr1, Pr2, P0, PG1, PG2 = sp.symbols('Pr1 Pr2 P0 PG1 PG2')

# ----------------------------
# States (θ1 is measured/fixed)
# ----------------------------
θ1 = sp.Function('θ1')(t)
ω1 = sp.Function('ω1')(t)
Tm1 = sp.Function('Tm1')(t)
θ2 = sp.Function('θ2')(t)
ω2 = sp.Function('ω2')(t)
Tm2 = sp.Function('Tm2')(t)
N = sp.Function('N')(t)

X = sp.Matrix([θ1, ω1, Tm1, θ2, ω2, Tm2, N])

# ----------------------------
# Control inputs
# ----------------------------
P01 = sp.Function('P01')(t)
P02 = sp.Function('P02')(t)
U = sp.Matrix([P01, P02])

# ----------------------------
# Disturbances
# ----------------------------
PL1 = sp.Function('PL1')(t)
PL2 = sp.Function('PL2')(t)
V = sp.Matrix([PL1, PL2])

# ----------------------------
# Measurements
# θ1 is measured/fixed
# ----------------------------
PGm1 = sp.Function('PGm1')(t)
PGm2 = sp.Function('PGm2')(t)

F12 = KL * (θ1 - θ2)

Y = sp.Matrix([θ1, PGm1, PGm2, F12])

# Measurement noise
W1, W2, W3, W4 = sp.symbols('W1 W2 W3 W4')
W = sp.Matrix([W1, W2, W3, W4])

# ----------------------------
# State equations (f_eqs)
# ----------------------------
ωr = (J1 * ω1 + J2 * ω2) / (J1 + J2)

f_eqs = sp.Matrix([
    # dθ1/dt
    ω1 - ω0,

    # dω1/dt
    (Tm1 - (PG1 / ω0) - D1 * (ω1 - ω0)) / J1,

    # dTm1/dt
    -α1 * (Tm1 - (N * Pr1 + P0) * ω1) - β1 * (ω1 - ω0),

    # dθ2/dt
    ω2 - ω0,

    # dω2/dt
    (Tm2 - (PG2 / ω0) - D2 * (ω2 - ω0)) / J2,

    # dTm2/dt
    -α2 * (Tm2 - (N * Pr2 + P0) * ω2) - β2 * (ω2 - ω0),

    # dN/dt
    Ks * (ωr - ω0)
])

# ----------------------------
# State-space matrices
# ----------------------------
A = f_eqs.jacobian(X)
B = f_eqs.jacobian(U)
D = f_eqs.jacobian(V)

r_sym = A.rank()

# Output matrices
C = Y.jacobian(X)
E = Y.jacobian(V)


# ----------------------------
# Display
# ----------------------------
sp.pprint(A)
sp.pprint(C)
