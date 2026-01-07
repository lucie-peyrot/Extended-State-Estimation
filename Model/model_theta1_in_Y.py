import sympy as sp

# ----------------------------
# Time symbol
# ----------------------------
t = sp.symbols('t')

# ----------------------------
# Parameters
# ----------------------------
J1, J2, D1, D2, KL, Ks, omega0 = sp.symbols('J1 J2 D1 D2 KL Ks omega0')
alpha1, alpha2, beta1, beta2 = sp.symbols('alpha1 alpha2 beta1 beta2')
Pr1, Pr2, P0, PG1, PG2 = sp.symbols('Pr1 Pr2 P0 PG1 PG2')

# ----------------------------
# States (theta1 is measured/fixed)
# ----------------------------
theta1 = sp.Function('theta1')(t)
omega1 = sp.Function('omega1')(t)
Tm1 = sp.Function('Tm1')(t)
theta2 = sp.Function('theta2')(t)
omega2 = sp.Function('omega2')(t)
Tm2 = sp.Function('Tm2')(t)
N = sp.Function('N')(t)

X = sp.Matrix([theta1, omega1, Tm1, theta2, omega2, Tm2, N])

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
# theta1 is measured/fixed
# ----------------------------
PGm1 = sp.Function('PGm1')(t)
PGm2 = sp.Function('PGm2')(t)

F12 = KL * (theta1 - theta2)

Y = sp.Matrix([theta1, PGm1, PGm2, F12])

# Measurement noise
W1, W2, W3, W4 = sp.symbols('W1 W2 W3 W4')
W = sp.Matrix([W1, W2, W3, W4])

# ----------------------------
# State equations (f_eqs)
# ----------------------------
omega_r = (J1 * omega0 + J2 * omega0) / (J1 + J2)  # nominal freq used

f_eqs = sp.Matrix([

    # dθ1/dt
    omega1 - omega0,

   # dω1/dt
    (Tm1 - (PG1/omega0) - D1*(omega1 - omega0))/J1,

    # dTm1/dt
    -alpha1*(Tm1 - (N*Pr1+P0)*omega1) - beta1*(omega1 - omega0),

    # dθ2/dt = ω2
    omega2 - omega0,

    # dω2/dt
    (Tm2 - (PG2/omega0) - D2*(omega2 - omega0))/J2,

    # dTm2/dt
    -alpha2*(Tm2 - (N*Pr2+P0)*omega2) - beta2*(omega2 - omega0),

    # dN/dt
    Ks*(omega_r - omega0)

])


# ----------------------------
# State-space matrices
# ----------------------------
A = sp.Matrix([
    [0, 1, 0, 0, 0, 0, 0],

    [0, -D1/J1,  1/J1,  KL/(J1*omega0), 0, 0, 0],

    [0, -beta1, -alpha1*omega0, 0, 0, 0, -alpha1*Pr1],

    [0, 0, 0, 0, 1, 0, 0],

    [0, KL/(J2*omega0), 0, -KL/(J2*omega0), -D2/J2, 1/J2, 0],

    [0, 0, 0, 0, -beta2, -alpha2*omega0, -alpha2*Pr2],

    [0, -Ks*J1/(J1+J2), 0, -Ks*J2/(J1+J2), 0, 0, 0]
])
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
