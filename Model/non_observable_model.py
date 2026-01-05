import sympy as sp

# ----------------------------
# Parameters
# ----------------------------
J1, J2, D1, D2, KL, Ks, omega0 = sp.symbols('J1 J2 D1 D2 KL Ks omega0')
alpha1, alpha2, beta1, beta2 = sp.symbols('alpha1 alpha2 beta1 beta2')
Pr1, Pr2, P01, P02 = sp.symbols('Pr1 Pr2 P01 P02')

# ----------------------------
# States
# ----------------------------
theta1, omega1, Tm1, theta2, omega2, Tm2, N = sp.symbols(
    'theta1 omega1 Tm1 theta2 omega2 Tm2 N'
)
X = sp.Matrix([theta1, omega1, Tm1, theta2, omega2, Tm2, N])

# ----------------------------
# Control inputs
# ----------------------------
U = sp.Matrix([P01, P02])

# ----------------------------
# Disturbances
# ----------------------------
PL1, PL2 = sp.symbols('PL1 PL2')
V = sp.Matrix([PL1, PL2])

# ----------------------------
# Measurements
# ----------------------------
PGm1, PGm2, Fm12 = sp.symbols('PGm1 PGm2 Fm12')
Y = sp.Matrix([PGm1, PGm2, Fm12])

# ----------------------------
# Linearized state equations (omega1 and omega2 replaced by omega0 in products/divisions)
# ----------------------------

# Power flows
F12 = KL * (theta1 - theta2)
Pc1 = P01 + N * Pr1
Pc2 = P02 + N * Pr2
omega_r = (J1 * omega0 + J2 * omega0) / (J1 + J2)  # nominal freq used

# State derivatives
f_eqs = sp.Matrix([
    # dtheta1/dt
    omega1 - omega0,
    
    # domega1/dt
    (Tm1 - PL1 - F12) / omega0 - D1 * (omega1 - omega0) / J1,
    
    # dTm1/dt
    -alpha1 * (Tm1 * omega0 - Pc1) - beta1 * (omega1 - omega0),
    
    # dtheta2/dt
    omega2 - omega0,
    
    # domega2/dt
    (Tm2 - PL2 + F12) / omega0 - D2 * (omega2 - omega0) / J2,
    
    # dTm2/dt
    -alpha2 * (Tm2 * omega0 - Pc2) - beta2 * (omega2 - omega0),
    
    # dN/dt
    -Ks * (omega_r - omega0)
])

# ----------------------------
# State-space matrices
# ----------------------------
A = f_eqs.jacobian(X)
B = f_eqs.jacobian(U)
D = f_eqs.jacobian(V)
C = sp.Matrix([
    [KL, 0, 0, -KL, 0, 0, 0],  # F12 measured
    [-KL, 0, 0, KL, 0, 0, 0],  # PGm1-PGm2
    [KL, 0, 0, -KL, 0, 0, 0]   # same as F12
])
E = sp.zeros(3, 2)  # if disturbances are known exactly

# ----------------------------
# Display matrices
# ----------------------------
sp.pprint(A)
sp.pprint(C)
