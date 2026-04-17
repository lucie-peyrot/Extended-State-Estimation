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
Pr1, Pr2, P01, P02 = sp.symbols('Pr1 Pr2 P01 P02')

# ----------------------------
# States
# ----------------------------
theta1, omega1, Tm1, theta2, omega2, Tm2, N = sp.symbols(
    'theta1 omega1 Tm1 theta2 omega2 Tm2 N'
)

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
PL1, PL2 = sp.symbols('PL1 PL2')

PL1 = sp.Function('PL1')(t)
PL2 = sp.Function('PL2')(t)
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
omega_r = (J1 * omega1 + J2 * omega2) / (J1 + J2)

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

# ----------------------------
PGm1 = sp.Function('PGm1')(t)
PGm2 = sp.Function('PGm2')(t)

F12 = KL * (theta1 - theta2)
PG1 = PL1 + F12
PG2 = PL2 - F12

Y = sp.Matrix([PG1, PG2, F12])

# Measurement noise
W1, W2, W3, W4 = sp.symbols('W1 W2 W3 W4')
W = sp.Matrix([W1, W2, W3, W4])

# ----------------------------
# State equations (f_eqs)
# ----------------------------
omega_r = (J1 * omega1 + J2 * omega2) / (J1 + J2)  # nominal freq used

f_eqs = sp.Matrix([

    # dθ1/dt
    omega1 - omega0,

   # dω1/dt
    (Tm1 - (PG1/omega0) - D1*(omega1 - omega0))/J1,

    # dTm1/dt
    -alpha1*(Tm1*omega0 - (N*Pr1+P01)*omega1) - beta1*(omega1 - omega0),

    # dθ2/dt = ω2
    omega2 - omega0,

    # dω2/dt
    (Tm2 - (PG2/omega0) - D2*(omega2 - omega0))/J2,

    # dTm2/dt
    -alpha2*(Tm2*omega0 - (N*Pr2+P02)*omega2) - beta2*(omega2 - omega0),

    # dN/dt
    -Ks*(omega_r - omega0)

])

# Linearized state equations 
# omega1/omega2 approximated around omega0 when they are multiplied or divided by other state variables
f_eqs_linear = sp.Matrix([

    # dθ1/dt
    omega1 - omega0,

   # dω1/dt
    (Tm1 - (PG1/omega0) - D1*(omega1 - omega0))/J1,

    # dTm1/dt
    -alpha1*(Tm1*omega0 - (N*Pr1+P0)*omega0) - beta1*(omega1 - omega0),

    # dθ2/dt = ω2
    omega2 - omega0,

    # dω2/dt
    (Tm2 - (PG2/omega0) - D2*(omega2 - omega0))/J2,

    # dTm2/dt
    -alpha2*(Tm2*omega0 - (N*Pr2+P02)*omega0) - beta2*(omega2 - omega0),

    # dN/dt
    -Ks*(omega_r - omega0)

])


# ----------------------------
# State-space matrices
# ----------------------------
A = f_eqs_linear.jacobian(X)
B = f_eqs_linear.jacobian(U)
D = f_eqs_linear.jacobian(V)

r_sym = A.rank()

# Output matrices
C = Y.jacobian(X)
E = Y.jacobian(V)

# ----------------------------
# Display
# ----------------------------
sp.pprint(A)
sp.pprint(C)


import sympy as sp
import json
import numpy as np

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


## Calcul du rang en symbolique en simplifaint les ratios
#a = -KL/(J1*ω0)
#a = -KL/(J1*ω0)
#b = -D1/J1
#c = 1/J1
#d =  KL/(J1*ω0)

#e = -β1
#f = -α1*ω0
#g =  Pr1*α1*ω0

#h =  KL/(J2*ω0)
#i = -KL/(J2*ω0)
#j = -D2/J2
#k = 1/J2

#l = -β2
#m = -α2*ω0
#n =  Pr2*α2*ω0

#p = -(J1*Ks)/(J1+J2)
#q = -(J2*Ks)/(J1+J2)

a,b,c,d,e,f,g,h,i,j,k,l,m,n,p,q = sp.symbols(
    'a b c d e f g h i j k l m n p q'
)
A_symb = sp.Matrix([
    [0, 1, 0, 0, 0, 0, 0],
    [a, b, c, d, 0, 0, 0],
    [0, e, f, 0, 0, 0, g],
    [0, 0, 0, 0, 1, 0, 0],
    [h, 0, 0, i, j, k, 0],
    [0, 0, 0, 0, l, m, n],
    [0, p, 0, 0, q, 0, 0]
])
sp.pprint(A_symb)
sp.pprint(A_symb.rank())

# Observability matrix O = [ C; C A; C A²; ... ; C Aⁿ⁻¹ ]
n = A_symb.rows

OO_symb = sp.Matrix.vstack(
    *[C * (A_symb**k) for k in range(n)]
)

sp.pprint(OO_symb.rank())
