module Model

using LinearAlgebra
using Symbolics
using ModelingToolkit

export A, B, C, D, E, W

@parameters t

# ----------------------------
# Parameters
# ----------------------------
@parameters J1 J2 D1 D2 KL Ks ω0 
@parameters α1 α2 β1 β2 Pr1 Pr2 P0 PG1 PG2

# ----------------------------
# States (θ1 is measured/fixed)
# ----------------------------
@variables ω1(t) Tm1(t)
@variables θ2(t) ω2(t) Tm2(t)
@variables N(t)

X = [ω1, Tm1, θ2, ω2, Tm2, N]

# ----------------------------
# Control inputs
# ----------------------------
@variables P01(t) P02(t)
U = [P01, P02]

# ----------------------------
# Disturbances
# ----------------------------
@variables PL1(t) PL2(t)
V = [PL1, PL2]

# ----------------------------
# Measurements
# θ1 is measured/fixed
# ----------------------------
@variables θ1(t) PGm1(t) PGm2(t)
F12 = KL*(θ1 - θ2)

Y = [θ1, PGm1, PGm2, F12]

# Measurement noise
@variables W1 W2 W3 W4
W = [W1, W2, W3, W4]

# ----------------------------
# State equations (f_eqs)
# ----------------------------
# Système dynamique
ωr = (J1*ω1 + J2*ω2)/(J1+J2)
f_eqs = [
    # dω1/dt
    (Tm1 - (PG1/ω0) - D1*(ω1 - ω0))/J1,

    # dTm1/dt
    -α1*(Tm1 - (N*Pr1+P0)*ω1) - β1*(ω1 - ω0),

    # dθ2/dt = ω2
    ω2 - ω0,

    # dω2/dt
    (Tm2 - (PG2/ω0) - D2*(ω2 - ω0))/J2,

    # dTm2/dt
    -α2*(Tm2 - (N*Pr2+P0)*ω2) - β2*(ω2 - ω0),

    # dN/dt
    Ks*(ωr - ω0)
]

# ----------------------------
# State-space matrices
# ----------------------------
A = Symbolics.jacobian(f_eqs, X)
B = Symbolics.jacobian(f_eqs, U)
D = Symbolics.jacobian(f_eqs, V)

# Output matrices
C = Symbolics.jacobian(Y, X)
E = Symbolics.jacobian(Y, V)

# Set C row for θ1 (measured/fixed) to zeros
C[1,:] .= 0
E[1,:] .= 0  # θ1 not influenced by disturbances

end # module