module Model
using LinearAlgebra
using Random
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
# States
# X = [θ1, ω1, Tm1, θ2, ω2, Tm2, N]
# ----------------------------
@variables θ1(t) ω1(t) Tm1(t)
@variables θ2(t) ω2(t) Tm2(t)
@variables N(t)

X = [θ1, ω1, Tm1, θ2, ω2, Tm2, N]

# ----------------------------
# Control inputs
# U = [P0_1, P0_2]
# ----------------------------
@variables P01(t) P02(t)
U = [P01, P02]

# ----------------------------
# Disturbances
# V = [PL1, PL2]
# ----------------------------
@variables PL1(t) PL2(t)
V = [PL1, PL2]

# ----------------------------
# Measurements
# Y = [PGm1, PGm2, F12]
# ----------------------------
@variables PGm1(t) PGm2(t)
F12 = KL*(θ1 - θ2)

Y = [PGm1,
    PGm2,
    F12
]
# Y = C X + E V + W
# E = [1 0;
#      0 1;
#      0 0]

# Système dynamique
ωr = (J1*ω1 + J2*ω2)/(J1+J2)
f_eqs = [

    # dθ1/dt = ω1
    ω1 - ω0,

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

# State equation matrices
A = Symbolics.jacobian(f_eqs, X)
B = Symbolics.jacobian(f_eqs, U)
D = Symbolics.jacobian(f_eqs, V)

# Output matrices
C = Symbolics.jacobian(Y, X)
E = Symbolics.jacobian(Y, V)
end # module