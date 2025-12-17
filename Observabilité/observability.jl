module Observability
using LinearAlgebra
using Random
using Symbolics
using ModelingToolkit
export A, C, OO

@parameters t

# ----------------------------
# Parameters
# ----------------------------
@parameters J1 J2 D1 D2 KL Ks ω0
@parameters α1 α2 β1 β2 Pr1 Pr2

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

f_eqs = [

    # dθ1/dt = ω1
    ω1,

    # dω1/dt
    (-KL/(J1*ω0))*θ1-(D1/J1)*ω1+ (1/J1)*Tm1+ (KL/(J1*ω0))*θ2,

    # dTm1/dt
    -β1*ω1 -α1*ω0*Tm1-α1*Pr1*N,

    # dθ2/dt = ω2
    ω2,

    # dω2/dt
    (KL/(J2*ω0))*θ1-(KL/(J2*ω0))*θ2-(D2/J2)*ω2+ (1/J2)*Tm2,

    # dTm2/dt
    -β2*ω2-α2*ω0*Tm2-α2*Pr2*N,

    # dN/dt
    -Ks*(J1/(J1+J2))*ω1-Ks*(J2/(J1+J2))*ω2
]

# State equation matrices
A = Symbolics.jacobian(f_eqs, X)
B = Symbolics.jacobian(f_eqs, U)
D = Symbolics.jacobian(f_eqs, V)

# Output matrices
C = Symbolics.jacobian(Y, X)
E = Symbolics.jacobian(Y, V)

@show typeof(A)
@show typeof(C)
@show size(A)
@show size(C)
# ----------------------------
# Observability matrix
# ----------------------------
# OO = [C; C*A; C*A^2; ...; C*A^(n-1)] pour n = taille de l'état
n = length(X)   # nombre d'états
OO = [C]       # initialisation

for i in 1:(n-1)
    push!(OO, Symbolics.simplify.(C * (A^i))) 
end

OO = vcat(OO...)  # concatène verticalement

@show OO
end # module