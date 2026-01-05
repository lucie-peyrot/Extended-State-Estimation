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


# dictionnaire de valeurs numériques
valeurs_numeriques = Dict(
    # --- 1. Paramètres du Réseau & Globaux ---
    KL => 3064.0,       # Raideur pour 2 lignes (MW/rad) [cite: 113]
    Ks => 0.05,         # Gain contrôle secondaire [cite: 99]
    ω0 => 100 * π,      # Fréquence nominale 50Hz (314.159...) [cite: 97]

    # --- 2. Paramètres Générateur 1 (Table 1)  ---
    J1 => 0.4,          # Inertie (Note: valeur très faible, possiblement normalisée)
    D1 => 0.04,         # Amortissement
    Pr1 => 100.0,       # Puissance de régulation (Pr)
    α1  => 100.0,        # Gain alpha (noté "C" dans le tableau)
    β1 => 2000.0,       # Gain beta

    # --- 3. Paramètres Générateur 2 (Table 1)  ---
    J2 => 0.1,          # Inertie
    D2 => 0.02,         # Amortissement
    Pr2 => 50.0,        # Puissance de régulation (Pr)
    α2 => 100.0,        # Gain alpha
    β2 => 2000.0,       # Gain beta

    # --- 4. Point de consigne de Puissance (Attention !) ---
    # Le document a deux consignes distinctes.
    # Ton code utilisait 'P0'. Idéalement, remplace P0 par P1_0 et P2_0 dans tes équations.
    # Je les définis ici :
    P0 => 0.0,          # Valeur par défaut si ton code utilise P0 additivement
    # Si tu modifies ton modèle, utilise ces valeurs pour les consignes :
    # P1_0 => 600.0, 
    # P2_0 => 400.0,

    # --- 5. États Initiaux (Point de Fonctionnement Stable) ---
    
    # Vitesses : Nominales
    ω1 => 100 * π, 
    ω2 => 100 * π,
    
    # Angles : Calculés par l'écoulement de puissance (Load Flow)
    # P_inj1 = P_G1 (600) - P_L1 (400) = +200 MW [cite: 102, 110]
    # F12 = 200 MW. Donc 200 = KL * (θ1 - θ2) => Δθ = 200 / 3064
    θ1 => 200.0 / 3064.0, # approx 0.065 rad
    θ2 => 0.0,            # Angle de référence

    # Couples Mécaniques (Tm)
    # À l'équilibre : Tm = P_Gen / ω0 [cite: 37, 40]
    # P_Gen1 = 600, P_Gen2 = 400
    Tm1 => 600.0 / (100 * π),  # approx 1.91
    Tm2 => 400.0 / (100 * π),  # approx 1.27

    # Variable N (Contrôle secondaire)
    # À l'état stable nominal, N = 0 [cite: 58]
    N => 0.0
)

println("--- Substitution Numérique ---")

# 3. Substitution et conversion en matrice de nombres (Float64)
# On utilise 'substitute' de Symbolics, puis 'Symbolics.value' pour en faire des nombres
A_num = Symbolics.value.(substitute(A, valeurs_numeriques))
B_num = Symbolics.value.(substitute(B, valeurs_numeriques))
C_num = Symbolics.value.(substitute(C, valeurs_numeriques))
D_num = Symbolics.value.(substitute(D, valeurs_numeriques)) 
E_num = Symbolics.value.(substitute(E, valeurs_numeriques))
OO_num = Symbolics.value.(substitute(OO, valeurs_numeriques))

@show rank(OO_num)
end # module