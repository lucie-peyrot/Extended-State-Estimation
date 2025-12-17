module Observability

using LinearAlgebra
using Symbolics
using ModelingToolkit
using JSON


include("../Model/model_theta1_in_Y.jl")
using .ModelTheta1InY
A = ModelTheta1InY.A
C = ModelTheta1InY.C
X = ModelTheta1InY.X

# ----------------------------
# Observability matrix
# ----------------------------
n = length(X)
OO = [C]
for i in 1:(n-1)
    push!(OO, Symbolics.simplify.(C * (A^i)))
end
OO = vcat(OO...)  # concatenate vertically

@show typeof(A)
@show typeof(C)
@show size(A)
@show size(C)
@show OO

@parameters J1 J2 D1 D2 KL Ks ω0 ω1 ω2 Tm1 Tm2 N
@parameters α1 α2 β1 β2 Pr1 Pr2 P0 PG1 PG2 θ1 


# dictionnaire de valeurs numériques
values_json = JSON.parsefile("Model/numerical_values.json")
valeurs_numeriques = Dict(Symbol(k)=>v for (k,v) in values_json)

println("--- Substitution Numérique ---")

# 3. Substitution et conversion en matrice de nombres (Float64)
# On utilise 'substitute' de Symbolics, puis 'Symbolics.value' pour en faire des nombres
A_num = Symbolics.value.(substitute(A, valeurs_numeriques))
C_num = Symbolics.value.(substitute(C, valeurs_numeriques))
OO_num = Symbolics.value.(substitute(OO, valeurs_numeriques))


end # module