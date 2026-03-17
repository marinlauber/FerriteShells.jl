abstract type AbstractMITC end
"""
    MITC{N, T}

Mixed Interpolation of Tensorial Components data for the N-node shell element (Bucalem & Bathe 1993).
Eliminates transverse shear locking by evaluating the covariant shear strains γ_α = a_α·d at fixed
tying points and interpolating back to Gauss points.

Static fields (N_tie, dNdξ_tie, h_tie) are precomputed once at construction.
Mutable fields (A*_tie, G₃_tie, T*_tie) are updated each `reinit!` call.
"""
struct MITC{N,T<:AbstractFloat} <: AbstractMITC
    N_tie_1    :: Matrix{T}          # shape functions at γ₁ tying pts  [n_shape × 6]
    dNdξ_tie_1 :: Matrix{Vec{2,T}}   # gradients       at γ₁ tying pts  [n_shape × 6]
    N_tie_2    :: Matrix{T}          # shape functions at γ₂ tying pts  [n_shape × 6]
    dNdξ_tie_2 :: Matrix{Vec{2,T}}   # gradients       at γ₂ tying pts  [n_shape × 6]
    h_tie_1    :: Matrix{T}          # MITC interp weights for γ₁  [n_qp × 6]
    h_tie_2    :: Matrix{T}          # MITC interp weights for γ₂  [n_qp × 6]
    A₁_tie_1 :: Vector{Vec{3,T}}; A₂_tie_1 :: Vector{Vec{3,T}}  # ref geometry at γ₁ tying pts
    G₃_tie_1 :: Vector{Vec{3,T}}; T₁_tie_1 :: Vector{Vec{3,T}}; T₂_tie_1 :: Vector{Vec{3,T}}
    A₁_tie_2 :: Vector{Vec{3,T}}; A₂_tie_2 :: Vector{Vec{3,T}}  # ref geometry at γ₂ tying pts
    G₃_tie_2 :: Vector{Vec{3,T}}; T₁_tie_2 :: Vector{Vec{3,T}}; T₂_tie_2 :: Vector{Vec{3,T}}
end

# empty MITC is standard
struct NoMITC <:AbstractMITC end

import Ferrite: reinit!
reinit!(a::AbstractMITC, args...) = nothing

# MITC3
# include("mitc/mitc3.jl")
# export MITC3

# MITC4
# include("mitc/mitc4.jl")
# export MITC4

# MITC6
# include("mitc/mitc6.jl")
# export MITC6

# MITC9
include("mitc/mitc9.jl")
export MITC9