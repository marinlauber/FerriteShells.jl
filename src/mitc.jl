abstract type AbstractMITC end
"""
    MITC{N,M,T}

Mixed Interpolation of Tensorial Components data for the N-node shell element (Bucalem & Bathe 1993).
Eliminates transverse shear locking by evaluating the covariant shear strains γ_α = a_α·d at fixed
tying points and interpolating back to Gauss points.

Static fields (N_tie, dNdξ_tie, h_tie) are precomputed once at construction.
Mutable fields (A*_tie, G₃_tie, T*_tie) are updated each `reinit!` call.
"""
struct MITC{N,M,T<:AbstractFloat} <: AbstractMITC
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
struct NoMITC <: AbstractMITC end

import Ferrite: reinit!
reinit!(::NoMITC, args...) = nothing

# default is no tying shear strain
@inline tying_shear_strains(::NoMITC, u_e) = nothing, nothing

# default shear strains
@inline shear_strains(a₁, a₂, d, ::Int, ::Nothing, ::Nothing, ::NoMITC) = dot(a₁, d), dot(a₂, d)

"""
    tying_shear_strains(mitc::MITC{N,M,T}, u_e)

Compute the covariant shear strains γ₁ = a₁·d and γ₂ = a₂·d at all `M` MITC tying points
from the current DOF vector `u_e` (5 DOFs/node: [u₁,u₂,u₃,φ₁,φ₂,…]).
Returns `(γ₁_k, γ₂_k)` as two NTuples of length `M`, ForwardDiff-safe.
Call once before the quadrature-point loop and pass to `shear_strains`.
"""
function tying_shear_strains(mitc::MITC{N,M}, u_e::AbstractVector{T}) where {N,M,T}
    γ₁_k = ntuple(Val(M)) do k
        G₃_k = mitc.G₃_tie_1[k]; T₁_k = mitc.T₁_tie_1[k]; T₂_k = mitc.T₂_tie_1[k]
        Δa₁ = zero(Vec{3,T}); d_k = zero(Vec{3,T})
        for I in 1:N
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * mitc.dNdξ_tie_1[I,k][1]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            d_k += mitc.N_tie_1[I,k] * (cosθ*G₃_k + sincθ*(φ₁*T₁_k + φ₂*T₂_k))
        end
        dot(mitc.A₁_tie_1[k] + Δa₁, d_k)
    end
    γ₂_k = ntuple(Val(M)) do k
        G₃_k = mitc.G₃_tie_2[k]; T₁_k = mitc.T₁_tie_2[k]; T₂_k = mitc.T₂_tie_2[k]
        Δa₂ = zero(Vec{3,T}); d_k = zero(Vec{3,T})
        for I in 1:N
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₂ += u_I * mitc.dNdξ_tie_2[I,k][2]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            d_k += mitc.N_tie_2[I,k] * (cosθ*G₃_k + sincθ*(φ₁*T₁_k + φ₂*T₂_k))
        end
        dot(mitc.A₂_tie_2[k] + Δa₂, d_k)
    end
    γ₁_k, γ₂_k
end

"""
    shear_strains(a₁, a₂, d, qp, γ₁_k, γ₂_k, mitc)

Return `(γ₁, γ₂)` at quadrature point `qp`.
With MITC: weighted sum of tying-point values from `tying_shear_strains`.
Without MITC: direct `dot(a₁, d)`, `dot(a₂, d)`.
"""
@inline function shear_strains(a₁, a₂, d, qp::Int, γ₁_k, γ₂_k, mitc::MITC{N,M,T}) where {N,M,T}
    γ₁ = zero(eltype(γ₁_k)); γ₂ = zero(eltype(γ₂_k))
    @inbounds for k in 1:M
        γ₁ += mitc.h_tie_1[qp, k] * γ₁_k[k]
        γ₂ += mitc.h_tie_2[qp, k] * γ₂_k[k]
    end
    γ₁, γ₂
end

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