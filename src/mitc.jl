abstract type AbstractMITC end

"""
    MITC{N,M,T}

Mixed Interpolation of Tensorial Components data for the N-node shell element (Bucalem & Bathe 1993).
Eliminates transverse shear locking by evaluating the covariant shear strains ``\\gamma_\\alpha = a_\\alpha \\cdot d`` at fixed
tying points and interpolating back to Gauss points.

Static fields (`N_tie`, `dNdξ_tie`, `h_tie`) are precomputed once at construction.
Mutable fields (`A*_tie`, `G₃_tie`, `T*_tie`) are updated each [`reinit!`](@ref) call.
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
    ξ_tie_1::Vector{Vec{2,T}};  ξ_tie_2::Vector{Vec{2,T}} # local coorindates of the tying points
    G₃_node :: Vector{Vec{3,T}}   # per-element-local-node frame (length N)
    T₁_node :: Vector{Vec{3,T}}
    T₂_node :: Vector{Vec{3,T}}
end
function MITC{N}(ip_shape::Interpolation, h_tie_1, h_tie_2, ξ_tie_1, ξ_tie_2) where N
    n_shape = getnbasefunctions(ip_shape)
    Nt = length(ξ_tie_1); T = eltype(ξ_tie_1[1])
    # shape values there
    N_tie_1 = zeros(T, n_shape, Nt);  dNdξ_tie_1 = Matrix{Vec{2,T}}(undef, n_shape, Nt)
    N_tie_2 = zeros(T, n_shape, Nt);  dNdξ_tie_2 = Matrix{Vec{2,T}}(undef, n_shape, Nt)
    for (k, ξ_k) in enumerate(ξ_tie_1)
        for I in 1:n_shape
            dN, Nval = Ferrite.reference_shape_gradient_and_value(ip_shape, ξ_k, I)
            N_tie_1[I, k] = Nval;  dNdξ_tie_1[I, k] = dN
        end
    end
    for (k, ξ_k) in enumerate(ξ_tie_2)
        for I in 1:n_shape
            dN, Nval = Ferrite.reference_shape_gradient_and_value(ip_shape, ξ_k, I)
            N_tie_2[I, k] = Nval;  dNdξ_tie_2[I, k] = dN
        end
    end
    MITC{N,Nt,T}(
        N_tie_1, dNdξ_tie_1, N_tie_2, dNdξ_tie_2, h_tie_1, h_tie_2,
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        ξ_tie_1, ξ_tie_2,
        fill(zero(Vec{3,T}), N), fill(zero(Vec{3,T}), N), fill(zero(Vec{3,T}), N),
    )
end

# empty MITC is standard
struct NoMITC <: AbstractMITC end

import Ferrite: reinit!

"""
    reinit!(mitc, ip_geo, x)

Update the MITC data for a cell with cell coordinates `x`.
The reference geometry at the tying points is recomputed and stored.
"""
reinit!

reinit!(::NoMITC, args...) = nothing
function reinit!(mitc::MITC{N,M,T}, ip_geo::Interpolation, x::AbstractVector{<:Vec{3}},
                 G₃_nodes::AbstractVector{<:Vec{3}}, T₁_nodes::AbstractVector{<:Vec{3}}, T₂_nodes::AbstractVector{<:Vec{3}}) where {N,M,T}
    n_geo = getnbasefunctions(ip_geo)
    for I in 1:N
        mitc.G₃_node[I] = G₃_nodes[I]
        mitc.T₁_node[I] = T₁_nodes[I]
        mitc.T₂_node[I] = T₂_nodes[I]
    end
    for (k, ξ_k) in enumerate(mitc.ξ_tie_1)
        A₁ = zero(Vec{3,T}); A₂ = zero(Vec{3,T}); G₃_avg = zero(Vec{3,T})
        for i in 1:n_geo
            dN, _ = Ferrite.reference_shape_gradient_and_value(ip_geo, ξ_k, i)
            A₁ += x[i] * dN[1]; A₂ += x[i] * dN[2]
            G₃_avg += Ferrite.reference_shape_value(ip_geo, ξ_k, i) * G₃_nodes[i]
        end
        G₃_k = G₃_avg / norm(G₃_avg)
        ref = abs(G₃_k[1]) < T(0.9) ? Vec{3,T}((1.,0.,0.)) : Vec{3,T}((0.,1.,0.))
        t₁ = ref - (ref ⋅ G₃_k) * G₃_k; T₁_k = t₁ / norm(t₁); T₂_k = G₃_k × T₁_k
        mitc.A₁_tie_1[k] = A₁; mitc.A₂_tie_1[k] = A₂
        mitc.G₃_tie_1[k] = G₃_k; mitc.T₁_tie_1[k] = T₁_k; mitc.T₂_tie_1[k] = T₂_k
    end
    for (k, ξ_k) in enumerate(mitc.ξ_tie_2)
        A₁ = zero(Vec{3,T}); A₂ = zero(Vec{3,T}); G₃_avg = zero(Vec{3,T})
        for i in 1:n_geo
            dN, _ = Ferrite.reference_shape_gradient_and_value(ip_geo, ξ_k, i)
            A₁ += x[i] * dN[1]; A₂ += x[i] * dN[2]
            G₃_avg += Ferrite.reference_shape_value(ip_geo, ξ_k, i) * G₃_nodes[i]
        end
        G₃_k = G₃_avg / norm(G₃_avg)
        ref = abs(G₃_k[1]) < T(0.9) ? Vec{3,T}((1.,0.,0.)) : Vec{3,T}((0.,1.,0.))
        t₁ = ref - (ref ⋅ G₃_k) * G₃_k; T₁_k = t₁ / norm(t₁); T₂_k = G₃_k × T₁_k
        mitc.A₁_tie_2[k] = A₁; mitc.A₂_tie_2[k] = A₂
        mitc.G₃_tie_2[k] = G₃_k; mitc.T₁_tie_2[k] = T₁_k; mitc.T₂_tie_2[k] = T₂_k
    end
end


# default is no tying shear strain
@inline tying_shear_strains(::NoMITC, u_e) = nothing, nothing

"""
    tying_shear_strains(mitc::MITC{N,M,T}, u_e)

Compute the covariant shear strains ``\\gamma_1 = a_1 \\cdot d`` and ``\\gamma_2 = a_2 \\cdot d`` at all `M` MITC tying points
from the current DOF vector `u_e` (5 DOFs/node: [``u_1``,``u_2``,``u_3``,``\\varphi_1``,``\\varphi_2``,``\\cdots``]).
Returns (`γ₁_k`, `γ₂_k`) as two NTuples of length `M`, ForwardDiff-safe.
Call once before the quadrature-point loop and pass to `shear_strains`.
"""
function tying_shear_strains(mitc::MITC{N,M}, u_e::AbstractVector{T}) where {N,M,T} # do not put T in type params of MITC, breaks autodiff
    γ₁_k = ntuple(Val(M)) do k
        Δa₁ = zero(Vec{3,T}); d_k = zero(Vec{3,T})
        for I in 1:N
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * mitc.dNdξ_tie_1[I,k][1]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            G₃_I = mitc.G₃_node[I]; T₁_I = mitc.T₁_node[I]; T₂_I = mitc.T₂_node[I]
            d_k += mitc.N_tie_1[I,k] * (cosθ*G₃_I + sincθ*(φ₁*T₁_I + φ₂*T₂_I))
        end
        dot(mitc.A₁_tie_1[k] + Δa₁, d_k) - dot(mitc.A₁_tie_1[k], mitc.G₃_tie_1[k])
    end
    γ₂_k = ntuple(Val(M)) do k
        Δa₂ = zero(Vec{3,T}); d_k = zero(Vec{3,T})
        for I in 1:N
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₂ += u_I * mitc.dNdξ_tie_2[I,k][2]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            G₃_I = mitc.G₃_node[I]; T₁_I = mitc.T₁_node[I]; T₂_I = mitc.T₂_node[I]
            d_k += mitc.N_tie_2[I,k] * (cosθ*G₃_I + sincθ*(φ₁*T₁_I + φ₂*T₂_I))
        end
        dot(mitc.A₂_tie_2[k] + Δa₂, d_k) - dot(mitc.A₂_tie_2[k], mitc.G₃_tie_2[k])
    end
    γ₁_k, γ₂_k
end

# default shear strains
@inline shear_strains(a₁, a₂, d, ::Int, ::Nothing, ::Nothing, ::NoMITC) = dot(a₁, d), dot(a₂, d)

"""
    shear_strains(a₁, a₂, d, qp, γ₁_k, γ₂_k, mitc)

Return (`γ₁`, `γ₂`) at quadrature point `qp`.
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
include("mitc/mitc4.jl")
export MITC4

# MITC6
# include("mitc/mitc6.jl")
# export MITC6

# MITC9
include("mitc/mitc9.jl")
export MITC9