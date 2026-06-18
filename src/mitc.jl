abstract type AbstractMITC end

"""
    MITC{N,M,T}

Mixed Interpolation of Tensorial Components data for the N-node shell element (Bucalem & Bathe 1993).
Eliminates transverse shear locking by evaluating the covariant shear strains ``\\gamma_\\alpha = a_\\alpha \\cdot d`` at fixed
tying points and interpolating back to Gauss points.

Static fields (`N_tie`, `dNdξ_tie`, `h_tie`) are precomputed once at construction.
Mutable fields (`A*_tie`, `G₃_tie`, `T*_tie`) are updated each [`reinit!`](@ref) call.
"""
struct MITC{N,M,T<:AbstractFloat,M12,Mem} <: AbstractMITC
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
    # in-plane shear (E₁₂) membrane tying: 2×2 set, bilinear interpolation (Mem=true only)
    dNdξ_tie_12 :: Matrix{Vec{2,T}}  # gradients at E₁₂ tying pts  [n_shape × M12]
    h_tie_12    :: Matrix{T}         # MITC interp weights for E₁₂  [n_qp × M12]
    ξ_tie_12    :: Vector{Vec{2,T}}
    A₁_tie_12   :: Vector{Vec{3,T}}; A₂_tie_12 :: Vector{Vec{3,T}}  # ref geometry at E₁₂ tying pts
    G₃_node :: Vector{Vec{3,T}}   # per-element-local-node frame (length N)
    T₁_node :: Vector{Vec{3,T}}
    T₂_node :: Vector{Vec{3,T}}
    # Reusable scratch for bending_tangent_RM! (overwritten each call; not thread-safe).
    a₁_tie_s :: Vector{Vec{3,T}}; a₂_tie_s :: Vector{Vec{3,T}}   # length M (tying points)
    d_tie1_s :: Vector{Vec{3,T}}; d_tie2_s :: Vector{Vec{3,T}}
    dd1_s    :: Vector{Vec{3,T}}; dd2_s    :: Vector{Vec{3,T}}   # length N (nodes), Rodrigues ∂d/∂φ
    Bγ₁u_s   :: Vector{Vec{3,T}}; Bγ₂u_s   :: Vector{Vec{3,T}}   # length N
    Bγ₁φ1_s  :: Vector{T}; Bγ₁φ2_s :: Vector{T}
    Bγ₂φ1_s  :: Vector{T}; Bγ₂φ2_s :: Vector{T}
end
function MITC{N,Mem}(ip_shape::Interpolation, h_tie_1, h_tie_2, ξ_tie_1, ξ_tie_2, h_tie_12, ξ_tie_12) where {N,Mem}
    n_shape = getnbasefunctions(ip_shape)
    Nt = length(ξ_tie_1); Nt12 = length(ξ_tie_12); T = eltype(ξ_tie_1[1])
    # shape values there
    N_tie_1 = zeros(T, n_shape, Nt);  dNdξ_tie_1 = Matrix{Vec{2,T}}(undef, n_shape, Nt)
    N_tie_2 = zeros(T, n_shape, Nt);  dNdξ_tie_2 = Matrix{Vec{2,T}}(undef, n_shape, Nt)
    dNdξ_tie_12 = Matrix{Vec{2,T}}(undef, n_shape, Nt12)
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
    for (k, ξ_k) in enumerate(ξ_tie_12)
        for I in 1:n_shape
            dN, _ = Ferrite.reference_shape_gradient_and_value(ip_shape, ξ_k, I)
            dNdξ_tie_12[I, k] = dN
        end
    end
    MITC{N,Nt,T,Nt12,Mem}(
        N_tie_1, dNdξ_tie_1, N_tie_2, dNdξ_tie_2, h_tie_1, h_tie_2,
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt), fill(zero(Vec{3,T}), Nt),
        ξ_tie_1, ξ_tie_2,
        dNdξ_tie_12, h_tie_12, ξ_tie_12,
        fill(zero(Vec{3,T}), Nt12), fill(zero(Vec{3,T}), Nt12),
        fill(zero(Vec{3,T}), N), fill(zero(Vec{3,T}), N), fill(zero(Vec{3,T}), N),
        Vector{Vec{3,T}}(undef, Nt), Vector{Vec{3,T}}(undef, Nt),
        Vector{Vec{3,T}}(undef, Nt), Vector{Vec{3,T}}(undef, Nt),
        Vector{Vec{3,T}}(undef, N),  Vector{Vec{3,T}}(undef, N),
        Vector{Vec{3,T}}(undef, N),  Vector{Vec{3,T}}(undef, N),
        Vector{T}(undef, N), Vector{T}(undef, N),
        Vector{T}(undef, N), Vector{T}(undef, N),
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
    for (k, ξ_k) in enumerate(mitc.ξ_tie_12)
        A₁ = zero(Vec{3,T}); A₂ = zero(Vec{3,T})
        for i in 1:n_geo
            dN, _ = Ferrite.reference_shape_gradient_and_value(ip_geo, ξ_k, i)
            A₁ += x[i] * dN[1]; A₂ += x[i] * dN[2]
        end
        mitc.A₁_tie_12[k] = A₁; mitc.A₂_tie_12[k] = A₂
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
            cosθ, sincθ = cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
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
            cosθ, sincθ = cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
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
@inline function shear_strains(a₁, a₂, d, qp::Int, γ₁_k, γ₂_k, mitc::MITC{N,M}) where {N,M}
    γ₁ = zero(eltype(γ₁_k)); γ₂ = zero(eltype(γ₂_k))
    @inbounds for k in 1:M
        γ₁ += mitc.h_tie_1[qp, k] * γ₁_k[k]
        γ₂ += mitc.h_tie_2[qp, k] * γ₂_k[k]
    end
    γ₁, γ₂
end

# Reference (u=0) shear to subtract so the strain is measured from the reference state.
# NoMITC: QP-direct `dot(A_α, d₀)` (the raw `shear_strains` is not yet referenced).
# MITC: the tying strains already subtract their own per-tying-point reference, so the
# interpolated `shear_strains` is referenced — subtracting `dot(A_α, d₀)` again would
# double-count (zero on flat elements, but a spurious reference shear on curved ones).
@inline reference_shear_offset(A₁, A₂, d₀, ::NoMITC) = dot(A₁, d₀), dot(A₂, d₀)
@inline reference_shear_offset(A₁, A₂, d₀, ::MITC)    = 0.0, 0.0

# default: no membrane tying (NoMITC and shear-only MITC with Mem=false → classical membrane)
@inline tying_membrane_strains(::NoMITC, u_e) = nothing, nothing, nothing
@inline tying_membrane_strains(::MITC{N,M,T,M12,false}, u_e) where {N,M,T,M12} = nothing, nothing, nothing

"""
    tying_membrane_strains(mitc::MITC{N,M,T,M12,true}, u_e)

Covariant Green–Lagrange membrane strains at the MITC tying points: normal components
``E_{11}=½(a_1·a_1-A_1·A_1)`` (sampled at `ξ_tie_1`) and ``E_{22}=½(a_2·a_2-A_2·A_2)``
(at `ξ_tie_2`), plus the in-plane shear ``E_{12}=½(a_1·a_2-A_1·A_2)`` sampled at the 2×2
set `ξ_tie_12`. Interpolating these to the quadrature points (via `h_tie`) relaxes membrane
locking on doubly-curved geometry — the in-plane counterpart of the shear MITC.
Returns (`E₁₁_k`, `E₂₂_k`, `E₁₂_k`) as NTuples, ForwardDiff-safe.
"""
function tying_membrane_strains(mitc::MITC{N,M,FT,M12,true}, u_e::AbstractVector{T}) where {N,M,FT,M12,T}
    E₁₁_k = ntuple(Val(M)) do k
        a₁ = mitc.A₁_tie_1[k]
        @inbounds for I in 1:N
            a₁ += Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2])) * mitc.dNdξ_tie_1[I,k][1]
        end
        (dot(a₁, a₁) - dot(mitc.A₁_tie_1[k], mitc.A₁_tie_1[k])) / 2
    end
    E₂₂_k = ntuple(Val(M)) do k
        a₂ = mitc.A₂_tie_2[k]
        @inbounds for I in 1:N
            a₂ += Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2])) * mitc.dNdξ_tie_2[I,k][2]
        end
        (dot(a₂, a₂) - dot(mitc.A₂_tie_2[k], mitc.A₂_tie_2[k])) / 2
    end
    E₁₂_k = ntuple(Val(M12)) do k
        a₁ = mitc.A₁_tie_12[k]; a₂ = mitc.A₂_tie_12[k]
        @inbounds for I in 1:N
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            a₁ += u_I * mitc.dNdξ_tie_12[I,k][1]; a₂ += u_I * mitc.dNdξ_tie_12[I,k][2]
        end
        (dot(a₁, a₂) - dot(mitc.A₁_tie_12[k], mitc.A₂_tie_12[k])) / 2
    end
    E₁₁_k, E₂₂_k, E₁₂_k
end

# Membrane metric c_ms = (a₁·a₁, a₁·a₂, a₂·a₂) used to form the strain (c_ms-A)/2.
# NoMITC and Mem=false: direct at the QP (classical membrane). Mem=true: all three
# components interpolated from their tying points (full MITC9 membrane).
@inline membrane_metric(A, qp, a₁, a₂, ::NoMITC, E₁₁_k, E₂₂_k, E₁₂_k) =
    SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
@inline membrane_metric(A, qp, a₁, a₂, ::MITC{N,M,T,M12,false}, E₁₁_k, E₂₂_k, E₁₂_k) where {N,M,T,M12} =
    SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
@inline function membrane_metric(A, qp::Int, a₁, a₂, mitc::MITC{N,M,FT,M12,true}, E₁₁_k, E₂₂_k, E₁₂_k) where {N,M,FT,M12}
    E₁₁ = zero(eltype(E₁₁_k)); E₂₂ = zero(eltype(E₂₂_k)); E₁₂ = zero(eltype(E₁₂_k))
    @inbounds for k in 1:M
        E₁₁ += mitc.h_tie_1[qp, k] * E₁₁_k[k]
        E₂₂ += mitc.h_tie_2[qp, k] * E₂₂_k[k]
    end
    @inbounds for k in 1:M12
        E₁₂ += mitc.h_tie_12[qp, k] * E₁₂_k[k]
    end
    SymmetricTensor{2,2}((A[1,1] + 2E₁₁, A[1,2] + 2E₁₂, A[2,2] + 2E₂₂))
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
export MITC9, MITC9M