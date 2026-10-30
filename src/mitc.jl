abstract type AbstractMITC end

"""
    MITC{N,M,T}

Mixed Interpolation of Tensorial Components data for the N-node shell element (Bucalem & Bathe 1993).
Eliminates transverse shear locking by evaluating the covariant shear strains ``\\gamma_\\alpha = a_\\alpha \\cdot d`` at fixed
tying points and interpolating back to Gauss points.

The `M` tying entries are *component-tagged*: entry `k` ties the covariant component `α_tie[k]`
at `ξ_tie[k]`, and both `h_tie_1` and `h_tie_2` span all entries,
``\\gamma_\\alpha(\\xi_q) = \\sum_k h^\\alpha_{qk}\\,\\gamma_k``. Quadrilateral schemes leave the off-component
columns zero (``\\gamma_1`` is built from ``\\gamma_1`` tying values only); triangular ones do not, since
the hypotenuse condition couples the two components (Lee & Bathe 2004).

Every scheme is declared the same way: a `tying_conditions(::typeof(MITCx))` method returning
its tying conditions and the assumed-strain space they are tied against, which the shared
`MITC{N}(ip_shape, qr, scheme)` constructor feeds to [`tying_weights`](@ref).

The reference geometry at the tying points comes from `scv_tie`, a `ShellCellValues` whose
"quadrature points" are the tying points (entry `k` is point `k`, the weights are unused), so the
tying points go through exactly the same [`reinit!`](@ref) and `reference_director` code as the
quadrature points. `N_tie` and `dN_tie` are read from `scv_tie` once at construction.
"""
struct MITC{N,M,T<:AbstractFloat,SV} <: AbstractMITC
    scv_tie :: SV                 # ShellCellValues (NoMITC) on the tying points; holds A_α, the node frames and N
    N_tie   :: Matrix{T}          # shape functions           at the tying points  [n_shape × M]
    dN_tie  :: Matrix{T}          # ∂N_I/∂ξ_{α_k} of the tied component            [n_shape × M]
    h_tie_1 :: Matrix{T}          # MITC interp weights for γ₁  [n_qp × M]
    h_tie_2 :: Matrix{T}          # MITC interp weights for γ₂  [n_qp × M]
    ξ_tie   :: Vector{Vec{2,T}}   # local coordinates of the tying points
    α_tie   :: Vector{Int}        # tied covariant component (1 or 2) of each entry
end
function MITC{N}(ip_shape::Interpolation, ip_geo::Interpolation, ξ_tie::Vector{Vec{2,T}}, α_tie::Vector{Int},
                 h_tie_1::Matrix{T}, h_tie_2::Matrix{T}) where {N,T}
    n_shape = getnbasefunctions(ip_shape)
    # `N_tie` is indexed alongside the per-node frames of `scv_tie`, which are sized to
    # `ip_shape`, so the tying scheme and the shape interpolation must have the same node count.
    n_shape == N || throw(ArgumentError(
        "MITC{$N} tying needs an $N-node shape interpolation, got $ip_shape with $n_shape base functions"))
    M = length(ξ_tie)
    qr_tie  = QuadratureRule{Ferrite.getrefshape(ip_shape)}(ones(T, M), ξ_tie)
    scv_tie = ShellCellValues(qr_tie, ip_geo, ip_shape)
    # shape values and the derivative along the tied direction at each tying point
    N_tie  = copy(scv_tie.N)
    dN_tie = [scv_tie.dNdξ[I, k][α_tie[k]] for I in 1:n_shape, k in 1:M]
    MITC{N,M,T,typeof(scv_tie)}(scv_tie, N_tie, dN_tie, h_tie_1, h_tie_2, ξ_tie, α_tie)
end

# Every scheme is declared by a `tying_conditions(::typeof(MITCx))` method returning its tying
# conditions and the assumed-strain space they are tied against; this is the shared body that
# turns that pair into the element data, so each scheme file is a docstring, a one-line
# constructor and the conditions table.
function MITC{N}(ip_shape::Interpolation, qr::QuadratureRule, scheme; ip_geo::Interpolation = ip_shape) where {N}
    conds, basis = tying_conditions(scheme)
    ξ_tie, α_tie, h_tie_1, h_tie_2 = tying_weights(qr, conds, basis)
    MITC{N}(ip_shape, ip_geo, ξ_tie, α_tie, h_tie_1, h_tie_2)
end

# safe copy for multi-threaded assembly
Base.copy(m::MITC) = typeof(m)(ntuple(i -> copy(getfield(m, i)), fieldcount(typeof(m)))...)

# empty MITC is standard
struct NoMITC <: AbstractMITC end
Base.copy(m::NoMITC) = m

import Ferrite: reinit!

"""
    reinit!(mitc, x, G₃_nodes, T₁_nodes, T₂_nodes)

Update the MITC data for a cell with cell coordinates `x` and nodal frames
`G₃_nodes`/`T₁_nodes`/`T₂_nodes` (length `N`, i.e. sized to `ip_shape`, not `ip_geo`).

The reference geometry at the tying points is recomputed by `reinit_geometry!` on `scv_tie`,
and the nodal frames are copied into it, so [`tying_tangent`](@ref) and
[`tying_reference_director`](@ref) follow the same code as the quadrature points.
"""
reinit!

reinit!(::NoMITC, args...) = nothing
function reinit!(mitc::MITC{N}, x::AbstractVector{<:Vec{3}}, G₃_nodes::AbstractVector{<:Vec{3}},
                 T₁_nodes::AbstractVector{<:Vec{3}}, T₂_nodes::AbstractVector{<:Vec{3}}) where {N}
    scv = mitc.scv_tie
    reinit_geometry!(scv, x)
    for I in 1:N
        scv.G₃_elem[I] = G₃_nodes[I]
        scv.T₁_elem[I] = T₁_nodes[I]
        scv.T₂_elem[I] = T₂_nodes[I]
    end
    return mitc
end

"""
    tying_tangent(mitc, k)

Reference covariant tangent ``A_{\\alpha_k}`` at tying entry `k`, from the geometric interpolation.
"""
@inline tying_tangent(mitc::MITC, k::Int) = mitc.α_tie[k] == 1 ? mitc.scv_tie.A₁[k] : mitc.scv_tie.A₂[k]

"""
    tying_reference_director(mitc, k)

Reference director ``d_0 = \\sum_I N_I(\\xi_k) G_{3,I}`` at tying entry `k`, left un-normalized:
exactly the field `tying_shear_strains` builds at ``u = 0``. Normalizing it leaves a reference
shear ``\\gamma_\\alpha(0) = (A_\\alpha\\cdot d_0)(1 - 1/\\Vert d_0\\Vert)`` whenever the nodal frames are not
all parallel, i.e. on any curved element driven with `NodeFrames`. Same rule as `reference_director`
on the QP-direct (NoMITC) path, which it calls.
"""
@inline tying_reference_director(mitc::MITC{N}, k::Int) where {N} = reference_director(mitc.scv_tie, k, N)


# default is no tying shear strain
@inline tying_shear_strains(::NoMITC, u_e) = nothing

"""
    tying_shear_strains(mitc::MITC{N,M,T}, u_e)

Compute the covariant shear strain ``\\gamma_k = a_{\\alpha_k} \\cdot d`` of every tying entry `k`
from the current DOF vector `u_e` (5 DOFs/node: [``u_1``,``u_2``,``u_3``,``\\varphi_1``,``\\varphi_2``,``\\cdots``]).
Returns an NTuple of length `M`, ForwardDiff-safe. Each value subtracts its own reference
``A_{\\alpha_k}\\cdot G_3``, so the tying strains vanish in the reference configuration.
Call once before the quadrature-point loop and pass to `shear_strains`.
"""
function tying_shear_strains(mitc::MITC{N,M}, u_e::AbstractVector{T}) where {N,M,T} # do not put T in type params of MITC, breaks autodiff
    ntuple(Val(M)) do k
        Δa = zero(Vec{3,T}); d_k = zero(Vec{3,T})
        for I in 1:N
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa += u_I * mitc.dN_tie[I,k]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            G₃_I = mitc.scv_tie.G₃_elem[I]; T₁_I = mitc.scv_tie.T₁_elem[I]; T₂_I = mitc.scv_tie.T₂_elem[I]
            d_k += mitc.N_tie[I,k] * (cosθ*G₃_I + sincθ*(φ₁*T₁_I + φ₂*T₂_I))
        end
        A_k = tying_tangent(mitc, k)
        dot(A_k + Δa, d_k) - dot(A_k, tying_reference_director(mitc, k))
    end
end

# default shear strains
@inline shear_strains(a₁, a₂, d, ::Int, ::Nothing, ::NoMITC) = dot(a₁, d), dot(a₂, d)

"""
    shear_strains(a₁, a₂, d, qp, γ_k, mitc)

Return (`γ₁`, `γ₂`) at quadrature point `qp`.
With MITC: weighted sum of the tying-entry values from `tying_shear_strains`.
Without MITC: direct `dot(a₁, d)`, `dot(a₂, d)`.
"""
@inline function shear_strains(a₁, a₂, d, qp::Int, γ_k, mitc::MITC{N,M,T}) where {N,M,T}
    γ₁ = zero(eltype(γ_k)); γ₂ = zero(eltype(γ_k))
    @inbounds for k in 1:M
        γ₁ += mitc.h_tie_1[qp, k] * γ_k[k]
        γ₂ += mitc.h_tie_2[qp, k] * γ_k[k]
    end
    γ₁, γ₂
end

# Reference (u=0) shear to subtract so the strain is measured from the reference state.
# NoMITC: QP-direct `dot(A_α, d₀)` (the raw `shear_strains` is not yet referenced).
# MITC: the tying strains already subtract their own per-tying-point reference, so the
# interpolated `shear_strains` is referenced — subtracting `dot(A_α, d₀)` again would
# double-count. That extra term is zero on flat elements (A_α ⟂ d₀) but a spurious
# reference shear on curved ones, which pre-stresses the reference and renders the
# tangent indefinite. Dispatch to 0 for MITC.
@inline reference_shear_offset(A₁, A₂, d₀, ::NoMITC) = dot(A₁, d₀), dot(A₂, d₀)
@inline reference_shear_offset(A₁, A₂, d₀, ::MITC)    = 0.0, 0.0

"""
    tying_weights(qr, conds, basis)

Build the tying entries and interpolation weights of an assumed transverse-shear field
``\\tilde\\gamma(\\xi) = \\sum_j c_j P_j(\\xi)`` from its tying conditions (Lee & Bathe 2004, §3.2).

* `basis[j](ξ) -> Vec{2}` — the ``j``-th assumed field, as covariant components ``(\\gamma_1,\\gamma_2)``.
* `conds[i] = (ξ, w)` — condition ``w \\cdot \\tilde\\gamma(\\xi) = w \\cdot \\gamma(\\xi)``, with `w` the tied
  direction: `Ê₁`/`Ê₂` on the ``\\xi_1``/``\\xi_2`` edges, `Ê_q` on the hypotenuse.

A condition needs the displacement-based ``\\gamma_\\alpha(\\xi)`` for every component with
``w_\\alpha \\neq 0``; the union of those ``(\\xi,\\alpha)`` pairs forms the tying entries ``k = 1\\ldots M``.
With `C[i,j] = w_i ⋅ P_j(ξ_i)` and `W[i,k] = w_i[α_k]` (entry `k` located at `ξ_i`, else 0), the
coefficients are ``c = C^{-1} W \\gamma^\\text{tie}``, hence
``h_\\alpha[q,k] = \\sum_j P_j(\\xi_q)_\\alpha (C^{-1}W)_{jk}``.
"""
function tying_weights(qr::QuadratureRule, conds, basis; atol = 1e-12)
    T = eltype(qr.weights)
    length(conds) == length(basis) ||
        throw(ArgumentError("$(length(conds)) tying conditions for $(length(basis)) basis fields"))
    ξ_tie = Vec{2,T}[]; α_tie = Int[]
    for (ξ, w) in conds, α in 1:2
        abs(w[α]) ≤ atol && continue
        any(k -> α_tie[k] == α && norm(ξ_tie[k] - ξ) ≤ atol, eachindex(ξ_tie)) && continue
        push!(ξ_tie, ξ); push!(α_tie, α)
    end
    M = length(ξ_tie); n_b = length(basis)
    C = zeros(T, n_b, n_b); W = zeros(T, n_b, M)
    for (i, (ξ, w)) in enumerate(conds)
        for j in 1:n_b
            C[i,j] = w ⋅ basis[j](ξ)
        end
        for k in 1:M
            norm(ξ_tie[k] - ξ) ≤ atol && (W[i,k] = w[α_tie[k]])
        end
    end
    A = C \ W
    n_qp = length(qr.weights)
    h₁ = zeros(T, n_qp, M); h₂ = zeros(T, n_qp, M)
    for q in 1:n_qp, j in 1:n_b
        P = basis[j](qr.points[q])
        for k in 1:M
            h₁[q,k] += P[1] * A[j,k]
            h₂[q,k] += P[2] * A[j,k]
        end
    end
    return ξ_tie, α_tie, h₁, h₂
end

# Tied directions used by the schemes below: the two natural directions, and — on triangles —
# the hypotenuse of the right-angled reference triangle, γ_q = (γ₂ - γ₁)/√2 (Lee & Bathe Eq. 19).
const Ê₁  = Vec{2}((1.0, 0.0))
const Ê₂  = Vec{2}((0.0, 1.0))
const Ê_q = Vec{2}((-1.0, 1.0)) / sqrt(2)

# MITC3
include("mitc/mitc3.jl")
export MITC3

# MITC4
include("mitc/mitc4.jl")
export MITC4

# MITC6
include("mitc/mitc6.jl")
export MITC6, MITC6a

# MITC9
include("mitc/mitc9.jl")
export MITC9
