
"""
    ShellCellValues{}

Stores precomputed shape function data and reference geometry for a shell element.
Works with Vec{3} node coordinates — no manual 2D projection required.

Shape function data (N, dNdξ, d2Ndξ2) is computed once at construction time
from ip_shape and qr.

On reinit!(scv, x):
   - Computes reference tangent vectors A₁, A₂ and second derivatives A₁₁, A₁₂, A₂₂
   - Computes reference unit normal G₃ and local frame T₁, T₂ (Gram–Schmidt)
   - Computes reference metric A_metric and curvature tensor B
   - Computes the area-weighted integration weight detJdV
"""
struct ShellCellValues{QR, IPG, IPS, T <: AbstractFloat} <: AbstractCellValues
    qr       :: QR
    ip_geo   :: IPG
    ip_shape :: IPS
    N        :: Matrix{T}
    dNdξ     :: Matrix{Vec{2, T}}
    d2Ndξ2   :: Matrix{SymmetricTensor{2, 2, T, 3}}
    detJdV   :: Vector{T}
    A₁       :: Vector{Vec{3, T}}
    A₂       :: Vector{Vec{3, T}}
    A₁₁      :: Vector{Vec{3, T}}
    A₁₂      :: Vector{Vec{3, T}}
    A₂₂      :: Vector{Vec{3, T}}
    A_metric :: Vector{SymmetricTensor{2, 2, T, 3}}
    G₃       :: Vector{Vec{3, T}}
    T₁       :: Vector{Vec{3, T}}
    T₂       :: Vector{Vec{3, T}}
    B        :: Vector{SymmetricTensor{2, 2, T, 3}}
end
export ShellCellValues

Ferrite.getdetJdV(scv::ShellCellValues, q::Int) = scv.detJdV[q]
Ferrite.getnquadpoints(scv::ShellCellValues) = getnquadpoints(scv.qr)
Ferrite.getnbasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_shape)
@propagate_inbounds Ferrite.getngeobasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_geo)

function ShellCellValues(qr::QuadratureRule, ip_geo::Interpolation, ip_shape::Interpolation)
    n_qp    = length(qr.weights)
    n_shape = getnbasefunctions(ip_shape)
    T       = Float64

    N      = zeros(T, n_shape, n_qp)
    dNdξ   = Matrix{Vec{2, T}}(undef, n_shape, n_qp)
    d2Ndξ2 = Matrix{SymmetricTensor{2, 2, T, 3}}(undef, n_shape, n_qp)
    for q in 1:n_qp
        ξ = qr.points[q]
        for I in 1:n_shape
            d2N, dN, Nval = Ferrite.reference_shape_hessian_gradient_and_value(ip_shape, ξ, I)
            N[I, q]      = Nval
            dNdξ[I, q]   = dN
            d2Ndξ2[I, q] = SymmetricTensor{2, 2, T}((d2N[1,1], d2N[1,2], d2N[2,2]))
        end
    end

    ShellCellValues(
        qr, ip_geo, ip_shape,
        N, dNdξ, d2Ndξ2,
        zeros(T, n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(Vec{3, T}), n_qp),
        fill(zero(SymmetricTensor{2, 2, T, 3}), n_qp),
    )
end

reinit!(scv::ShellCellValues, cell) = reinit!(scv, getcoordinates(cell))
function reinit!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}})
    n_geo = getnbasefunctions(scv.ip_geo)
    for q in eachindex(scv.qr.weights)
        ξ = scv.qr.points[q]
        A₁  = zero(Vec{3,Float64}); A₂  = zero(Vec{3,Float64})
        A₁₁ = zero(Vec{3,Float64}); A₁₂ = zero(Vec{3,Float64}); A₂₂ = zero(Vec{3,Float64})
        for i in 1:n_geo
            d2N, dN, _ = Ferrite.reference_shape_hessian_gradient_and_value(scv.ip_geo, ξ, i)
            A₁  += x[i] * dN[1];     A₂  += x[i] * dN[2]
            A₁₁ += x[i] * d2N[1,1]; A₁₂ += x[i] * d2N[1,2]; A₂₂ += x[i] * d2N[2,2]
        end
        n_vec = A₁ × A₂
        area  = norm(n_vec)
        G₃    = n_vec / area
        T₁    = A₁ / norm(A₁)
        T₂    = (G₃ × T₁) / norm(G₃ × T₁)

        scv.detJdV[q]   = area * scv.qr.weights[q]
        scv.A₁[q]       = A₁;  scv.A₂[q]  = A₂
        scv.A₁₁[q]      = A₁₁; scv.A₁₂[q] = A₁₂; scv.A₂₂[q] = A₂₂
        scv.A_metric[q]  = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
        scv.G₃[q]        = G₃;  scv.T₁[q]  = T₁;  scv.T₂[q]  = T₂
        scv.B[q]         = SymmetricTensor{2,2,Float64}((dot(A₁₁,G₃), dot(A₁₂,G₃), dot(A₂₂,G₃)))
    end
    return nothing
end
