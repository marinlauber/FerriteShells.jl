
"""
    ShellCellValues{}

Stores precomputed geometry quantities for a flat triangular shell element.
Works with Vec{3} node coordinates — no manual 2D projection required.
ip_geo and ip_shape can be any Ferrite interpolation (Lagrange, etc.).

On reinit!(scv, x):
   - Builds the 3×2 surface Jacobian from the geometry interpolation
   - Extracts the orthonormal local frame (t1, t2, n) via Gram-Schmidt
   - Projects shape function gradients to the local 2D tangent plane
   - Computes the area-weighted integration weight detJdV
"""
struct ShellCellValues{QR, IPG, IPS, T <: AbstractFloat} <: AbstractCellValues
    qr          :: QR
    ip_geo      :: IPG
    ip_shape    :: IPS
    N           :: Matrix{T}           # shape values       (n_qp × n_shape)
    ∇N          :: Matrix{Vec{2, T}}   # local 2D gradients (n_qp × n_shape)
    detJdV      :: Vector{T}           # area element × weight (n_qp,)
    local_frame :: Matrix{T}           # 3×3 rotation matrix [t1 | t2 | n]
    J_loc       :: Matrix{T}           # 2×2 local Jacobian  [J1·t1 J2·t1; J1·t2 J2·t2]
    aα          :: Matrix{Vec{3, T}}
    Aα          :: Matrix{Vec{3, T}}
    aαβ         :: Matrix
    Aαβ         :: Matrix
end
export ShellCellValues

Ferrite.getdetJdV(scv::ShellCellValues, q::Int) = scv.detJdV[q]
Ferrite.getnquadpoints(scv::ShellCellValues) = getnquadpoints(scv.qr)
Ferrite.getnbasefunctions(scv::ShellCellValues) = getnbasefunctions(scv.ip_shape)

function ShellCellValues(qr::QuadratureRule, ip_geo::Interpolation, ip_shape::Interpolation)
    n_qp    = length(qr.weights)
    n_shape = getnbasefunctions(ip_shape)
    ShellCellValues(
        qr, ip_geo, ip_shape,
        zeros(n_qp, n_shape),
        fill(zero(Vec{2, Float64}), n_qp, n_shape),
        zeros(n_qp),
        zeros(3, 3),
        zeros(2, 2),
        fill(zero(Vec{3, Float64}), n_qp, n_shape),
        fill(zero(Vec{3, Float64}), n_qp, n_shape),
        fill(zero(SymmetricTensor{2,2,Float64}), n_qp, n_shape),
        fill(zero(SymmetricTensor{2,2,Float64}), n_qp, n_shape),
    )
end

reinit!(scv::ShellCellValues, cell) = reinit!(scv, getcoordinates(cell))
function reinit!(scv::ShellCellValues, x::AbstractVector{<:Vec{3}})
    n_geo   = getnbasefunctions(scv.ip_geo)
    n_shape = getnbasefunctions(scv.ip_shape)

    for q in eachindex(scv.qr.weights)
        ξ = scv.qr.points[q]

        # Surface Jacobian columns: J1 = ∂x/∂ξ₁, J2 = ∂x/∂ξ₂  (both Vec{3})
        J1 = zero(eltype(x))
        J2 = zero(eltype(x))
        for i in 1:n_geo
            dNdξ = Ferrite.reference_shape_gradient(scv.ip_geo, ξ, i)
            J1  += x[i] * dNdξ[1]
            J2  += x[i] * dNdξ[2]
        end

        # Normal vector and area element (‖J1 × J2‖ = element area scale)
        n_vec         = J1 × J2
        area          = norm(n_vec)
        scv.detJdV[q] = area * scv.qr.weights[q]

        # Orthonormal local frame via Gram-Schmidt
        t1     = J1 / norm(J1)
        n_unit = n_vec / area
        t2     = n_unit × t1

        # Store local frame and local Jacobian from the first quadrature point.
        # For linear geometry J is constant over the element, so this is exact.
        if q == 1
            scv.local_frame[:, 1] .= Tuple(t1)
            scv.local_frame[:, 2] .= Tuple(t2)
            scv.local_frame[:, 3] .= Tuple(n_unit)
            scv.J_loc[1, 1] = J1 ⋅ t1
            scv.J_loc[1, 2] = J2 ⋅ t1
            scv.J_loc[2, 1] = J1 ⋅ t2
            scv.J_loc[2, 2] = J2 ⋅ t2
        end

        # Metric tensor g = JᵀJ (2×2, symmetric)
        g11   = J1 ⋅ J1
        g12   = J1 ⋅ J2
        g22   = J2 ⋅ J2
        det_g = g11 * g22 - g12^2

        # Shape values and local 2D gradients
        for i in 1:n_shape
            scv.N[q, i] = Ferrite.reference_shape_value(scv.ip_shape, ξ, i)
            dNdξ        = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, i)

            # Contravariant gradient components (pseudoinverse: J⁺ = g⁻¹ Jᵀ)
            α1 = (g22 * dNdξ[1] - g12 * dNdξ[2]) / det_g
            α2 = (g11 * dNdξ[2] - g12 * dNdξ[1]) / det_g

            # Physical gradient (Vec{3} in the tangent plane), projected to (t1, t2)
            ∇N_global    = J1 * α1 + J2 * α2
            scv.∇N[q, i] = Vec{2}((∇N_global ⋅ t1, ∇N_global ⋅ t2))
        end
    end
    return nothing
end
