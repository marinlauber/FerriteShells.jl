
"""
    kinematics(scv, qp, x, u_e)

Compute the current kinematics at a quadrature point given reference geometry and
nodal displacements. `u_e` is a flat vector of length `3 * n_nodes` (u₁, v₁, w₁, …).
Returns `(a₁, a₂, A_metric, a_metric)` — current basis vectors and reference/current
metric tensors A_{αβ} = A_α · A_β, a_{αβ} = a_α · a_β.
"""
function kinematics(scv, qp, x, u_e::AbstractVector{T}) where T
    ξ = scv.qr.points[qp]
    A₁ = zero(Vec{3,Float64}); A₂ = zero(Vec{3,Float64})
    a₁ = zero(Vec{3,T});       a₂ = zero(Vec{3,T})
    for i in 1:getnbasefunctions(scv.ip_geo)
        dNdξ = Ferrite.reference_shape_gradient(scv.ip_geo, ξ, i)
        ui   = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
        A₁  += x[i] * dNdξ[1]
        A₂  += x[i] * dNdξ[2]
        a₁  += (x[i] + ui) * dNdξ[1]
        a₂  += (x[i] + ui) * dNdξ[2]
    end
    A_metric = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
    a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
    return (a₁, a₂, A_metric, a_metric)
end
