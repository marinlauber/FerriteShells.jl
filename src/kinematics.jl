
"""
    kinematics(scv, qp, x, u_e)

Compute the current kinematics at a quadrature point, given the reference geometry and current displacements.
Returns the current basis vectors a₁, a₂, and the reference and current metric tensors
A_{αβ} = A_α · A_β and a_{αβ} = a_α · a_β.
"""
function kinematics(scv, qp, x, u_e::AbstractVector{<:Vec{3,T}}; director=false) where T
    # construct current geometry for this quadrature point (using current displacements u)
    ξ = scv.qr.points[qp]
    # get reference geometry for this quadrature point
    A₁ = zero(Vec{3,T})
    A₂ = zero(Vec{3,T})
    a₁ = zero(Vec{3,T})
    a₂ = zero(Vec{3,T})
    for i in 1:getnbasefunctions(scv.ip_geo)
        dNdξ = Ferrite.reference_shape_gradient(scv.ip_geo, ξ, i)
        A₁  += x[i] * dNdξ[1]
        A₂  += x[i] * dNdξ[2]
        a₁  += (x[i] + u_e[i]) * dNdξ[1]
        a₂  += (x[i] + u_e[i]) * dNdξ[2]
    end
    A_metric = SymmetricTensor{2,2}((dot(A₁,A₁),dot(A₁,A₂),dot(A₂,A₂)))
    a_metric = SymmetricTensor{2,2}((dot(a₁,a₁),dot(a₁,a₂),dot(a₂,a₂)))
    # if we dont need directors, we can skip computing them and save some time
    !director && return (a₁,a₂,A_metric,a_metric)
    ∂d₁ = zero(Vec{3,Float64})
    ∂d₂ = zero(Vec{3,Float64})
    for I in 1:n_nodes
        NI = shape_value(scv.ip_shape, I, qp)
        ∂NI1, ∂NI2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, I)
        ∂d₁ += NI * zero(Vec{3,Float64}) + ∂NI1 * zero(Vec{3,Float64})
        ∂d₂ += NI * zero(Vec{3,Float64}) + ∂NI2 * zero(Vec{3,Float64})
    end
    return (a₁, a₂, A_metric, a_metric, ∂d₁, ∂d₂)
end