
"""
    kinematics(scv, qp, x, u_e)

Compute the current kinematics at a quadrature point, given the reference geometry and current displacements.
Returns the current basis vectors a₁, a₂, and the reference and current metric tensors
A_{αβ} = A_α · A_β and a_{αβ} = a_α · a_β.
"""
function kinematics(scv, qp, x, u_e::AbstractVector{<:Vec{3,T}}) where T
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
    return a₁,a₂,A_metric,a_metric
end


struct ShellKinematics{T}
    a1::Vec{3,T}
    a2::Vec{3,T}
    A_metric::SymmetricTensor{2,2,T}
    a_metric::SymmetricTensor{2,2,T}
    E::SymmetricTensor{2,2,T}
    K::SymmetricTensor{2,2,T}
    γ::Vec{2,T}
end

function update!(scv::ShellCellValues, q::Int, u_e, d_e)
    dNdxi = shape_gradient(scv.cellvalues, q)
    # compute refernce basis vector
    A1 = zero(Vec{3,T})
    A2 = zero(Vec{3,T})
    a1 = zero(Vec{3,T})
    a2 = zero(Vec{3,T})
    for I in 1:getnbasefunctions(scv.cellvalues)
        A1 += scv.Xnodal[I] * dNdxi[I,1]
        A2 += scv.Xnodal[I] * dNdxi[I,2]
        a1 += (scv.Xnodal[I] + u_e[I]) * dNdxi[I,1]
        a2 += (scv.Xnodal[I] + u_e[I]) * dNdxi[I,2]
    end
    # metric
    A_metric = SymmetricTensor{2,2}(dot(A1,A1),dot(A1,A2),dot(A2,A2))
    a_metric = SymmetricTensor{2,2}(dot(a1,a1),dot(a1,a2),dot(a2,a2))
    # normal and area element
    n = normalize(cross(a1,a2))
    detJ = norm(cross(a1,a2))
    # director field and derivatives
    d = zero(Vec{3,T})
    dd1 = zero(Vec{3,T})
    dd2 = zero(Vec{3,T})
    for I in 1:getnbasefunctions(scv.cellvalues)
        NI = shape_value(scv.cellvalues, I, q)
        d += d_e[I] * NI
        dd1 += d_e[I] * dNdxi[I,1]
        dd2 += d_e[I] * dNdxi[I,2]
    end
    # TODO this can be enforced differently as well
    d = normalize(d)
    return ShellGeometry(A1, A2, A_metric, a1, a2, a_metric, n, detJ, d, dd1, dd2)
end

struct ShellGeometry{T}
    A1::Vec{3,T}
    A2::Vec{3,T}
    A_metric::SymmetricTensor{2,2,T}

    a1::Vec{3,T}
    a2::Vec{3,T}
    a_metric::SymmetricTensor{2,2,T}

    normal::Vec{3,T}
    detJ::T

    d::Vec{3,T}
    dd1::Vec{3,T}
    dd2::Vec{3,T}
end

"""
    E_{αβ} = 1/2 (a_{αβ} - A_{αβ})
Green-Lagrange strain in the local tangent plane, from the metric tensors.
    a_{αβ} = a_α · a_β and A_{αβ} = A_α · A_β
with a_α = ∂x/∂ξ_α and A_α = ∂X/∂ξ_α (current and reference basis vectors).
"""
function compute_membrane_strain(geom::ShellGeometry)
    A = geom.A_metric
    a = geom.a_metric
    return 0.5 * (a - A) # Green-Lagrange strain in the local tangent plane
end

"""
Curvature from director gradients
        b_{αβ} = d_{,α} · a_β
Symmetrized appropriately for bending strain measure
     K_{αβ} = 1/2(b_{αβ} - d_{βα}) - B_{αβ}
where reference curvature can be zero for flat shells initially.
"""
function compute_bending_strain(geom::ShellGeometry)
    a1 = geom.a1
    a2 = geom.a2
    dd1 = geom.dd1
    dd2 = geom.dd2

    k11 = dot(dd1, a1)
    k12 = 0.5*(dot(dd1,a2) + dot(dd2,a1))
    k22 = dot(dd2, a2)

    return SymmetricTensor{2,2}(k11, k12, k22)
end

function compute_raw_shear(geom)
    γ1 = dot(geom.d, geom.a1)
    γ2 = dot(geom.d, geom.a2)
    return Vec{2}(γ1, γ2)
end

"""
    γ_{α} = d · a_α
Shear strain measure from the director field and current basis vectors.
In a more complete formulation, this would be the difference between the current
shear and the reference shear, but for a flat reference geometry the initial shear is zero
"""
function compute_shell_strain(geom)
    E = compute_membrane_strain(geom)
    K = compute_bending_strain(geom)
    γ = compute_raw_shear(geom)

    return ShellStrain(E, K, γ)
end