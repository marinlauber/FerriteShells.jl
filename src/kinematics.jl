
"""
    kinematics(scv, qp, u_e)

Compute current kinematics at quadrature point `qp` given nodal displacements `u_e`
(flat vector of length `3 * n_nodes`: [u₁, v₁, w₁, …]).

Reference geometry (A₁, A₂, A_metric) is read from `scv`, which must have been
`reinit!`-ed with the element coordinates before calling this function.

Returns `(a₁, a₂, A_metric, a_metric)`.
"""
function kinematics(scv, qp, u_e::AbstractVector{T}) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
    for i in 1:n_nodes
        ui  = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
        Δa₁ += ui * scv.dNdξ[i, qp][1]
        Δa₂ += ui * scv.dNdξ[i, qp][2]
    end
    a₁       = scv.A₁[qp] + Δa₁
    a₂       = scv.A₂[qp] + Δa₂
    a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
    return a₁, a₂, scv.A_metric[qp], a_metric
end

# TODO remove these or find a better fix
@inline function kinematics_strains(scv::ShellCellValues{QR,IPG,IPS,T,E}, qp, u_e) where {QR,IPG,IPS,T,E<:LinearStrain}
    n_nodes = getnbasefunctions(scv.ip_shape)
    Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
    for i in 1:n_nodes
        ui  = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
        Δa₁ += ui * scv.dNdξ[i, qp][1]
        Δa₂ += ui * scv.dNdξ[i, qp][2]
    end
    a₁ = scv.A₁[qp] + Δa₁
    a₂ = scv.A₂[qp] + Δa₂
    e12 = 0.5 * (dot(scv.A₁[qp],Δa₂) + dot(scv.A₂[qp],Δa₁))
    return a₁, a₂, SymmetricTensor{2,2,T}((dot(scv.A₁[qp],Δa₁), e12, dot(scv.A₂[qp],Δa₂)))
end
@inline function kinematics_strains(scv::ShellCellValues{QR,IPG,IPS,T,E}, qp, u_e) where {QR,IPG,IPS,T,E<:GreenLagrangeStrain}
    n_nodes = getnbasefunctions(scv.ip_shape)
    Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
    for i in 1:n_nodes
        ui  = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
        Δa₁ += ui * scv.dNdξ[i, qp][1]
        Δa₂ += ui * scv.dNdξ[i, qp][2]
    end
    a₁ = scv.A₁[qp] + Δa₁
    a₂ = scv.A₂[qp] + Δa₂
    return a₁, a₂, 0.5 * (SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂))) - scv.A_metric[qp])
end