# Shell kinematics shared by the membrane, KL and RM kernels

# Deformed covariant basis (a₁, a₂) at qp with the displacement dofs of u_e at scv.ru.
@inline function covariant_basis(scv, qp, u_e::AbstractVector{T}) where T
    ru  = scv.ru
    Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
    for I in 1:getnbasefunctions(scv)
        dN = scv.dNdξ[I, qp]
        u_I = Vec{3}(@view u_e[ru[3I-2:3I]])
        Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
    end
    scv.A₁[qp] + Δa₁, scv.A₂[qp] + Δa₂
end

# Green–Lagrange membrane strain E = ½(a_metric - A_metric_ref).
@inline membrane_strain(a₁, a₂, A_metric_ref) = 0.5 * (SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂))) - A_metric_ref)

# Scalar factors of the Rodrigues director and its first and second derivatives
@inline function rodrigues_coeffs(θ²::T) where T
    if θ² < 1e-6
        return one(T) - θ²/2 + θ²^2/24, one(T) - θ²/6 + θ²^2/120, -one(T)/3 + θ²/30, one(T)/15
    end
    θ = sqrt(θ²)
    c = cos(θ)
    s = sin(θ)/θ
    sc = (c - s) / θ²
    c, s, sc, (-s - 3sc) / θ²
end

# Rodrigues director of a node d = cos θ G₃ + sinc θ (φ₁T₁ + φ₂T₂)
@inline function rodrigues_director(φ₁, φ₂, G₃, T₁, T₂)
    c, s = rodrigues_coeffs(φ₁*φ₁ + φ₂*φ₂)
    c*G₃ + s*(φ₁*T₁ + φ₂*T₂)
end

# Rodrigues director and derivatives: d, ∂d/∂φ₁, ∂d/∂φ₂ and ∂²d/∂φ₁², ∂²d/∂φ₁∂φ₂, ∂²d/∂φ₂²
@inline function rodrigues(φ₁, φ₂, G₃, T₁, T₂)
    c, s, sc, scc = rodrigues_coeffs(φ₁*φ₁ + φ₂*φ₂)
    d    = c*G₃ + s*(φ₁*T₁ + φ₂*T₂)
    dd₁  = (s + sc*φ₁^2)*T₁ + sc*φ₁*φ₂*T₂ - s*φ₁*G₃
    dd₂  = sc*φ₁*φ₂*T₁ + (s + sc*φ₂^2)*T₂ - s*φ₂*G₃
    d²₁₁ = (3sc*φ₁ + scc*φ₁^3)*T₁ + (sc + scc*φ₁^2)*φ₂*T₂ - (s + sc*φ₁^2)*G₃
    d²₁₂ = (sc + scc*φ₁^2)*φ₂*T₁ + (sc + scc*φ₂^2)*φ₁*T₂ - sc*φ₁*φ₂*G₃
    d²₂₂ = (sc + scc*φ₂^2)*φ₁*T₁ + (3sc*φ₂ + scc*φ₂^3)*T₂ - (s + sc*φ₂^2)*G₃
    (; d, dd₁, dd₂, d²₁₁, d²₁₂, d²₂₂)
end

# Per-node directors
@inline function nodal_directors(G₃s, T₁s, T₂s, u_e, rθ, ::Val{N}) where N
    ntuple(I -> rodrigues_director(u_e[rθ[2I-1]], u_e[rθ[2I]], G₃s[I], T₁s[I], T₂s[I]), Val(N))
end
@inline nodal_directors(scv, u_e) = nodal_directors(scv.G₃_elem, scv.T₁_elem, scv.T₂_elem,
                                                    u_e, scv.rθ, Val(getnbasefunctions(scv.ip_shape)))
# full Rodrigues data of the element
@inline function nodal_rotations(scv, u_e)
    rθ = scv.rθ
    ntuple(Val(getnbasefunctions(scv.ip_shape))) do I
        rodrigues(u_e[rθ[2I-1]], u_e[rθ[2I]], scv.G₃_elem[I], scv.T₁_elem[I], scv.T₂_elem[I])
    end
end

# Director d and its covariant derivatives d₁ = d,₁, d₂ = d,₂ at qp
@inline function interpolate_director(scv, qp, dirs)
    V = eltype(dirs)
    d = zero(V); d₁ = zero(V); d₂ = zero(V)
    for I in eachindex(dirs)
        d  += scv.N[I, qp]       * dirs[I]
        d₁ += scv.dNdξ[I, qp][1] * dirs[I]
        d₂ += scv.dNdξ[I, qp][2] * dirs[I]
    end
    d, d₁, d₂
end

# Director field (d, d₁, d₂) at qp straight from the element vector.
@inline director_field(scv, qp, u_e) = interpolate_director(scv, qp, nodal_directors(scv, u_e))

# Bending curvature change κ_αβ = ½(a_α·d,β + a_β·d,α) − B₀_αβ
@inline function curvature_tensor(a₁, a₂, d₁, d₂, B)
    κ₁₁ = dot(a₁, d₁) - B[1,1]
    κ₁₂ = 0.5*(dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
    κ₂₂ = dot(a₂, d₂) - B[2,2]
    SymmetricTensor{2,2}((κ₁₁, κ₁₂, κ₂₂))
end

# MITC tying strains of the element from its nodal directors; nothing without MITC.
@inline element_tying_strains(scv, u_e, dirs) = tying_strains(scv.mitc, shear_tying_state(scv.mitc, u_e, scv.ru, dirs))

# Reissner-Mindlin covariant basis, director and its derivatives, curvature κ, and transverse shear strains
@inline function rm_kinematics(scv, qp, u_e, dirs, γ_k)
    a₁, a₂ = covariant_basis(scv, qp, u_e)
    d, d₁, d₂ = interpolate_director(scv, qp, dirs)
    κ = curvature_tensor(a₁, a₂, d₁, d₂, scv.B₀[qp])
    γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ_k, scv.mitc)
    γ₀ = scv.γ₀[qp]
    a₁, a₂, d, d₁, d₂, κ, γ₁ - γ₀[1], γ₂ - γ₀[2]
end
