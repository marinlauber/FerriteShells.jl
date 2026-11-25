# Reissner-Mindlin bending and transverse shear, and the RM element kernel

# Per-QP strain energy density for the RM element, κ_s = 5/6 shear correction.
@inline function rm_qp_energy(mat::LinearElastic, c_ms::SymmetricTensor{2,2,T},
                              κ, γ₁, γ₂, A_metric, A₁, A₂, G₃) where T
    N, _, D, Cs = shell_stress_and_stiffness(mat, c_ms, A_metric)
    E = (c_ms - A_metric) / 2
    return 0.5*(N ⊡ E) + 0.5*(κ ⊡ D ⊡ κ) + 0.5*(Cs[1,1]*γ₁^2 + 2*Cs[1,2]*γ₁*γ₂ + Cs[2,2]*γ₂^2)
end

# Hyperelastic: 3-point through-thickness Gauss quadrature of W(C(ξ₃)).
@inline function rm_qp_energy(mat::Hyperelastic, c_ms::SymmetricTensor{2,2,T},
                              κ, γ₁, γ₂, A_metric, A₁, A₂, G₃) where T
    det_A, Jinv = reference_frame(A_metric, A₁, A₂, G₃)
    ξ3s, w3s = gauss3_thickness(mat.thickness)
    W = zero(T)
    for (ξ3, w3) in zip(ξ3s, w3s)
        c_ξ = c_ms + 2*ξ3*κ
        W  += W_phys(mat, c_ξ, γ₁, γ₂, det_A, Jinv) * w3
    end
    return W
end

"""
    energy_RM(u_flat, scv::ShellCellValues, mat)

Total Reissner–Mindlin strain energy (membrane + bending + transverse shear) per element.
`u_flat = u[celldofs(cell)]`, with the displacement and rotation dofs at `scv.ru` and `scv.rθ`.
Material dispatch happens at the per-QP level via `rm_qp_energy`.
"""
function energy_RM(u_flat, scv::ShellCellValues, mat)
    T    = eltype(u_flat)
    dirs = nodal_directors(scv, u_flat)
    γ_k  = element_tying_strains(scv, u_flat, dirs)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, _, _, _, κ, γ₁, γ₂ = rm_kinematics(scv, qp, u_flat, dirs, γ_k)
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        W += rm_qp_energy(mat, c_ms, κ, γ₁, γ₂, scv.A_metric[qp],
                          scv.A₁[qp], scv.A₂[qp], scv.G₃[qp]) * scv.detJdV[qp]
    end
    return W
end

"""
    residuals_RM_FD!(re, scv, u_e, mat)

Reissner–Mindlin total residual (membrane + bending + shear) via ForwardDiff.gradient of `energy_RM`.
`u_e = u[celldofs(cell)]`, with the displacement and rotation dofs at `scv.ru` and `scv.rθ`.
"""
function residuals_RM_FD!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> energy_RM(u, scv, mat), u_e)
end

"""
    tangent_RM_FD!(ke, scv, u_e, mat)

Reissner–Mindlin total tangent (membrane + bending + shear) via ForwardDiff.hessian of `energy_RM`.
`u_e = u[celldofs(cell)]`, with the displacement and rotation dofs at `scv.ru` and `scv.rθ`.
"""
function tangent_RM_FD!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> energy_RM(u, scv, mat), u_e)
end

# Positions in u_e of node I's five dofs (u₁,u₂,u₃,φ₁,φ₂).
@inline node_dofs(ru, rθ, I) = (ru[3I-2], ru[3I-1], ru[3I], rθ[2I-1], rθ[2I])

# Shear-strain operators at qp: for every node J, the sensitivities ∂γ_α/∂(u_J, φ_J) as
# 5-tuples (u₁,u₂,u₃,φ₁,φ₂), one per component α; `node_dofs` gives their positions in u_e.
#
# Without MITC, γ_α = a_α·d at the quadrature point, so ∂γ_α/∂u_J = ∂_αN_J d and
# ∂γ_α/∂φ_J = N_J a_α·∂d_J/∂φ. With MITC, γ_α is the tying interpolation Σ_k h^α_k a_{α_k}·d at
# the tying points, and the same derivatives are taken there and interpolated. The nodal
# frames of the MITC object are those of `scv`, so both share the nodal Rodrigues data `rot`.
@inline function shear_operators(scv, qp, kin, rot, ::Nothing)
    a₁, a₂, d = kin
    ntuple(Val(length(rot))) do J
        ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]; NJ = scv.N[J, qp]
        dd1 = rot[J].dd₁; dd2 = rot[J].dd₂
        u1 = ∂NJ1*d; u2 = ∂NJ2*d
        ((u1[1], u1[2], u1[3], NJ*dot(a₁, dd1), NJ*dot(a₁, dd2)),
         (u2[1], u2[2], u2[3], NJ*dot(a₂, dd1), NJ*dot(a₂, dd2)))
    end
end
@inline function shear_operators(scv, qp, kin, rot, tie::Tuple)
    mitc = scv.mitc
    T = eltype(kin[1])
    ntuple(Val(length(rot))) do J
        dd1 = rot[J].dd₁; dd2 = rot[J].dd₂
        b1u = zero(Vec{3,T}); b2u = zero(Vec{3,T})
        b1φ1 = zero(T); b1φ2 = zero(T); b2φ1 = zero(T); b2φ2 = zero(T)
        @inbounds for k in eachindex(tie)
            h1 = mitc.h_tie_1[qp,k]; h2 = mitc.h_tie_2[qp,k]
            (h1 == 0 && h2 == 0) && continue     # off-component entry of a quad scheme
            at, dt = tie[k]
            Bu  = mitc.dN_tie[J,k] * dt
            Bφ1 = mitc.N_tie[J,k] * dot(at, dd1)
            Bφ2 = mitc.N_tie[J,k] * dot(at, dd2)
            b1u += h1 * Bu;  b1φ1 += h1 * Bφ1;  b1φ2 += h1 * Bφ2
            b2u += h2 * Bu;  b2φ1 += h2 * Bφ1;  b2φ2 += h2 * Bφ2
        end
        ((b1u[1], b1u[2], b1u[3], b1φ1, b1φ2), (b2u[1], b2u[2], b2u[3], b2φ1, b2φ2))
    end
end

# Coefficients (g₁, g₂) with ∂²γ_α/∂u_I∂φ_{J,l} = g_α ∂d_J/∂φ_l.
@inline function shear_coupling(scv, qp, ::Nothing, I, J)
    ∂NI1, ∂NI2 = scv.dNdξ[I, qp]; NJ = scv.N[J, qp]
    ∂NI1*NJ, ∂NI2*NJ
end
@inline function shear_coupling(scv, qp, tie::Tuple, I, J)
    mitc = scv.mitc
    g1 = 0.0; g2 = 0.0
    @inbounds for k in eachindex(tie)
        g = mitc.dN_tie[I,k] * mitc.N_tie[J,k]
        g1 += mitc.h_tie_1[qp,k] * g
        g2 += mitc.h_tie_2[qp,k] * g
    end
    g1, g2
end

# Vectors (W₁, W₂) with ∂²γ_α/∂φ_{I,k}∂φ_{I,l} = W_α·∂²d_I/∂φ_k∂φ_l.
@inline shear_director_weight(scv, qp, kin, ::Nothing, I) = (NI = scv.N[I, qp]; (NI*kin[1], NI*kin[2]))
@inline function shear_director_weight(scv, qp, kin, tie::Tuple, I)
    mitc = scv.mitc
    T = eltype(kin[1])
    W₁ = zero(Vec{3,T}); W₂ = zero(Vec{3,T})
    @inbounds for k in eachindex(tie)
        w = mitc.N_tie[I,k] * tie[k][1]
        W₁ += mitc.h_tie_1[qp,k] * w
        W₂ += mitc.h_tie_2[qp,k] * w
    end
    W₁, W₂
end

"""
    bending_residuals_RM!(re, scv, u_e, mat)

RM bending + transverse shear residual, explicit index-notation form.

Displacement DOFs: ``r_I^u = (\\partial_1 N_I P^1 + \\partial_2 N_I P^2) \\, d\\Omega``
where ``P^\\alpha = M^{\\alpha\\beta} d_{,\\beta}``, ``M = D : \\kappa`` (bending moment).

Rotation DOFs: ``r_{I,k}^\\varphi = F_I \\cdot (\\partial d_I/\\partial \\varphi_k) \\, d\\Omega``
where ``F_I = \\partial_1 N_I S^1 + \\partial_2 N_I S^2``, ``S^\\alpha = M^{\\alpha\\beta} a_\\beta``.

Transverse shear adds ``Q^\\alpha \\, \\partial\\gamma_\\alpha/\\partial u`` to every DOF, with
``Q^\\alpha = \\kappa_s G t A^{\\alpha\\beta} \\gamma_\\beta``. Without MITC,
``\\partial\\gamma_\\alpha/\\partial u_I = \\partial_\\alpha N_I d`` and
``\\partial\\gamma_\\alpha/\\partial\\varphi_{I,k} = N_I a_\\alpha \\cdot \\partial d_I/\\partial\\varphi_k``; with MITC they are
taken through the tying interpolation (see [`bending_tangent_RM!`](@ref)), so the residual is
the exact gradient of `energy_RM`.
"""
bending_residuals_RM!(re, scv::ShellCellValues, u_e, mat) = rm_element!(nothing, re, scv, u_e, mat, Val(:bending))

function bending_residual_qp!(re, scv, qp, kin, D, Cs, rot, tie)
    a₁, a₂, _, d₁, d₂, κ, γ₁, γ₂ = kin
    M   = D ⊡ κ
    Q₁  = Cs[1,1]*γ₁ + Cs[1,2]*γ₂
    Q₂  = Cs[2,1]*γ₁ + Cs[2,2]*γ₂
    P¹  = M[1,1]*d₁ + M[1,2]*d₂
    P²  = M[2,1]*d₁ + M[2,2]*d₂
    S¹  = M[1,1]*a₁ + M[1,2]*a₂
    S²  = M[2,1]*a₁ + M[2,2]*a₂
    dΩ  = scv.detJdV[qp]
    Bγ  = shear_operators(scv, qp, kin, rot, tie)
    ru, rθ = scv.ru, scv.rθ
    for I in eachindex(rot)
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        @views re[ru[3I-2:3I]] .+= (∂NI1*P¹ + ∂NI2*P²) * dΩ
        F_I = ∂NI1*S¹ + ∂NI2*S²
        re[rθ[2I-1]] += dot(F_I, rot[I].dd₁) * dΩ
        re[rθ[2I]  ] += dot(F_I, rot[I].dd₂) * dΩ
        b1, b2 = Bγ[I]
        dI = node_dofs(ru, rθ, I)
        for p in 1:5
            re[dI[p]] += (Q₁*b1[p] + Q₂*b2[p]) * dΩ
        end
    end
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)

RM bending and transverse shear tangent, explicit index-notation form.

Bending, per (I,J) pair:

- **uu** (3×3): ``\\partial_\\alpha N_I \\partial_\\gamma N_J (D^{\\alpha\\beta\\gamma\\delta} d_{,\\beta} \\otimes d_{,\\delta})`` — stiffness block with ``d_1, d_2``.
- **uφ** (3×2): ``\\partial_\\alpha N_I \\delta M^{\\alpha\\beta} d_{,\\beta} + g_{IJ} \\mathrm{dd}_{Jl}``, ``g_{IJ} = \\partial_\\alpha N_I M^{\\alpha\\beta} \\partial_\\beta N_J``; **φu** is its transpose.
- **φφ** (2×2): material part ``\\partial F_I/\\partial\\varphi_{l,J} \\cdot \\mathrm{dd}_{Ik}`` + geometric part ``F_I \\cdot \\partial^2 d_I/\\partial\\varphi_k\\partial\\varphi_l`` (J=I only).

Transverse shear: material part ``C_s^{\\alpha\\beta} B^\\gamma_\\alpha(I) \\otimes B^\\gamma_\\beta(J)`` over all five DOFs of
each node, with ``B^\\gamma_\\alpha = \\partial\\gamma_\\alpha/\\partial(u, \\varphi)``, plus the geometric part
``Q^\\alpha \\partial^2\\gamma_\\alpha``. With MITC the shear sensitivities are obtained by differentiating
through the tying interpolation:

```math
\\frac{\\partial\\gamma_\\alpha(q)}{\\partial u_J}
  = \\sum_k h^\\alpha_{qk} \\frac{\\partial N_J}{\\partial\\xi_\\alpha}(\\xi_k)\\cdot d(\\xi_k)
```
```math
\\frac{\\partial\\gamma_\\alpha(q)}{\\partial\\varphi_{J,l}}
  = \\sum_k h^\\alpha_{qk}\\, N_J(\\xi_k)\\left(a_\\alpha(\\xi_k)\\cdot\\frac{\\partial d_J}{\\partial\\varphi_l}\\right)
```

which keeps the tangent consistent with [`bending_residuals_RM!`](@ref), so Newton converges
quadratically. Without MITC the tying point is the quadrature point itself (``h = 1``).
"""
bending_tangent_RM!(ke, scv::ShellCellValues, u_e, mat) = rm_element!(ke, nothing, scv, u_e, mat, Val(:bending))

function bending_tangent_qp!(ke, scv, qp, kin, D, Cs, rot, tie)
    a₁, a₂, _, d₁, d₂, κ, γ₁, γ₂ = kin
    T   = eltype(a₁)
    M   = D ⊡ κ
    Q₁  = Cs[1,1]*γ₁ + Cs[1,2]*γ₂
    Q₂  = Cs[2,1]*γ₁ + Cs[2,2]*γ₂
    S¹  = M[1,1]*a₁ + M[1,2]*a₂
    S²  = M[2,1]*a₁ + M[2,2]*a₂
    L₁₁, L₁₂, L₂₂ = stiffness_block(D, d₁, d₂)
    dΩ  = scv.detJdV[qp]
    Bγ  = shear_operators(scv, qp, kin, rot, tie)
    ru, rθ = scv.ru, scv.rθ
    for I in eachindex(rot)
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        F_I = ∂NI1*S¹ + ∂NI2*S²                  # bending traction at I
        dd_I1, dd_I2 = rot[I].dd₁, rot[I].dd₂
        b1I, b2I = Bγ[I]
        dI = node_dofs(ru, rθ, I)
        for J in eachindex(rot)
            ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
            # uu bending: stiffness block with d₁,d₂
            K_uu = ∂NI1*∂NJ1*L₁₁ + ∂NI1*∂NJ2*L₁₂ + ∂NI2*∂NJ1*transpose(L₁₂) + ∂NI2*∂NJ2*L₂₂
            @views ke[ru[3I-2:3I], ru[3J-2:3J]] .+= K_uu * dΩ
            # shear material: 5×5 outer product over all DOFs of (I,J)
            b1J, b2J = Bγ[J]
            dJ = node_dofs(ru, rθ, J)
            for p in 1:5, q in 1:5
                cbq = b1I[p]*(Cs[1,1]*b1J[q] + Cs[1,2]*b2J[q]) +
                      b2I[p]*(Cs[2,1]*b1J[q] + Cs[2,2]*b2J[q])
                ke[dI[p], dJ[q]] += cbq * dΩ
            end
            # uφ/φu: bending material + bending and shear geometric; φφ: bending material
            g_IJ = ∂NI1*(M[1,1]*∂NJ1+M[1,2]*∂NJ2) + ∂NI2*(M[2,1]*∂NJ1+M[2,2]*∂NJ2)
            g₁, g₂ = shear_coupling(scv, qp, tie, I, J)
            g_IJ += Q₁*g₁ + Q₂*g₂
            for (l, dd_Jl) in ((1, rot[J].dd₁), (2, rot[J].dd₂))
                c₁ = dot(a₁, dd_Jl); c₂ = dot(a₂, dd_Jl)
                δκ = SymmetricTensor{2,2,T}((∂NJ1*c₁, 0.5*(∂NJ1*c₂+∂NJ2*c₁), ∂NJ2*c₂))
                δM = D ⊡ δκ
                col = rθ[2J-2+l]
                Kuφ = ∂NI1*(δM[1,1]*d₁+δM[1,2]*d₂) + ∂NI2*(δM[2,1]*d₁+δM[2,2]*d₂) + g_IJ*dd_Jl
                @views ke[ru[3I-2:3I], col] .+= Kuφ * dΩ
                @views ke[col, ru[3I-2:3I]] .+= Kuφ * dΩ
                δF_I = ∂NI1*(δM[1,1]*a₁ + δM[1,2]*a₂) + ∂NI2*(δM[2,1]*a₁ + δM[2,2]*a₂)
                ke[rθ[2I-1], col] += dot(δF_I, dd_I1) * dΩ
                ke[rθ[2I],   col] += dot(δF_I, dd_I2) * dΩ
            end
        end
        # φφ geometric (J = I): (F_I + Q^α W_α)·∂²d_I/∂φ_k∂φ_l
        d2_11, d2_12, d2_22 = rot[I].d²₁₁, rot[I].d²₁₂, rot[I].d²₂₂
        W₁, W₂ = shear_director_weight(scv, qp, kin, tie, I)
        V_I = F_I + Q₁*W₁ + Q₂*W₂
        ke[rθ[2I-1], rθ[2I-1]] += dot(V_I, d2_11) * dΩ
        ke[rθ[2I-1], rθ[2I]  ] += dot(V_I, d2_12) * dΩ
        ke[rθ[2I],   rθ[2I-1]] += dot(V_I, d2_12) * dΩ
        ke[rθ[2I],   rθ[2I]  ] += dot(V_I, d2_22) * dΩ
    end
end

"""
    residuals_RM!(re, scv, u_e, mat)

Complete RM element residual (membrane + bending + transverse shear): the sum of
[`membrane_residuals!`](@ref) and [`bending_residuals_RM!`](@ref) in a single quadrature
loop that evaluates the material once per point.
"""
residuals_RM!(re, scv::ShellCellValues, u_e, mat) = rm_element!(nothing, re, scv, u_e, mat, Val(:all))

"""
    tangent_RM!(ke, scv, u_e, mat)

Complete RM element tangent: the sum of [`membrane_tangent!`](@ref) and
[`bending_tangent_RM!`](@ref) in a single quadrature loop that evaluates the material once
per point.
"""
tangent_RM!(ke, scv::ShellCellValues, u_e, mat) = rm_element!(ke, nothing, scv, u_e, mat, Val(:all))

"""
    residuals_and_tangent_RM!(ke, re, scv, u_e, mat)

[`residuals_RM!`](@ref) and [`tangent_RM!`](@ref) together, sharing the kinematics and the
material evaluation at each quadrature point — the per-element call of a Newton iteration.
Prefer it over the split membrane/bending kernels unless the two parts need different
treatment (e.g. a membrane-only problem or different quadrature rules).
"""
residuals_and_tangent_RM!(ke, re, scv::ShellCellValues, u_e, mat) = rm_element!(ke, re, scv, u_e, mat, Val(:all))

# Material response for the requested parts; membrane-only skips the shear stiffness, which
# is costly for Hyperelastic.
@inline qp_material(mat, ::Val{:membrane}, c_ms, A_metric, A₁, A₂, G₃) =
    (membrane_stress_and_tangent(mat, c_ms, A_metric, A₁, A₂, G₃)..., nothing, nothing)
@inline qp_material(mat, ::Val{:bending}, c_ms, A_metric, A₁, A₂, G₃) =
    (nothing, nothing, bending_and_shear_stiffness(mat, c_ms, A_metric, A₁, A₂, G₃)...)
@inline qp_material(mat, ::Val{:all}, c_ms, A_metric, A₁, A₂, G₃) =
    shell_stress_and_stiffness(mat, c_ms, A_metric, A₁, A₂, G₃)

# Element kernel shared by all explicit entry points
function rm_element!(ke, re, scv::ShellCellValues, u_e::AbstractVector{T}, mat, parts::Val{P}) where {T,P}
    mem  = P !== :bending
    bend = P !== :membrane
    rot  = bend ? nodal_rotations(scv, u_e) : nothing
    dirs = bend ? map(r -> r.d, rot) : nothing
    tie  = bend ? shear_tying_state(scv.mitc, u_e, scv.ru, dirs) : nothing
    γ_k  = bend ? tying_strains(scv.mitc, tie) : nothing
    for qp in 1:getnquadpoints(scv)
        kin = bend ? rm_kinematics(scv, qp, u_e, dirs, γ_k) : covariant_basis(scv, qp, u_e)
        a₁, a₂ = kin
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        N, C, D, Cs = qp_material(mat, parts, c_ms, scv.A_metric[qp], scv.A₁[qp], scv.A₂[qp], scv.G₃[qp])
        if re !== nothing
            mem  && membrane_residual_qp!(re, scv, qp, a₁, a₂, N)
            bend && bending_residual_qp!(re, scv, qp, kin, D, Cs, rot, tie)
        end
        if ke !== nothing
            mem  && membrane_tangent_qp!(ke, scv, qp, a₁, a₂, N, C)
            bend && bending_tangent_qp!(ke, scv, qp, kin, D, Cs, rot, tie)
        end
    end
end