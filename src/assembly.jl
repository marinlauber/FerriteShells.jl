using ForwardDiff

# Compute (cos(√θ²), sin(√θ²)/√θ²) from θ² = φ₁²+φ₂² without calling sqrt at 0.
# ForwardDiff-safe: the branch is on the primal value only, so dual perturbations
# flow through whichever branch is active.  In the else branch √θ² > 0 so the
# sqrt gradient (1/2√θ²) is finite.
@inline function _cos_sinc_sq(θ²::T) where T
    if θ² < 1e-6
        return one(T) - θ²/2 + θ²^2/24,  one(T) - θ²/6 + θ²^2/120
    else
        θ = sqrt(θ²)
        return cos(θ), sin(θ)/θ
    end
end

# Returns (cosθ, sinc(θ), s_cc) where s_cc = (cosθ - sinc(θ))/θ².
# s_cc is the coefficient in ∂d_I/∂φ_k (Rodrigues director Jacobian).
# Taylor at θ²→0: s_cc → -1/3 + θ²/30. ForwardDiff-safe: branch on primal only.
@inline function _cos_sinc_sincc_sq(θ²::T) where T
    if θ² < 1e-6
        cosθ  = one(T) - θ²/2  + θ²^2/24
        sincθ = one(T) - θ²/6  + θ²^2/120
        sccθ  = -one(T)/3 + θ²/30
    else
        θ     = sqrt(θ²)
        cosθ  = cos(θ)
        sincθ = sin(θ) / θ
        sccθ  = (cosθ - sincθ) / θ²
    end
    cosθ, sincθ, sccθ
end

"""
    membrane_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love membrane residual. `u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function membrane_residuals_KL!(re, scv::ShellCellValues, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, a_metric = kinematics(scv, qp, u_e)
        E = 0.5 * (a_metric - A_metric)
        N = contravariant_elasticity(mat, A_metric) ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
                ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
            @views re[3I-2:3I] .+= v * dΩ
        end
    end
end

"""
    membrane_tangent_KL!(ke, scv, u_e, mat)

Kirchhoff–Love membrane tangent. `u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function membrane_tangent_KL!(ke, scv::ShellCellValues, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, a_metric = kinematics(scv, qp, u_e)
        E = 0.5 * (a_metric - A_metric)
        C = contravariant_elasticity(mat, A_metric)
        N = C ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
                geo_scalar = ∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) +
                             ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)
                Kgeo = geo_scalar * one(SymmetricTensor{2,3})
                H1 = SymmetricTensor{2,2}((∂NJ1, 0.5∂NJ2, 0.0))
                H2 = SymmetricTensor{2,2}((0.0, 0.5∂NJ1, ∂NJ2))
                D1 = C ⊡ H1
                D2 = C ⊡ H2
                Kmat = zero(Tensor{2,3})
                for (α,∂NIα) in enumerate((∂NI1, ∂NI2)), (β,aβ) in enumerate((a₁, a₂))
                    v = D1[α,β]*a₁ + D2[α,β]*a₂
                    Kmat += ∂NIα * (aβ ⊗ v)
                end
                @views ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * dΩ
            end
        end
    end
end

"""
Kirchhoff–Love bending strain energy.
Curvature change κ_{\\alpha\\beta} = b_{\\alpha\\beta} - B_{\\alpha\\beta} (current minus reference second fundamental form).
Requires Q2+ elements for full κ (Q4 only captures twist κ₁₂).
`u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function bending_energy_KL(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    u_e     = [Vec{3,T}((u_flat[3i-2], u_flat[3i-1], u_flat[3i])) for i in 1:n_nodes]
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        a₁  = Vec{3,T}(Tuple(scv.A₁[qp]));  a₂  = Vec{3,T}(Tuple(scv.A₂[qp]))
        a₁₁ = Vec{3,T}(Tuple(scv.A₁₁[qp])); a₁₂ = Vec{3,T}(Tuple(scv.A₁₂[qp])); a₂₂ = Vec{3,T}(Tuple(scv.A₂₂[qp]))
        for I in 1:n_nodes
            UI  = u_e[I]
            dN  = scv.dNdξ[I, qp]
            d2N = scv.d2Ndξ2[I, qp]
            a₁  += UI * dN[1];     a₂  += UI * dN[2]
            a₁₁ += UI * d2N[1,1]; a₁₂ += UI * d2N[1,2]; a₂₂ += UI * d2N[2,2]
        end
        n_n = (a₁ × a₂) / norm(a₁ × a₂)
        b   = SymmetricTensor{2,2,T}((dot(a₁₁,n_n), dot(a₁₂,n_n), dot(a₂₂,n_n)))
        κ   = b - scv.B[qp]
        D   = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        W  += 0.5 * (κ ⊡ D ⊡ κ) * scv.detJdV[qp]
    end
    return W
end

function bending_residuals_KL!(re, scv::ShellCellValues, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy_KL(u, scv, mat), u_e)
end

function bending_tangent_KL!(ke, scv::ShellCellValues, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy_KL(u, scv, mat), u_e)
end

"""
Reissner–Mindlin membrane strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).
Only the displacement DOFs (indices 5I-4:5I-2) contribute to membrane energy.
"""
function membrane_energy_RM(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_flat[5I-4], u_flat[5I-3], u_flat[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁       = scv.A₁[qp] + Δa₁
        a₂       = scv.A₂[qp] + Δa₂
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E        = 0.5 * (a_metric - scv.A_metric[qp])
        C        = contravariant_elasticity(mat, scv.A_metric[qp])
        W       += 0.5 * (E ⊡ C ⊡ E) * scv.detJdV[qp]
    end
    return W
end

"""
    membrane_residuals_RM!(re, scv, u_e, mat)

Reissner–Mindlin membrane residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_residuals_RM!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> membrane_energy_RM(u, scv, mat), u_e)
end

"""
    membrane_residuals_RM_explicit!(re, scv, u_e, mat)

RM membrane residual: ``r_I = \\int N^{\\alpha\\beta} \\partial N_I^\\alpha a_\\beta dΩ``.
Stress resultant rows ``P_\\alpha = N^{\\alpha\\beta} a_\\beta`` are precomputed once per QP.
"""
function membrane_residuals_RM_explicit!(re, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E = 0.5 * (a_metric - scv.A_metric[qp])
        N = contravariant_elasticity(mat, scv.A_metric[qp]) ⊡ E
        P₁ = N[1,1]*a₁ + N[1,2]*a₂
        P₂ = N[2,1]*a₁ + N[2,2]*a₂
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            @views re[5I-4:5I-2] .+= (∂NI1*P₁ + ∂NI2*P₂) * dΩ
        end
    end
end


"""
    membrane_tangent_RM!(ke, scv, u_e, mat)

Reissner–Mindlin membrane tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_tangent_RM!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> membrane_energy_RM(u, scv, mat), u_e)
end # 1050 μs (26 allocations: 115.21 KiB) on a 45x45 matrix

# Precompute per-QP "frame stiffness" tensors M_{\\alphaδ} = C^{\\alpha\\betaγδ} a_\\beta⊗a_γ.
# The material tangent is then K^mat_IJ = ∂N_I^\\alpha ∂N_J^δ M_{\\alphaδ} (summed over \\alpha,δ ∈ {1,2}).
# Uses C's full symmetry to reduce to 3 unique tensors (M₂₁ = transpose(M₁₂)).
@inline function frame_stiffness(C::SymmetricTensor{4,2,T}, a₁::Vec{3,T}, a₂::Vec{3,T}) where T
    M₁₁ = C[1,1,1,1]*(a₁⊗a₁) + C[1,1,1,2]*(a₁⊗a₂ + a₂⊗a₁) + C[1,2,1,2]*(a₂⊗a₂)
    M₁₂ = C[1,1,1,2]*(a₁⊗a₁) + C[1,1,2,2]*(a₁⊗a₂) + C[1,2,1,2]*(a₂⊗a₁) + C[1,2,2,2]*(a₂⊗a₂)
    M₂₂ = C[1,2,1,2]*(a₁⊗a₁) + C[1,2,2,2]*(a₁⊗a₂ + a₂⊗a₁) + C[2,2,2,2]*(a₂⊗a₂)
    M₁₁, M₁₂, M₂₂
end

"""
    membrane_tangent_RM_explicit!(ke, scv, u_e, mat)

RM membrane tangent.
Material part: ``K^\\text{mat}_{IJ} = \\partial N_I^\\alpha \\partial N_J^\\delta M_{\\alpha\\delta}`` where
``M_{\\alpha\\delta} = C^{\\alpha\\beta\\gamma\\delta} a_\\beta\\otimes a_\\gamma``.
Geometric part: ``K^\\text{geo}_{IJ} = (\\partial N_I^\\alpha N^{\\alpha\\beta} \\partial N_J^\\beta) \\mathbb{h}_3``.
Both M_{\\alphaδ} and N are precomputed once per QP outside the node loops.
"""
function membrane_tangent_RM_explicit!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E = 0.5 * (a_metric - scv.A_metric[qp])
        C = contravariant_elasticity(mat, scv.A_metric[qp])
        N = C ⊡ E
        M₁₁, M₁₂, M₂₂ = frame_stiffness(C, a₁, a₂)
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
                K_mat = ∂NI1*∂NJ1*M₁₁ + ∂NI1*∂NJ2*M₁₂ + ∂NI2*∂NJ1*transpose(M₁₂) + ∂NI2*∂NJ2*M₂₂
                K_geo = (∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) + ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)) * one(SymmetricTensor{2,3})
                @views ke[5I-4:5I-2, 5J-4:5J-2] .+= (K_mat + K_geo) * dΩ
            end
        end
    end
end # 19.969 μs (0 allocations: 0 bytes) on a 45x45 matrix (50x speedup)

"""
Reissner–Mindlin bending + transverse shear strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).

Director parametrization: d_I = G₃ + φ₁_I·T₁ + φ₂_I·T₂
where G₃ is the reference unit normal and T₁, T₂ are reference tangents from scv.

Bending strain:  κ_{\\alpha\\beta} = ½(a_\\alpha·d,\\beta + a_\\beta·d,\\alpha) - B_{\\alpha\\beta}
Transverse shear: γ_\\alpha = a_\\alpha·d

Shear correction factor κ_s = 5/6 is applied.
"""
function bending_shear_energy_RM(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_flat[5I-4], u_flat[5I-3], u_flat[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁
        a₂ = scv.A₂[qp] + Δa₂
        G₃ = scv.G₃[qp]; T₁ = scv.T₁[qp]; T₂ = scv.T₂[qp]
        d  = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            φ₁  = u_flat[5I-1]; φ₂ = u_flat[5I]
            θ²  = φ₁*φ₁ + φ₂*φ₂                 # |φ|² without sqrt (avoids 0/0 ForwardDiff gradient)
            cosθ, sincθ = _cos_sinc_sq(θ²)
            # Geometrically exact: d = cos(|φ|)G₃ + sin(|φ|)/|φ|·(φ₁T₁+φ₂T₂).
            # G₃⊥T₁,T₂ → |d|=cos²+sin²=1.  Matches additive at first order.
            d_I = cosθ*G₃ + sincθ * (φ₁*T₁ + φ₂*T₂)
            d  += scv.N[I, qp]      * d_I
            d₁ += scv.dNdξ[I, qp][1] * d_I
            d₂ += scv.dNdξ[I, qp][2] * d_I
        end
        B   = scv.B[qp]
        κ₁₁ = dot(a₁, d₁) - B[1,1]
        κ₁₂ = 0.5 * (dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
        κ₂₂ = dot(a₂, d₂) - B[2,2]
        κ   = SymmetricTensor{2,2,T}((κ₁₁, κ₁₂, κ₂₂))
        γ₁  = dot(a₁, d)
        γ₂  = dot(a₂, d)
        D    = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        Aup  = inv(scv.A_metric[qp])
        G_sh = mat.E / (2*(1 + mat.ν))
        κ_s  = 5.0/6.0
        W_bend  = 0.5 * (κ ⊡ D ⊡ κ)
        W_shear = 0.5 * κ_s * G_sh * mat.thickness *
                  (Aup[1,1]*γ₁^2 + 2*Aup[1,2]*γ₁*γ₂ + Aup[2,2]*γ₂^2)
        W += (W_bend + W_shear) * scv.detJdV[qp]
    end
    return W
end

"""
    bending_residuals_RM!(re, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_residuals_RM!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_shear_energy_RM(u, scv, mat), u_e)
end
"""
    bending_residuals_RM_explicit!(re, scv, u_e, mat)

RM bending + transverse shear residual, explicit index-notation form.

Displacement DOFs: `r_I^u = (∂₁N_I P¹ + ∂₂N_I P²) dΩ`
where `P^\\alpha = M^{\\alpha\\beta} d_{,\\beta} + Q^\\alpha d`, M = D⊡κ (bending moment), Q^\\alpha = κ_s G t A^{\\alpha\\beta} γ_\\beta.

Rotation DOFs: `r_{I,k}^φ = F_I · (∂d_I/∂φ_k) dΩ`
where `F_I = ∂₁N_I S¹ + ∂₂N_I S² + N_I (Q₁ a₁ + Q₂ a₂)`, S^\\alpha = M^{\\alpha\\beta} a_\\beta.
"""
function bending_residuals_RM_explicit!(re, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    G_sh = mat.E / (2*(1 + mat.ν))
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        G₃ = scv.G₃[qp]; T₁ = scv.T₁[qp]; T₂ = scv.T₂[qp]
        d  = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            θ²  = φ₁*φ₁ + φ₂*φ₂
            cosθ, sincθ = _cos_sinc_sq(θ²)
            d_I = cosθ*G₃ + sincθ*(φ₁*T₁ + φ₂*T₂)
            d  += scv.N[I, qp]         * d_I
            d₁ += scv.dNdξ[I, qp][1]  * d_I
            d₂ += scv.dNdξ[I, qp][2]  * d_I
        end
        B   = scv.B[qp]
        κ₁₁ = dot(a₁, d₁) - B[1,1]
        κ₁₂ = 0.5 * (dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
        κ₂₂ = dot(a₂, d₂) - B[2,2]
        κ   = SymmetricTensor{2,2,T}((κ₁₁, κ₁₂, κ₂₂))
        γ₁  = dot(a₁, d); γ₂ = dot(a₂, d)
        D   = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        M   = D ⊡ κ
        Aup = inv(scv.A_metric[qp])
        cs  = 5.0/6.0 * G_sh * mat.thickness
        Q₁  = cs * (Aup[1,1]*γ₁ + Aup[1,2]*γ₂)
        Q₂  = cs * (Aup[2,1]*γ₁ + Aup[2,2]*γ₂)
        P¹  = M[1,1]*d₁ + M[1,2]*d₂ + Q₁*d
        P²  = M[2,1]*d₁ + M[2,2]*d₂ + Q₂*d
        S¹  = M[1,1]*a₁ + M[1,2]*a₂
        S²  = M[2,1]*a₁ + M[2,2]*a₂
        dΩ  = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            NI = scv.N[I, qp]
            @views re[5I-4:5I-2] .+= (∂NI1*P¹ + ∂NI2*P²) * dΩ
            F_I = ∂NI1*S¹ + ∂NI2*S² + NI*(Q₁*a₁ + Q₂*a₂)
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            θ² = φ₁*φ₁ + φ₂*φ₂
            _, sincθ, sccθ = _cos_sinc_sincc_sq(θ²)
            dd_dφ₁ = (sincθ + sccθ*φ₁*φ₁)*T₁ + sccθ*φ₁*φ₂*T₂ - sincθ*φ₁*G₃
            dd_dφ₂ = sccθ*φ₁*φ₂*T₁ + (sincθ + sccθ*φ₂*φ₂)*T₂ - sincθ*φ₂*G₃
            re[5I-1] += dot(F_I, dd_dφ₁) * dΩ
            re[5I  ] += dot(F_I, dd_dφ₂) * dΩ
        end
    end
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_tangent_RM!(ke, scv, u, mat)
    ke .+= ForwardDiff.hessian(u -> bending_shear_energy_RM(u, scv, mat), u)
end

"""
    bending_tangent_RM_explicit!(ke, scv, u_e, mat)

RM bending + transverse shear tangent, explicit index-notation form. Four blocks per (I,J) pair:

- **uu** (3×3): `∂_\\alphaN_I ∂_γN_J (D^{\\alpha\\betaγδ} d_{,\\beta}⊗d_{,δ}) + q_{IJ}(d⊗d)` — frame_stiffness with d₁,d₂.
- **uφ** (3×2): `∂_\\alphaN_I[δM^{\\alpha\\beta}d_{,\\beta} + δQ^\\alpha N_J d] + (g_{IJ}+q_I N_J)dd_{Jl}`.
- **φu** (2×3): filled by transposing the uφ block for (I,J) into the (J,I) position.
- **φφ** (2×2): material part `∂F_I/∂φ_{lJ}·dd_{Ik}` + geometric part `F_I·∂²d_I/∂φ_k∂φ_l` (J=I only).
"""
function bending_tangent_RM_explicit!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    G_sh = mat.E / (2*(1 + mat.ν))
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        G₃ = scv.G₃[qp]; T₁ = scv.T₁[qp]; T₂ = scv.T₂[qp]
        d = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            θ² = φ₁*φ₁ + φ₂*φ₂
            cosθ, sincθ = _cos_sinc_sq(θ²)
            d_I = cosθ*G₃ + sincθ*(φ₁*T₁ + φ₂*T₂)
            d  += scv.N[I, qp]        * d_I
            d₁ += scv.dNdξ[I, qp][1] * d_I
            d₂ += scv.dNdξ[I, qp][2] * d_I
        end
        B   = scv.B[qp]
        κ₁₁ = dot(a₁, d₁) - B[1,1]
        κ₁₂ = 0.5*(dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
        κ₂₂ = dot(a₂, d₂) - B[2,2]
        κ   = SymmetricTensor{2,2,T}((κ₁₁, κ₁₂, κ₂₂))
        γ₁  = dot(a₁, d); γ₂ = dot(a₂, d)
        D   = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        M   = D ⊡ κ
        Aup = inv(scv.A_metric[qp])
        cs  = 5.0/6.0 * G_sh * mat.thickness
        Q₁  = cs*(Aup[1,1]*γ₁ + Aup[1,2]*γ₂)
        Q₂  = cs*(Aup[2,1]*γ₁ + Aup[2,2]*γ₂)
        S¹  = M[1,1]*a₁ + M[1,2]*a₂
        S²  = M[2,1]*a₁ + M[2,2]*a₂
        L₁₁, L₁₂, L₂₂ = frame_stiffness(D, d₁, d₂)
        dΩ  = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]; NI = scv.N[I, qp]
            F_I = ∂NI1*S¹ + ∂NI2*S² + NI*(Q₁*a₁ + Q₂*a₂)
            φ₁_I = u_e[5I-1]; φ₂_I = u_e[5I]
            θ²_I  = φ₁_I^2 + φ₂_I^2
            _, s_I, sc_I = _cos_sinc_sincc_sq(θ²_I)
            dd_I1 = (s_I + sc_I*φ₁_I^2)*T₁ + sc_I*φ₁_I*φ₂_I*T₂ - s_I*φ₁_I*G₃
            dd_I2 = sc_I*φ₁_I*φ₂_I*T₁ + (s_I + sc_I*φ₂_I^2)*T₂ - s_I*φ₂_I*G₃
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]; NJ = scv.N[J, qp]
                # uu block: frame_stiffness with d₁,d₂ + shear term
                q_IJ = cs*(∂NI1*(Aup[1,1]*∂NJ1+Aup[1,2]*∂NJ2) + ∂NI2*(Aup[2,1]*∂NJ1+Aup[2,2]*∂NJ2))
                K_uu = ∂NI1*∂NJ1*L₁₁ + ∂NI1*∂NJ2*L₁₂ + ∂NI2*∂NJ1*transpose(L₁₂) + ∂NI2*∂NJ2*L₂₂ + q_IJ*(d⊗d)
                @views ke[5I-4:5I-2, 5J-4:5J-2] .+= K_uu * dΩ
                # uφ and φφ material blocks: loop over rotation directions l=1,2 of node J
                φ₁_J = u_e[5J-1]; φ₂_J = u_e[5J]
                θ²_J  = φ₁_J^2 + φ₂_J^2
                _, s_J, sc_J = _cos_sinc_sincc_sq(θ²_J)
                dd_J1 = (s_J + sc_J*φ₁_J^2)*T₁ + sc_J*φ₁_J*φ₂_J*T₂ - s_J*φ₁_J*G₃
                dd_J2 = sc_J*φ₁_J*φ₂_J*T₁ + (s_J + sc_J*φ₂_J^2)*T₂ - s_J*φ₂_J*G₃
                g_IJ  = ∂NI1*(M[1,1]*∂NJ1+M[1,2]*∂NJ2) + ∂NI2*(M[2,1]*∂NJ1+M[2,2]*∂NJ2)
                q_I   = ∂NI1*Q₁ + ∂NI2*Q₂
                for (l, dd_Jl) in zip(1:2, (dd_J1, dd_J2))
                    c₁  = dot(a₁, dd_Jl); c₂ = dot(a₂, dd_Jl)
                    δκ  = SymmetricTensor{2,2,T}((∂NJ1*c₁, 0.5*(∂NJ1*c₂+∂NJ2*c₁), ∂NJ2*c₂))
                    δM  = D ⊡ δκ
                    δQ₁ = cs*(Aup[1,1]*c₁ + Aup[1,2]*c₂)
                    δQ₂ = cs*(Aup[2,1]*c₁ + Aup[2,2]*c₂)
                    col = 5J - 2 + l
                    # uφ block: material (bending+shear) + director contributions
                    v_bend  = ∂NI1*(δM[1,1]*d₁+δM[1,2]*d₂) + ∂NI2*(δM[2,1]*d₁+δM[2,2]*d₂)
                    v_shear = (∂NI1*δQ₁ + ∂NI2*δQ₂)*NJ*d
                    v_dir   = (g_IJ + q_I*NJ)*dd_Jl
                    Kuφ_col = v_bend + v_shear + v_dir
                    @views ke[5I-4:5I-2, col] .+= Kuφ_col * dΩ
                    @views ke[col, 5I-4:5I-2] .+= Kuφ_col * dΩ  # φu for (J,I) by symmetry
                    # φφ material block: δF_I = ∂_αN_I δS^α + N_I N_J δQ^α a_α
                    δS¹   = δM[1,1]*a₁ + δM[1,2]*a₂
                    δS²   = δM[2,1]*a₁ + δM[2,2]*a₂
                    δF_I  = ∂NI1*δS¹ + ∂NI2*δS² + NI*NJ*(δQ₁*a₁ + δQ₂*a₂)
                    ke[5I-1, col] += dot(δF_I, dd_I1) * dΩ
                    ke[5I,   col] += dot(δF_I, dd_I2) * dΩ
                end
            end
            # φφ geometric part (J = I only): F_I · ∂²d_I/∂φ_k∂φ_l
            # sccc = (-sinc - 3scc)/θ², Taylor at θ²→0: 1/15
            sccc_I = θ²_I < 1e-6 ? one(T)/15 : (-s_I - 3sc_I)/θ²_I
            d2_11 = (3sc_I*φ₁_I + sccc_I*φ₁_I^3)*T₁ + (sc_I + sccc_I*φ₁_I^2)*φ₂_I*T₂ - (s_I + sc_I*φ₁_I^2)*G₃
            d2_12 = (sc_I + sccc_I*φ₁_I^2)*φ₂_I*T₁ + (sc_I + sccc_I*φ₂_I^2)*φ₁_I*T₂ - sc_I*φ₁_I*φ₂_I*G₃
            d2_22 = (sc_I + sccc_I*φ₂_I^2)*φ₁_I*T₁ + (3sc_I*φ₂_I + sccc_I*φ₂_I^3)*T₂ - (s_I + sc_I*φ₂_I^2)*G₃
            ke[5I-1, 5I-1] += dot(F_I, d2_11) * dΩ
            ke[5I-1, 5I  ] += dot(F_I, d2_12) * dΩ
            ke[5I,   5I-1] += dot(F_I, d2_12) * dΩ
            ke[5I,   5I  ] += dot(F_I, d2_22) * dΩ
        end
    end
end

"""

Assemble external traction into force vector f for embedded shell elements (2D mesh in 3D).
`traction` is either a Vec{3} (uniform) or a callable x::Vec{3} -> Vec{3}.
Uses a `FacetQuadratureRule` and computes the edge length element directly from 3D node positions,
bypassing the sdim mismatch that prevents standard `FacetValues` from working on embedded meshes.
For RefQuadrilateral: facets 1,3 (bottom/top) vary in ξ₁; facets 2,4 (right/left) vary in ξ₂.
"""
function assemble_traction!(f, dh, facetset, ip::Interpolation, fqr::FacetQuadratureRule, traction)
    t_func = traction isa Function ? traction : (_ -> Vec{3}(traction))
    n_base = getnbasefunctions(ip)
    fe     = zeros(ndofs_per_cell(dh))
    ndofs_per_node = ndofs_per_cell(dh) ÷ n_base
    is_interleaved = ndofs_per_node == 5 && length(Ferrite.getfieldnames(dh)) == 1
    @inline block(I) = is_interleaved ? (5I-4:5I-2) : (3I-2:3I)
    for fc in FacetIterator(dh, facetset)
        fill!(fe, 0.0)
        x        = getcoordinates(fc)
        facet_nr = fc.current_facet_id
        qr_f     = fqr.facet_rules[facet_nr]
        tdir     = facet_nr ∈ (1, 3) ? 1 : 2  # parametric direction along edge
        for (ξ, w) in zip(qr_f.points, qr_f.weights)
            xp = zero(Vec{3,Float64})
            Jt = zero(Vec{3,Float64})  # physical tangent along edge
            for I in 1:n_base
                dN, N = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                xp += N * x[I]
                Jt += dN[tdir] * x[I]
            end
            dΓ = norm(Jt) * w
            t  = t_func(xp)
            for I in 1:n_base
                N = Ferrite.reference_shape_value(ip, ξ, I)
                fe[block(I)] .+= N * t * dΓ
            end
        end
        f[celldofs(fc)] .+= fe
    end
end

"""
scv.detJdV[qp] = ‖A₁ × A₂‖ · w (reference area × weight).
cross(a₁, a₂) already has magnitude ‖a₁ × a₂‖ (current area per parametric area).
multiplying by w integrates over the parameter domain
"""
# Follower pressure residual
function assemble_pressure!(re, scv::ShellCellValues, u_e::AbstractVector{T}, p) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        w = scv.qr.weights[qp]
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * scv.dNdξ[I, qp][1]
            Δa₂ += u_I * scv.dNdξ[I, qp][2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        n_w = cross(a₁, a₂)
        for I in 1:n_nodes
            @views re[5I-4:5I-2] .+= p * scv.N[I, qp] * n_w * w
        end
    end
end
# skew-symmetric spin tensor: spin(v)·u = v × u.
# Stored column-major in Tensors.jl: col_k = (0, v₃, -v₂), (-v₃, 0, v₁), (v₂, -v₁, 0).
@inline spin(v::Vec{3,T}) where T = Tensor{2,3,T}((zero(T), v[3], -v[2], -v[3], zero(T), v[1], v[2], -v[1], zero(T)))

"""
Load-stiffness K_pres = ∂F_p/∂u. Follower pressure n = a₁×a₂ depends on u through a₁,a₂.
∂n/∂u_J = ∂₁N_J (eₗ×a₂) + ∂₂N_J (a₁×eₗ) = ∂₁N_J (-spin(a₂)) + ∂₂N_J spin(a₁).
K_IJ = p N_I w [∂₁N_J (-spin(a₂)) + ∂₂N_J spin(a₁)]  (displacement-displacement block only).
"""
function assemble_pressure_tangent!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, p) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        w = scv.qr.weights[qp]
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        neg_sp_a₂ = -spin(a₂)   # columns: eₗ × a₂
        sp_a₁     =  spin(a₁)   # columns: a₁ × eₗ
        pw = p * w
        for I in 1:n_nodes
            NI = scv.N[I, qp]
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
                @views ke[5I-4:5I-2, 5J-4:5J-2] .+= (pw * NI) * (∂NJ1 * neg_sp_a₂ + ∂NJ2 * sp_a₁)
            end
        end
    end
end

"""
    apply_pointload!(f, dh, nodeset_name, load)

Add a concentrated force `load::Vec{3}` to the displacement DOFs of all nodes in `nodeset_name`.
Works for both single-field (`:u` only) and two-field (`:u`, `:θ`) DofHandlers; in both cases the
`:u` DOFs for node I in a cell occupy local positions `3I-2:3I`.
"""
function apply_pointload!(f, dh, nodeset_name::String, load::Vec{3})
    node_set  = getnodeset(dh.grid, nodeset_name)
    processed = Set{Int}()
    for cell in CellIterator(dh)
        nodes = getnodes(cell)
        cd    = celldofs(cell)
        for (I, gid) in enumerate(nodes)
            if gid ∈ node_set && gid ∉ processed
                push!(processed, gid)
                @views f[cd[3I-2:3I]] .+= load
            end
        end
    end
end