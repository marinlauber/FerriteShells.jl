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

# Deformed covariant basis (a₁, a₂) at qp from RM 5-DOF displacement vector u_e.
@inline function covariant_basis(scv, qp, u_e::AbstractVector{T}, n_nodes) where T
    Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
    for I in 1:n_nodes
        dN = scv.dNdξ[I, qp]
        u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
        Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
    end
    scv.A₁[qp] + Δa₁, scv.A₂[qp] + Δa₂
end

# Green–Lagrange membrane strain E = ½(a_metric - A_metric_ref).
@inline membrane_strain(a₁, a₂, A_metric_ref) =
    0.5 * (SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂))) - A_metric_ref)

# Interpolated director d and covariant derivatives d₁ = d,₁, d₂ = d,₂ at qp.
# Rodrigues: d_I = cos|φ|·G₃_I + sinc|φ|·(φ₁T₁_I+φ₂T₂_I), |d_I| = 1 (per-node frames).
@inline function director_field(scv, qp, u_e::AbstractVector{T}, n_nodes) where T
    d = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
    for I in 1:n_nodes
        G₃_I = Vec{3,T}(Tuple(scv.G₃_elem[I]))
        T₁_I = Vec{3,T}(Tuple(scv.T₁_elem[I]))
        T₂_I = Vec{3,T}(Tuple(scv.T₂_elem[I]))
        φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
        cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
        d_I = cosθ*G₃_I + sincθ*(φ₁*T₁_I + φ₂*T₂_I)
        d  += scv.N[I, qp]       * d_I
        d₁ += scv.dNdξ[I, qp][1] * d_I
        d₂ += scv.dNdξ[I, qp][2] * d_I
    end
    d, d₁, d₂
end

# Reference director at qp: d₀ = Σ N_I(qp) G₃_elem[I]. Used for shear reference subtraction.
# At u=0, director_field returns d = d₀, ensuring zero shear strain in the reference config.
@inline function reference_director(scv, qp, n_nodes)
    d₀ = zero(Vec{3,Float64})
    for I in 1:n_nodes
        d₀ += scv.N[I, qp] * scv.G₃_elem[I]
    end
    d₀
end

# Bending curvature change κ_αβ = ½(a_α·d,β + a_β·d,α) - B_αβ.
@inline function curvature_tensor(a₁, a₂, d₁, d₂, B)
    κ₁₁ = dot(a₁, d₁) - B[1,1]
    κ₁₂ = 0.5*(dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
    κ₂₂ = dot(a₂, d₂) - B[2,2]
    SymmetricTensor{2,2}((κ₁₁, κ₁₂, κ₂₂))
end

# Rodrigues director Jacobian ∂d_I/∂φ₁ and ∂d_I/∂φ₂ at a single node.
# Returns (sincθ, sccθ, dd₁, dd₂); sincθ/sccθ are reused in the tangent's geometric block.
@inline function rodrigues_jac(φ₁, φ₂, G₃, T₁, T₂)
    _, sincθ, sccθ = _cos_sinc_sincc_sq(φ₁*φ₁ + φ₂*φ₂)
    dd₁ = (sincθ + sccθ*φ₁^2)*T₁ + sccθ*φ₁*φ₂*T₂ - sincθ*φ₁*G₃
    dd₂ = sccθ*φ₁*φ₂*T₁ + (sincθ + sccθ*φ₂^2)*T₂ - sincθ*φ₂*G₃
    sincθ, sccθ, dd₁, dd₂
end

"""
    membrane_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love membrane residual. `u_e` is a flat vector of length 3·`n_nodes`: [``u_1``,``u_2``,``u_3``, ``\\cdots``].
"""
function membrane_residuals_KL!(re, scv::ShellCellValues, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, _ = kinematics(scv, qp, u_e)
        E = membrane_strain(a₁, a₂, A_metric)
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

Kirchhoff–Love membrane tangent. `u_e` is a flat vector of length 3·`n_nodes`: [``u_1``,``u_2``,``u_3``, ``\\cdots``].
"""
function membrane_tangent_KL!(ke, scv::ShellCellValues, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, _ = kinematics(scv, qp, u_e)
        E = membrane_strain(a₁, a₂, A_metric)
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
    bending_energy_KL(u_flat, scv::ShellCellValues, mat)

Computes the Kirchhoff–Love bending energy through the curvature change ``\\kappa_{\\alpha\\beta} = b_{\\alpha\\beta} - B_{\\alpha\\beta}``
(current minus reference second fundamental form).

**Arguments:**
* `u_flat`: a flat vector of length 3·`n_nodes` containing the displacement degrees of freedom of that element [``u_1``,``u_2``,``u_3``, ``\\cdots``].
* `scv`: a [`ShellCellValues`](@ref) object containing the precomputed shape function values and derivatives.
* `mat`: an instance of an `AbstractMaterial` describing the material properties.

**Note:**
To capture the full curvature change ``\\kappa``, quadratic elements are required (linear element only captures twist ``\\kappa_{12}``).
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
    membrane_energy_RM(u_flat, scv::ShellCellValues, mat)

Reissner–Mindlin membrane strain energy.
DOF layout: 5 DOFs per node — [``u_1``,``u_2``,``u_3``, ``\\varphi_1``, ``\\varphi_2``, ``\\cdots``] (flat vector of length 5·`n_nodes`).
Only the displacement DOFs (indices `5I-4:5I-2``) contribute to membrane energy.
"""
function membrane_energy_RM(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_flat, n_nodes)
        E = membrane_strain(a₁, a₂, scv.A_metric[qp])
        C = contravariant_elasticity(mat, scv.A_metric[qp])
        W += 0.5 * (E ⊡ C ⊡ E) * scv.detJdV[qp]
    end
    return W
end

"""
    membrane_residuals_RM_FD!(re, scv, u_e, mat)

Reissner–Mindlin membrane residual. `u_e` is a flat vector of length 5·`n_nodes`.
"""
function membrane_residuals_RM_FD!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> membrane_energy_RM(u, scv, mat), u_e)
end

"""
    membrane_residuals_RM!(re, scv, u_e, mat)

RM membrane residual: ``r_I = \\int N^{\\alpha\\beta} \\partial N_I^\\alpha a_\\beta \\, d\\Omega``.
Stress resultant rows ``P_\\alpha = N^{\\alpha\\beta} a_\\beta`` are precomputed once per QP.
"""
function membrane_residuals_RM!(re, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        E = membrane_strain(a₁, a₂, scv.A_metric[qp])
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
    membrane_tangent_RM_FD!(ke, scv, u_e, mat)

Reissner–Mindlin membrane tangent. `u_e` is a flat vector of length 5·`n_nodes`.
"""
function membrane_tangent_RM_FD!(ke, scv, u_e, mat)
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
    membrane_tangent_RM!(ke, scv, u_e, mat)

RM membrane tangent.
Material part: ``K^\\text{mat}_{IJ} = \\partial N_I^\\alpha \\partial N_J^\\delta M_{\\alpha\\delta}`` where
``M_{\\alpha\\delta} = C^{\\alpha\\beta\\gamma\\delta} a_\\beta\\otimes a_\\gamma``.
Geometric part: ``K^\\text{geo}_{IJ} = (\\partial N_I^\\alpha N^{\\alpha\\beta} \\partial N_J^\\beta) \\mathbb{h}_3``.
Both ``M_{\\alpha\\delta}`` and ``N`` are precomputed once per QP outside the node loops.
"""
function membrane_tangent_RM!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        E = membrane_strain(a₁, a₂, scv.A_metric[qp])
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
    bending_shear_energy_RM(u_flat, scv::ShellCellValues, mat)

Reissner–Mindlin bending + transverse shear strain energy.
DOF layout: 5 DOFs per node — [``u_1``,``u_2``,``u_3``, ``\\varphi_1``, ``\\varphi_2``, ``\\cdots``] (flat vector of length 5·`n_nodes`).

Director: ``d_I = G_3 + \\varphi_{1,I} T_1 + \\varphi_{2,I} T_2`` where ``G_3`` is the reference unit normal
and ``T_1``, ``T_2`` are reference tangents from `scv`.

Bending strain: ``\\kappa_{\\alpha\\beta} = \\frac{1}{2}(a_\\alpha \\cdot d_{,\\beta} + a_\\beta \\cdot d_{,\\alpha}) - B_{\\alpha\\beta}``

Transverse shear: ``\\gamma_\\alpha = a_\\alpha \\cdot d``

Shear correction factor ``\\kappa_s = 5/6`` is applied.
"""
function bending_shear_energy_RM(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    γ₁_k, γ₂_k = tying_shear_strains(scv.mitc, u_flat)
    G_sh = mat.E / (2*(1 + mat.ν))
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_flat, n_nodes)
        d, d₁, d₂ = director_field(scv, qp, u_flat, n_nodes)
        κ   = curvature_tensor(a₁, a₂, d₁, d₂, scv.B[qp])
        γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ₁_k, γ₂_k, scv.mitc)
        d₀  = reference_director(scv, qp, n_nodes)
        γ₁ -= dot(scv.A₁[qp], d₀); γ₂ -= dot(scv.A₂[qp], d₀)
        D    = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        Aup  = inv(scv.A_metric[qp])
        W_bend  = 0.5 * (κ ⊡ D ⊡ κ)
        W_shear = 0.5 * (5.0/6.0) * G_sh * mat.thickness *
                  (Aup[1,1]*γ₁^2 + 2*Aup[1,2]*γ₁*γ₂ + Aup[2,2]*γ₂^2)
        W += (W_bend + W_shear) * scv.detJdV[qp]
    end
    return W
end

"""
    bending_residuals_RM_FD!(re, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_residuals_RM_FD!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_shear_energy_RM(u, scv, mat), u_e)
end
"""
    bending_residuals_RM!(re, scv, u_e, mat)

RM bending + transverse shear residual, explicit index-notation form.

Displacement DOFs: ``r_I^u = (\\partial_1 N_I P^1 + \\partial_2 N_I P^2) \\, d\\Omega``
where ``P^\\alpha = M^{\\alpha\\beta} d_{,\\beta} + Q^\\alpha d``, ``M = D : \\kappa`` (bending moment),
``Q^\\alpha = \\kappa_s G t A^{\\alpha\\beta} \\gamma_\\beta``.

Rotation DOFs: ``r_{I,k}^\\varphi = F_I \\cdot (\\partial d_I/\\partial \\varphi_k) \\, d\\Omega``
where ``F_I = \\partial_1 N_I S^1 + \\partial_2 N_I S^2 + N_I (Q_1 a_1 + Q_2 a_2)``,
``S^\\alpha = M^{\\alpha\\beta} a_\\beta``.
"""
function bending_residuals_RM!(re, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    G_sh = mat.E / (2*(1 + mat.ν))
    γ₁_k, γ₂_k = tying_shear_strains(scv.mitc, u_e)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)
        κ   = curvature_tensor(a₁, a₂, d₁, d₂, scv.B[qp])
        γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ₁_k, γ₂_k, scv.mitc)
        d₀  = reference_director(scv, qp, n_nodes)
        γ₁ -= dot(scv.A₁[qp], d₀); γ₂ -= dot(scv.A₂[qp], d₀)
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
            _, _, dd_dφ₁, dd_dφ₂ = rodrigues_jac(φ₁, φ₂, scv.G₃_elem[I], scv.T₁_elem[I], scv.T₂_elem[I])
            re[5I-1] += dot(F_I, dd_dφ₁) * dΩ
            re[5I  ] += dot(F_I, dd_dφ₂) * dΩ
        end
    end
end

"""
    bending_tangent_RM_FD!(ke, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_tangent_RM_FD!(ke, scv, u, mat)
    ke .+= ForwardDiff.hessian(u -> bending_shear_energy_RM(u, scv, mat), u)
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)  [MITC dispatch]

Consistent RM bending + transverse shear tangent for MITC elements.
The MITC shear sensitivities ``\\partial\\gamma_\\alpha/\\partial u`` are obtained by differentiating through the
tying interpolation:

```math
\\frac{\\partial\\gamma_\\alpha(q)}{\\partial u_J}
  = \\sum_k h^\\alpha_{qk} \\frac{\\partial N_J}{\\partial\\xi_\\alpha}(\\xi_k)\\cdot d(\\xi_k)
```
```math
\\frac{\\partial\\gamma_\\alpha(q)}{\\partial\\varphi_{J,l}}
  = \\sum_k h^\\alpha_{qk}\\, N_J(\\xi_k)\\left(a_\\alpha(\\xi_k)\\cdot\\frac{\\partial d_J}{\\partial\\varphi_l}(\\xi_k)\\right)
```

where ``\\partial d_J/\\partial\\varphi_l(\\xi_k)`` is the Rodrigues Jacobian at node J evaluated with the reference
geometry ``(G_3, T_1, T_2)`` at tying point k.  This ensures exact consistency with
`bending_residuals_RM!` so Newton converges quadratically.

Bending (``\\kappa``) terms are unchanged from the NoMITC path — only shear (``Q``) terms differ.
"""
function bending_tangent_RM!(ke, scv::ShellCellValues{QR,IPG,IPS,FT,E,M}, u_e::AbstractVector{T}, mat) where {QR,IPG,IPS,FT<:AbstractFloat,E<:AbstractStrainMeasure,M<:MITC,T}
    mitc    = scv.mitc
    n_nodes = getnbasefunctions(scv.ip_shape)
    Nt      = length(mitc.ξ_tie_1)
    G_sh    = mat.E / (2*(1 + mat.ν))
    cs      = T(5//6) * G_sh * mat.thickness

    # Deformed tangents and interpolated directors at tying points (QP-independent).
    a₁_tie = Vector{Vec{3,T}}(undef, Nt)
    a₂_tie = Vector{Vec{3,T}}(undef, Nt)
    d_tie1 = Vector{Vec{3,T}}(undef, Nt)
    d_tie2 = Vector{Vec{3,T}}(undef, Nt)
    for k in 1:Nt
        Δa₁ = zero(Vec{3,T}); d_k1 = zero(Vec{3,T})
        for I in 1:n_nodes
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * mitc.dNdξ_tie_1[I,k][1]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            G₃_I = Vec{3,T}(mitc.G₃_node[I]); T₁_I = Vec{3,T}(mitc.T₁_node[I]); T₂_I = Vec{3,T}(mitc.T₂_node[I])
            d_k1 += mitc.N_tie_1[I,k] * (cosθ*G₃_I + sincθ*(φ₁*T₁_I + φ₂*T₂_I))
        end
        a₁_tie[k] = Vec{3,T}(mitc.A₁_tie_1[k]) + Δa₁
        d_tie1[k]  = d_k1
    end
    for k in 1:Nt
        Δa₂ = zero(Vec{3,T}); d_k2 = zero(Vec{3,T})
        for I in 1:n_nodes
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₂ += u_I * mitc.dNdξ_tie_2[I,k][2]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
            cosθ, sincθ = _cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            G₃_I = Vec{3,T}(mitc.G₃_node[I]); T₁_I = Vec{3,T}(mitc.T₁_node[I]); T₂_I = Vec{3,T}(mitc.T₂_node[I])
            d_k2 += mitc.N_tie_2[I,k] * (cosθ*G₃_I + sincθ*(φ₁*T₁_I + φ₂*T₂_I))
        end
        a₂_tie[k] = Vec{3,T}(mitc.A₂_tie_2[k]) + Δa₂
        d_tie2[k]  = d_k2
    end

    # Rodrigues Jacobians at tying points per node: ∂d_J/∂φ_l evaluated at tying-point geometry.
    dd1_t1 = Matrix{Vec{3,T}}(undef, n_nodes, Nt)
    dd2_t1 = Matrix{Vec{3,T}}(undef, n_nodes, Nt)
    dd1_t2 = Matrix{Vec{3,T}}(undef, n_nodes, Nt)
    dd2_t2 = Matrix{Vec{3,T}}(undef, n_nodes, Nt)
    for J in 1:n_nodes
        φ₁_J = u_e[5J-1]; φ₂_J = u_e[5J]
        G₃_J = Vec{3,T}(mitc.G₃_node[J]); T₁_J = Vec{3,T}(mitc.T₁_node[J]); T₂_J = Vec{3,T}(mitc.T₂_node[J])
        _, _, dd1, dd2 = rodrigues_jac(φ₁_J, φ₂_J, G₃_J, T₁_J, T₂_J)
        for k in 1:Nt
            dd1_t1[J,k] = dd1; dd2_t1[J,k] = dd2
        end
        for k in 1:Nt
            dd1_t2[J,k] = dd1; dd2_t2[J,k] = dd2
        end
    end

    γ₁_k, γ₂_k = tying_shear_strains(mitc, u_e)

    # Per-QP workspace: MITC shear sensitivities
    Bγ₁u  = Vector{Vec{3,T}}(undef, n_nodes)
    Bγ₂u  = Vector{Vec{3,T}}(undef, n_nodes)
    Bγ₁φ1 = Vector{T}(undef, n_nodes)
    Bγ₁φ2 = Vector{T}(undef, n_nodes)
    Bγ₂φ1 = Vector{T}(undef, n_nodes)
    Bγ₂φ2 = Vector{T}(undef, n_nodes)

    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)
        κ   = curvature_tensor(a₁, a₂, d₁, d₂, scv.B[qp])
        γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ₁_k, γ₂_k, mitc)
        d₀  = reference_director(scv, qp, n_nodes)
        γ₁ -= dot(scv.A₁[qp], d₀); γ₂ -= dot(scv.A₂[qp], d₀)
        D   = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        Mb  = D ⊡ κ
        Aup = inv(scv.A_metric[qp])
        Q₁  = cs*(Aup[1,1]*γ₁ + Aup[1,2]*γ₂)
        Q₂  = cs*(Aup[2,1]*γ₁ + Aup[2,2]*γ₂)
        S¹  = Mb[1,1]*a₁ + Mb[1,2]*a₂
        S²  = Mb[2,1]*a₁ + Mb[2,2]*a₂
        L₁₁, L₁₂, L₂₂ = frame_stiffness(D, d₁, d₂)
        dΩ  = scv.detJdV[qp]

        # MITC-consistent shear sensitivities for each node J at this QP
        for J in 1:n_nodes
            Bγ₁u[J]  = zero(Vec{3,T}); Bγ₂u[J]  = zero(Vec{3,T})
            Bγ₁φ1[J] = zero(T);        Bγ₁φ2[J] = zero(T)
            Bγ₂φ1[J] = zero(T);        Bγ₂φ2[J] = zero(T)
            @inbounds for k in 1:Nt
                h1 = mitc.h_tie_1[qp,k]; h2 = mitc.h_tie_2[qp,k]
                Bγ₁u[J]  += h1 * mitc.dNdξ_tie_1[J,k][1] * d_tie1[k]
                Bγ₂u[J]  += h2 * mitc.dNdξ_tie_2[J,k][2] * d_tie2[k]
                Bγ₁φ1[J] += h1 * mitc.N_tie_1[J,k] * dot(a₁_tie[k], dd1_t1[J,k])
                Bγ₁φ2[J] += h1 * mitc.N_tie_1[J,k] * dot(a₁_tie[k], dd2_t1[J,k])
                Bγ₂φ1[J] += h2 * mitc.N_tie_2[J,k] * dot(a₂_tie[k], dd1_t2[J,k])
                Bγ₂φ2[J] += h2 * mitc.N_tie_2[J,k] * dot(a₂_tie[k], dd2_t2[J,k])
            end
        end

        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]; NI = scv.N[I, qp]
            F_I = ∂NI1*S¹ + ∂NI2*S² + NI*(Q₁*a₁ + Q₂*a₂)
            φ₁_I = u_e[5I-1]; φ₂_I = u_e[5I]
            s_I, sc_I, dd_I1, dd_I2 = rodrigues_jac(φ₁_I, φ₂_I, scv.G₃_elem[I], scv.T₁_elem[I], scv.T₂_elem[I])
            θ²_I = φ₁_I^2 + φ₂_I^2
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]; NJ = scv.N[J, qp]
                # uu block: bending frame stiffness (unchanged) + MITC-consistent shear
                K_bend  = ∂NI1*∂NJ1*L₁₁ + ∂NI1*∂NJ2*L₁₂ + ∂NI2*∂NJ1*transpose(L₁₂) + ∂NI2*∂NJ2*L₂₂
                K_shear = cs*(Aup[1,1]*(Bγ₁u[I]⊗Bγ₁u[J]) + Aup[1,2]*(Bγ₁u[I]⊗Bγ₂u[J]) +
                              Aup[2,1]*(Bγ₂u[I]⊗Bγ₁u[J]) + Aup[2,2]*(Bγ₂u[I]⊗Bγ₂u[J]))
                @views ke[5I-4:5I-2, 5J-4:5J-2] .+= (K_bend + K_shear) * dΩ
                # uφ and φφ material blocks
                φ₁_J = u_e[5J-1]; φ₂_J = u_e[5J]
                _, _, dd_J1, dd_J2 = rodrigues_jac(φ₁_J, φ₂_J, scv.G₃_elem[J], scv.T₁_elem[J], scv.T₂_elem[J])
                g_IJ = ∂NI1*(Mb[1,1]*∂NJ1+Mb[1,2]*∂NJ2) + ∂NI2*(Mb[2,1]*∂NJ1+Mb[2,2]*∂NJ2)
                q_I  = ∂NI1*Q₁ + ∂NI2*Q₂
                for (l, dd_Jl, Bγ₁φl, Bγ₂φl) in ((1, dd_J1, Bγ₁φ1[J], Bγ₂φ1[J]),
                                                    (2, dd_J2, Bγ₁φ2[J], Bγ₂φ2[J]))
                    c₁ = dot(a₁, dd_Jl); c₂ = dot(a₂, dd_Jl)  # bending δκ: QP geometry
                    δκ = SymmetricTensor{2,2,T}((∂NJ1*c₁, 0.5*(∂NJ1*c₂+∂NJ2*c₁), ∂NJ2*c₂))
                    δM = D ⊡ δκ
                    # MITC-consistent shear for uφ: Bγ_φl contains N_tie factor, no NJ
                    δQ₁ = cs*(Aup[1,1]*Bγ₁φl + Aup[1,2]*Bγ₂φl)
                    δQ₂ = cs*(Aup[2,1]*Bγ₁φl + Aup[2,2]*Bγ₂φl)
                    col = 5J - 2 + l
                    v_bend  = ∂NI1*(δM[1,1]*d₁+δM[1,2]*d₂) + ∂NI2*(δM[2,1]*d₁+δM[2,2]*d₂)
                    v_shear = (∂NI1*δQ₁ + ∂NI2*δQ₂) * d    # no NJ: Bγ already accounts for it
                    v_dir   = (g_IJ + q_I*NJ) * dd_Jl       # Q·δd: director at QP
                    @views ke[5I-4:5I-2, col] .+= (v_bend + v_shear + v_dir) * dΩ
                    @views ke[col, 5I-4:5I-2] .+= (v_bend + v_shear + v_dir) * dΩ  # φu by symmetry
                    δS¹  = δM[1,1]*a₁ + δM[1,2]*a₂
                    δS²  = δM[2,1]*a₁ + δM[2,2]*a₂
                    δF_I = ∂NI1*δS¹ + ∂NI2*δS² + NI*(δQ₁*a₁ + δQ₂*a₂)  # no NJ: in Bγ
                    ke[5I-1, col] += dot(δF_I, dd_I1) * dΩ
                    ke[5I,   col] += dot(δF_I, dd_I2) * dΩ
                end
            end
            # φφ geometric part (J=I): F_I uses MITC Q₁,Q₂ — already correct
            G₃_I = scv.G₃_elem[I]; T₁_I = scv.T₁_elem[I]; T₂_I = scv.T₂_elem[I]
            sccc_I = θ²_I < 1e-6 ? one(T)/15 : (-s_I - 3sc_I)/θ²_I
            d2_11 = (3sc_I*φ₁_I + sccc_I*φ₁_I^3)*T₁_I + (sc_I + sccc_I*φ₁_I^2)*φ₂_I*T₂_I - (s_I + sc_I*φ₁_I^2)*G₃_I
            d2_12 = (sc_I + sccc_I*φ₁_I^2)*φ₂_I*T₁_I + (sc_I + sccc_I*φ₂_I^2)*φ₁_I*T₂_I - sc_I*φ₁_I*φ₂_I*G₃_I
            d2_22 = (sc_I + sccc_I*φ₂_I^2)*φ₁_I*T₁_I + (3sc_I*φ₂_I + sccc_I*φ₂_I^3)*T₂_I - (s_I + sc_I*φ₂_I^2)*G₃_I
            ke[5I-1, 5I-1] += dot(F_I, d2_11) * dΩ
            ke[5I-1, 5I  ] += dot(F_I, d2_12) * dΩ
            ke[5I,   5I-1] += dot(F_I, d2_12) * dΩ
            ke[5I,   5I  ] += dot(F_I, d2_22) * dΩ
        end
    end
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)

RM bending and transverse shear tangent, explicit index-notation form. Four blocks per (I,J) pair:

- **uu** (3×3): ``\\partial_\\alpha N_I \\partial_\\gamma N_J (D^{\\alpha\\beta\\gamma\\delta} d_{,\\beta} \\otimes d_{,\\delta}) + q_{IJ}(d \\otimes d)`` — frame stiffness with ``d_1, d_2``.
- **uφ** (3×2): ``\\partial_\\alpha N_I [\\delta M^{\\alpha\\beta} d_{,\\beta} + \\delta Q^\\alpha N_J d] + (g_{IJ}+q_I N_J) \\mathrm{dd}_{Jl}``.
- **φu** (2×3): filled by transposing the **uφ** block for (I,J) into the (J,I) position.
- **φφ** (2×2): material part ``\\partial F_I/\\partial\\varphi_{l,J} \\cdot \\mathrm{dd}_{Ik}`` + geometric part ``F_I \\cdot \\partial^2 d_I/\\partial\\varphi_k\\partial\\varphi_l`` (J=I only).
"""
function bending_tangent_RM!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    G_sh = mat.E / (2*(1 + mat.ν))
    γ₁_k, γ₂_k = tying_shear_strains(scv.mitc, u_e)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
        d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)
        κ   = curvature_tensor(a₁, a₂, d₁, d₂, scv.B[qp])
        γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ₁_k, γ₂_k, scv.mitc)
        d₀  = reference_director(scv, qp, n_nodes)
        γ₁ -= dot(scv.A₁[qp], d₀); γ₂ -= dot(scv.A₂[qp], d₀)
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
            s_I, sc_I, dd_I1, dd_I2 = rodrigues_jac(φ₁_I, φ₂_I, scv.G₃_elem[I], scv.T₁_elem[I], scv.T₂_elem[I])
            θ²_I  = φ₁_I^2 + φ₂_I^2
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]; NJ = scv.N[J, qp]
                # uu block: frame_stiffness with d₁,d₂ + shear term
                q_IJ = cs*(∂NI1*(Aup[1,1]*∂NJ1+Aup[1,2]*∂NJ2) + ∂NI2*(Aup[2,1]*∂NJ1+Aup[2,2]*∂NJ2))
                K_uu = ∂NI1*∂NJ1*L₁₁ + ∂NI1*∂NJ2*L₁₂ + ∂NI2*∂NJ1*transpose(L₁₂) + ∂NI2*∂NJ2*L₂₂ + q_IJ*(d⊗d)
                @views ke[5I-4:5I-2, 5J-4:5J-2] .+= K_uu * dΩ
                # uφ and φφ material blocks: loop over rotation directions l=1,2 of node J
                φ₁_J = u_e[5J-1]; φ₂_J = u_e[5J]
                _, _, dd_J1, dd_J2 = rodrigues_jac(φ₁_J, φ₂_J, scv.G₃_elem[J], scv.T₁_elem[J], scv.T₂_elem[J])
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
            G₃_I = scv.G₃_elem[I]; T₁_I = scv.T₁_elem[I]; T₂_I = scv.T₂_elem[I]
            sccc_I = θ²_I < 1e-6 ? one(T)/15 : (-s_I - 3sc_I)/θ²_I
            d2_11 = (3sc_I*φ₁_I + sccc_I*φ₁_I^3)*T₁_I + (sc_I + sccc_I*φ₁_I^2)*φ₂_I*T₂_I - (s_I + sc_I*φ₁_I^2)*G₃_I
            d2_12 = (sc_I + sccc_I*φ₁_I^2)*φ₂_I*T₁_I + (sc_I + sccc_I*φ₂_I^2)*φ₁_I*T₂_I - sc_I*φ₁_I*φ₂_I*G₃_I
            d2_22 = (sc_I + sccc_I*φ₂_I^2)*φ₁_I*T₁_I + (3sc_I*φ₂_I + sccc_I*φ₂_I^3)*T₂_I - (s_I + sc_I*φ₂_I^2)*G₃_I
            ke[5I-1, 5I-1] += dot(F_I, d2_11) * dΩ
            ke[5I-1, 5I  ] += dot(F_I, d2_12) * dΩ
            ke[5I,   5I-1] += dot(F_I, d2_12) * dΩ
            ke[5I,   5I  ] += dot(F_I, d2_22) * dΩ
        end
    end
end

"""
    mass_matrix!(me, scv::ShellCellValues, ρ::T, mat)

Mass matrix for embedded shell elements (2D mesh in 3D). Only translational DOFs contribute to kinetic energy.
"""
function mass_matrix!(me, scv::ShellCellValues, ρ::T, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    ρt = ρ * mat.thickness
    for qp in 1:getnquadpoints(scv)
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes, J in 1:n_nodes
            m = scv.N[I, qp] * scv.N[J, qp] * ρt * dΩ
            me[5I-4, 5J-4] += m
            me[5I-3, 5J-3] += m
            me[5I-2, 5J-2] += m
        end
    end
end

# ∂ξ/∂t for the mapping t ∈ [-1,1] → 2D cell reference coord along each facet.
# Derived from ξ(t) = 0.5*(1-t)*A + 0.5*(1+t)*B (vertices A, B of facet in ref space).
# RefQuadrilateral vertices: (-1,-1),(1,-1),(1,1),(-1,1); all facets are axis-aligned.
# RefTriangle vertices: (0,0),(1,0),(0,1); facet 2 (hypotenuse) has mixed direction.
@inline _facet_dxi(::Lagrange{RefQuadrilateral}, f::Int) = f ∈ (1,3) ? Vec{2}((1.0, 0.0)) : Vec{2}((0.0, 1.0))
@inline _facet_dxi(::Lagrange{RefTriangle},      f::Int) =
    f == 1 ? Vec{2}(( 0.5,  0.0)) :
    f == 2 ? Vec{2}((-0.5,  0.5)) :
             Vec{2}(( 0.0, -0.5))

"""
Assemble external traction into force vector f for embedded shell elements (2D mesh in 3D).
`traction` is either a `Vec{3}` (uniform) or a callable `x::Vec{3} -> Vec{3}`.
Uses a `FacetQuadratureRule` and computes the edge length element directly from 3D node positions,
bypassing the sdim mismatch that prevents standard `FacetValues` from working on embedded meshes.
Works for RefQuadrilateral and RefTriangle of any interpolation order.
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
        dxi      = _facet_dxi(ip, facet_nr)   # ∂ξ/∂t in reference coords
        for (ξ, w) in zip(qr_f.points, qr_f.weights)
            xp = zero(Vec{3,Float64})
            Jt = zero(Vec{3,Float64})  # physical tangent: J * dxi
            for I in 1:n_base
                dN, N = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                xp += N * x[I]
                Jt += dot(dN, dxi) * x[I]
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
    assemble_pressure!(re, scv, u_e, p)

Follower pressure residual for embedded shell elements.
``\\mathrm{detJdV}[q] = \\|A_1 \\times A_2\\| \\cdot w`` (reference area times quadrature weight).
``\\mathrm{cross}(a_1, a_2)`` has magnitude ``\\|a_1 \\times a_2\\|`` (current area per parametric area).
"""
# Follower pressure residual
function assemble_pressure!(re, scv::ShellCellValues, u_e::AbstractVector{T}, p) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        w = scv.qr.weights[qp]
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
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
    assemble_pressure_tangent!(ke, scv, u_e, p)

Load-stiffness ``K_\\text{pres} = \\partial F_p/\\partial u``. Follower pressure ``n = a_1 \\times a_2``
depends on `u` through ``a_1, a_2``.

```math
\\frac{\\partial n}{\\partial u_J} = \\partial_1 N_J (e_l \\times a_2) + \\partial_2 N_J (a_1 \\times e_l)
 = \\partial_1 N_J (-\\mathrm{spin}(a_2)) + \\partial_2 N_J\\,\\mathrm{spin}(a_1)
```
```math
K_{IJ} = p N_I w \\bigl[\\partial_1 N_J (-\\mathrm{spin}(a_2)) + \\partial_2 N_J\\,\\mathrm{spin}(a_1)\\bigr]
```
(displacement-displacement block only).
"""
function assemble_pressure_tangent!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, p) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        w = scv.qr.weights[qp]
        a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
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