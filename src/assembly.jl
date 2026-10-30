using ForwardDiff

# Compute (cos(√θ²), sin(√θ²)/√θ²) from θ² = φ₁²+φ₂² without calling sqrt at 0.
# ForwardDiff-safe: the branch is on the primal value only, so dual perturbations
# flow through whichever branch is active.  In the else branch √θ² > 0 so the
# sqrt gradient (1/2√θ²) is finite.
@inline function cos_sinc_sq(θ²::T) where T
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
@inline function cos_sinc_sincc_sq(θ²::T) where T
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
@inline membrane_strain(a₁, a₂, A_metric_ref) = 0.5 * (SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂))) - A_metric_ref)

# Interpolated director d and covariant derivatives d₁ = d,₁, d₂ = d,₂ at qp.
# Rodrigues: d_I = cos|φ|·G₃_I + sinc|φ|·(φ₁T₁_I+φ₂T₂_I), |d_I| = 1 (per-node frames).
@inline function director_field(scv, qp, u_e::AbstractVector{T}, n_nodes) where T
    d = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
    for I in 1:n_nodes
        # get sotred frame data for node I
        G₃_I = scv.G₃_elem[I]
        T₁_I = scv.T₁_elem[I]
        T₂_I = scv.T₂_elem[I]
        # get rotation dofs for node I
        φ₁ = u_e[5I-1]; φ₂ = u_e[5I]
        cosθ, sincθ = cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
        # and compute Rodrigues interpolation
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

# Bending curvature change κ_αβ = ½(a_α·d,β + a_β·d,α) − B₀_αβ, with B₀ the
# reference director-gradient curvature (see reference_director_curvature!) —
# NOT the patch curvature B, which would leave a mesh-persistent reference
# bending strain on curved or warped elements.
@inline function curvature_tensor(a₁, a₂, d₁, d₂, B)
    κ₁₁ = dot(a₁, d₁) - B[1,1]
    κ₁₂ = 0.5*(dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
    κ₂₂ = dot(a₂, d₂) - B[2,2]
    SymmetricTensor{2,2}((κ₁₁, κ₁₂, κ₂₂))
end

# Rodrigues director Jacobian ∂d_I/∂φ₁ and ∂d_I/∂φ₂ at a single node.
# Returns (sincθ, sccθ, dd₁, dd₂); sincθ/sccθ are reused in the tangent's geometric block.
@inline function rodrigues_jac(φ₁, φ₂, G₃, T₁, T₂)
    _, sincθ, sccθ = cos_sinc_sincc_sq(φ₁*φ₁ + φ₂*φ₂)
    dd₁ = (sincθ + sccθ*φ₁^2)*T₁ + sccθ*φ₁*φ₂*T₂ - sincθ*φ₁*G₃
    dd₂ = sccθ*φ₁*φ₂*T₁ + (sincθ + sccθ*φ₂^2)*T₂ - sincθ*φ₂*G₃
    sincθ, sccθ, dd₁, dd₂
end

# Second derivatives of the Rodrigues director d(φ) w.r.t. (φ₁,φ₂): returns
# (∂²d/∂φ₁², ∂²d/∂φ₁∂φ₂, ∂²d/∂φ₂²). `s`=sinc, `sc`=scc, `sccc`=(-s-3sc)/θ² (Taylor 1/15).
@inline function rodrigues_hess(s, sc, sccc, φ₁, φ₂, G₃, T₁, T₂)
    d2_11 = (3sc*φ₁ + sccc*φ₁^3)*T₁ + (sc + sccc*φ₁^2)*φ₂*T₂ - (s + sc*φ₁^2)*G₃
    d2_12 = (sc + sccc*φ₁^2)*φ₂*T₁ + (sc + sccc*φ₂^2)*φ₁*T₂ - sc*φ₁*φ₂*G₃
    d2_22 = (sc + sccc*φ₂^2)*φ₁*T₁ + (3sc*φ₂ + sccc*φ₂^3)*T₂ - (s + sc*φ₂^2)*G₃
    d2_11, d2_12, d2_22
end

"""
    membrane_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love membrane residual. `u_e` is a flat vector of length 3·`n_nodes`: [``u_1``,``u_2``,``u_3``, ``\\cdots``].
"""
function membrane_residuals_KL!(re, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for i in 1:n_nodes
            ui = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
            Δa₁ += ui * scv.dNdξ[i, qp][1]; Δa₂ += ui * scv.dNdξ[i, qp][2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        N, _ = membrane_stress_and_tangent(mat, c_ms, scv.A_metric[qp],
                   scv.A₁[qp], scv.A₂[qp], scv.G₃[qp])
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
function membrane_tangent_KL!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for i in 1:n_nodes
            ui = Vec{3,T}((u_e[3i-2], u_e[3i-1], u_e[3i]))
            Δa₁ += ui * scv.dNdξ[i, qp][1]; Δa₂ += ui * scv.dNdξ[i, qp][2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        N, C = membrane_stress_and_tangent(mat, c_ms, scv.A_metric[qp],
                   scv.A₁[qp], scv.A₂[qp], scv.G₃[qp])
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


# Per-QP bending energy for KL, dispatched on material type.
# For LinearElastic: ½ κ:D:κ  (D evaluated at reference metric — safe, A_metric is plain Float64).
@inline function bending_kl_qp_energy(mat::LinearElastic, c_ms, κ, A_metric, A₁, A₂, G₃)
    D, _ = bending_and_shear_stiffness(mat, c_ms, A_metric)
    0.5 * (κ ⊡ D ⊡ κ)
end

# Hyperelastic: through-thickness Gauss quadrature in the physical Cartesian frame.
# Subtracts midsurface term to isolate the bending contribution (KL has no shear).
@inline function bending_kl_qp_energy(mat::Hyperelastic, c_ms::SymmetricTensor{2,2,T},
                                       κ, A_metric, A₁, A₂, G₃) where T
    det_A = det(A_metric)
    Jinv  = inv(J_ref(A₁, A₂, G₃))
    ξ3s, w3s = gauss3_thickness(mat.thickness)
    W_ms = W_phys(mat, c_ms, det_A, Jinv)
    W = zero(T)
    for (ξ3, w3) in zip(ξ3s, w3s)
        W += W_phys(mat, c_ms + 2*ξ3*κ, det_A, Jinv) * w3
    end
    return W - mat.thickness * W_ms
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
    W  = zero(T)
    for qp in 1:getnquadpoints(scv)
        A₁q = scv.A₁[qp]; A₂q = scv.A₂[qp]
        a₁  = A₁q; a₂  = A₂q
        a₁₁ = Vec{3,T}(Tuple(scv.A₁₁[qp])); a₁₂ = Vec{3,T}(Tuple(scv.A₁₂[qp])); a₂₂ = Vec{3,T}(Tuple(scv.A₂₂[qp]))
        for I in 1:n_nodes
            UI  = u_e[I]; dN  = scv.dNdξ[I, qp]; d2N = scv.d2Ndξ2[I, qp]
            a₁  += UI * dN[1];     a₂  += UI * dN[2]
            a₁₁ += UI * d2N[1,1]; a₁₂ += UI * d2N[1,2]; a₂₂ += UI * d2N[2,2]
        end
        n_n  = (a₁ × a₂) / norm(a₁ × a₂)
        b    = SymmetricTensor{2,2,T}((dot(a₁₁,n_n), dot(a₁₂,n_n), dot(a₂₂,n_n)))
        κ    = b - scv.B[qp]
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        W   += bending_kl_qp_energy(mat, c_ms, κ, scv.A_metric[qp], A₁q, A₂q, scv.G₃[qp]) * scv.detJdV[qp]
    end
    return W
end

"""
    bending_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love bending residual via `ForwardDiff.gradient` of [`bending_energy_KL`](@ref).
`u_e` is a flat vector of length 3·`n_nodes`: [``u_1``,``u_2``,``u_3``, ``\\cdots``].
"""
function bending_residuals_KL!(re, scv::ShellCellValues, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy_KL(u, scv, mat), u_e)
end

"""
    bending_tangent_KL!(ke, scv, u_e, mat)

Kirchhoff–Love bending tangent via `ForwardDiff.hessian` of [`bending_energy_KL`](@ref).
`u_e` is a flat vector of length 3·`n_nodes`: [``u_1``,``u_2``,``u_3``, ``\\cdots``].
"""
function bending_tangent_KL!(ke, scv::ShellCellValues, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy_KL(u, scv, mat), u_e)
end

"""
    membrane_residuals_RM!(re, scv, u_e, mat)

RM membrane residual: ``r_I = \\int N^{\\alpha\\beta} \\partial N_I^\\alpha a_\\beta \\, d\\Omega``.
Stress resultant rows ``P_\\alpha = N^{\\alpha\\beta} a_\\beta`` are precomputed once per QP.
"""
membrane_residuals_RM!(re, scv::ShellCellValues, u_e, mat) = rm_element!(nothing, re, scv, u_e, mat, Val(:membrane))

@inline function membrane_residual_qp!(re, scv, qp, a₁, a₂, N)
    P₁ = N[1,1]*a₁ + N[1,2]*a₂
    P₂ = N[2,1]*a₁ + N[2,2]*a₂
    dΩ = scv.detJdV[qp]
    for I in 1:getnbasefunctions(scv.ip_shape)
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        @views re[5I-4:5I-2] .+= (∂NI1*P₁ + ∂NI2*P₂) * dΩ
    end
end


# Per-QP strain energy density for the RM element, dispatched on material type.
# LinearElastic: closed-form quadratic forms with κ_s = 5/6 shear correction.
# Hyperelastic: 3-point through-thickness Gauss quadrature of W(C(ξ₃)).
@inline function rm_qp_energy(mat::LinearElastic, c_ms::SymmetricTensor{2,2,T},
                              κ, γ₁, γ₂, A_metric, A₁, A₂, G₃) where T
    N, _ = membrane_stress_and_tangent(mat, c_ms, A_metric)
    D, Cs = bending_and_shear_stiffness(mat, c_ms, A_metric)
    E = (c_ms - A_metric) / 2
    return 0.5*(N ⊡ E) + 0.5*(κ ⊡ D ⊡ κ) + 0.5*(Cs[1,1]*γ₁^2 + 2*Cs[1,2]*γ₁*γ₂ + Cs[2,2]*γ₂^2)
end

@inline function rm_qp_energy(mat::Hyperelastic, c_ms::SymmetricTensor{2,2,T},
                              κ, γ₁, γ₂, A_metric, A₁, A₂, G₃) where T
    det_A = det(A_metric)
    Jinv  = inv(J_ref(A₁, A₂, G₃))
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
DOF layout: 5 DOFs per node — [u₁,u₂,u₃,φ₁,φ₂,…].
Material dispatch happens at the per-QP level via `rm_qp_energy`.
"""
function energy_RM(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    γ_k = tying_shear_strains(scv.mitc, u_flat)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂ = covariant_basis(scv, qp, u_flat, n_nodes)
        d, d₁, d₂ = director_field(scv, qp, u_flat, n_nodes)
        κ   = curvature_tensor(a₁, a₂, d₁, d₂, scv.B₀[qp])
        γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ_k, scv.mitc)
        d₀  = reference_director(scv, qp, n_nodes)
        r₁, r₂ = reference_shear_offset(scv.A₁[qp], scv.A₂[qp], d₀, scv.mitc)
        γ₁ -= r₁; γ₂ -= r₂
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        W += rm_qp_energy(mat, c_ms, κ, γ₁, γ₂, scv.A_metric[qp],
                          scv.A₁[qp], scv.A₂[qp], scv.G₃[qp]) * scv.detJdV[qp]
    end
    return W
end

"""
    residuals_RM_FD!(re, scv, u_e, mat)

Reissner–Mindlin total residual (membrane + bending + shear) via ForwardDiff.gradient of `energy_RM`.
`u_e` is a flat vector of length 5·`n_nodes`.
"""
function residuals_RM_FD!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> energy_RM(u, scv, mat), u_e)
end

"""
    tangent_RM_FD!(ke, scv, u_e, mat)

Reissner–Mindlin total tangent (membrane + bending + shear) via ForwardDiff.hessian of `energy_RM`.
`u_e` is a flat vector of length 5·`n_nodes`.
"""
function tangent_RM_FD!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> energy_RM(u, scv, mat), u_e)
end

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
membrane_tangent_RM!(ke, scv::ShellCellValues, u_e, mat) = rm_element!(ke, nothing, scv, u_e, mat, Val(:membrane))

function membrane_tangent_qp!(ke, scv, qp, a₁, a₂, N, C)
    M₁₁, M₁₂, M₂₂ = frame_stiffness(C, a₁, a₂)
    dΩ = scv.detJdV[qp]
    n_nodes = getnbasefunctions(scv.ip_shape)
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

# 3-point Gauss-Legendre on [-t/2, t/2]: exact for polynomials up to degree 5 in ξ₃.
@inline function gauss3_thickness(t)
    h = t / 2; r = sqrt(3.0/5.0)
    return (-h*r, 0.0, h*r), (5.0/9.0*h, 8.0/9.0*h, 5.0/9.0*h)
end


# Deformed state at qp for the RM bending kernels: covariant basis, director d and its
# derivatives d₁, d₂, curvature change κ, and shear strains γ measured from the reference.
@inline function rm_kinematics(scv, qp, u_e, γ_k, n_nodes)
    a₁, a₂ = covariant_basis(scv, qp, u_e, n_nodes)
    d, d₁, d₂ = director_field(scv, qp, u_e, n_nodes)
    κ = curvature_tensor(a₁, a₂, d₁, d₂, scv.B₀[qp])
    γ₁, γ₂ = shear_strains(a₁, a₂, d, qp, γ_k, scv.mitc)
    r₁, r₂ = reference_shear_offset(scv.A₁[qp], scv.A₂[qp], reference_director(scv, qp, n_nodes), scv.mitc)
    a₁, a₂, d, d₁, d₂, κ, γ₁ - r₁, γ₂ - r₂
end

# Per-node Rodrigues data (sinc, scc, ∂d_I/∂φ₁, ∂d_I/∂φ₂), independent of the quadrature point.
# Stack-allocated tuples keep the kernels allocation-free and ForwardDiff-safe.
@inline function nodal_rotations(scv, u_e, ::Val{NN}) where NN
    ntuple(Val(NN)) do I
        rodrigues_jac(u_e[5I-1], u_e[5I], scv.G₃_elem[I], scv.T₁_elem[I], scv.T₂_elem[I])
    end
end

# Deformed tangent and interpolated director (a_k, d_k) at every MITC tying point.
shear_tying_state(::NoMITC, u_e) = nothing
function shear_tying_state(mitc::MITC{NN,MM}, u_e::AbstractVector{T}) where {NN,MM,T}
    ntuple(Val(MM)) do k
        Δa = zero(Vec{3,T}); dk = zero(Vec{3,T})
        @inbounds for I in 1:NN
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa += u_I * mitc.dN_tie[I,k]
            φ₁ = u_e[5I-1]; φ₂ = u_e[5I]; cosθ, sincθ = cos_sinc_sq(φ₁*φ₁ + φ₂*φ₂)
            dk += mitc.N_tie[I,k] * (cosθ*mitc.G₃_node[I] + sincθ*(φ₁*mitc.T₁_node[I] + φ₂*mitc.T₂_node[I]))
        end
        (mitc.A_tie[k] + Δa, dk)
    end
end

# Shear-strain operators at qp: for every node J, the sensitivities ∂γ_α/∂(u_J, φ_J) as
# 5-tuples (u₁,u₂,u₃,φ₁,φ₂), one per component α.
#
# Without MITC, γ_α = a_α·d at the quadrature point, so ∂γ_α/∂u_J = ∂_αN_J d and
# ∂γ_α/∂φ_J = N_J a_α·∂d_J/∂φ. With MITC, γ_α is the tying interpolation Σ_k h^α_k a_{α_k}·d at
# the tying points, and the same derivatives are taken there and interpolated. The nodal
# frames of the MITC object are those of `scv`, so both share the Rodrigues Jacobians `rot`.
@inline function shear_operators(scv, qp, kin, rot, ::Nothing, ::Val{NN}) where NN
    a₁, a₂, d = kin
    ntuple(Val(NN)) do J
        ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]; NJ = scv.N[J, qp]
        dd1 = rot[J][3]; dd2 = rot[J][4]
        u1 = ∂NJ1*d; u2 = ∂NJ2*d
        ((u1[1], u1[2], u1[3], NJ*dot(a₁, dd1), NJ*dot(a₁, dd2)),
         (u2[1], u2[2], u2[3], NJ*dot(a₂, dd1), NJ*dot(a₂, dd2)))
    end
end
@inline function shear_operators(scv, qp, kin, rot, tie::Tuple, ::Val{NN}) where NN
    mitc = scv.mitc
    T = eltype(kin[1])
    ntuple(Val(NN)) do J
        dd1 = rot[J][3]; dd2 = rot[J][4]
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

function bending_residual_qp!(re, scv, qp, kin, D, Cs, rot, tie, vn::Val{NN}) where NN
    a₁, a₂, _, d₁, d₂, κ, γ₁, γ₂ = kin
    M   = D ⊡ κ
    Q₁  = Cs[1,1]*γ₁ + Cs[1,2]*γ₂
    Q₂  = Cs[2,1]*γ₁ + Cs[2,2]*γ₂
    P¹  = M[1,1]*d₁ + M[1,2]*d₂
    P²  = M[2,1]*d₁ + M[2,2]*d₂
    S¹  = M[1,1]*a₁ + M[1,2]*a₂
    S²  = M[2,1]*a₁ + M[2,2]*a₂
    dΩ  = scv.detJdV[qp]
    Bγ  = shear_operators(scv, qp, kin, rot, tie, vn)
    for I in 1:NN
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        @views re[5I-4:5I-2] .+= (∂NI1*P¹ + ∂NI2*P²) * dΩ
        F_I = ∂NI1*S¹ + ∂NI2*S²
        re[5I-1] += dot(F_I, rot[I][3]) * dΩ
        re[5I  ] += dot(F_I, rot[I][4]) * dΩ
        b1, b2 = Bγ[I]
        for p in 1:5
            re[5(I-1)+p] += (Q₁*b1[p] + Q₂*b2[p]) * dΩ
        end
    end
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)

RM bending and transverse shear tangent, explicit index-notation form.

Bending, per (I,J) pair:

- **uu** (3×3): ``\\partial_\\alpha N_I \\partial_\\gamma N_J (D^{\\alpha\\beta\\gamma\\delta} d_{,\\beta} \\otimes d_{,\\delta})`` — frame stiffness with ``d_1, d_2``.
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

function bending_tangent_qp!(ke, scv, qp, u_e::AbstractVector{T}, kin, D, Cs, rot, tie, vn::Val{NN}) where {T,NN}
    a₁, a₂, _, d₁, d₂, κ, γ₁, γ₂ = kin
    M   = D ⊡ κ
    Q₁  = Cs[1,1]*γ₁ + Cs[1,2]*γ₂
    Q₂  = Cs[2,1]*γ₁ + Cs[2,2]*γ₂
    S¹  = M[1,1]*a₁ + M[1,2]*a₂
    S²  = M[2,1]*a₁ + M[2,2]*a₂
    L₁₁, L₁₂, L₂₂ = frame_stiffness(D, d₁, d₂)
    dΩ  = scv.detJdV[qp]
    Bγ  = shear_operators(scv, qp, kin, rot, tie, vn)
    for I in 1:NN
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        F_I = ∂NI1*S¹ + ∂NI2*S²                  # bending traction at I
        s_I, sc_I, dd_I1, dd_I2 = rot[I]
        b1I, b2I = Bγ[I]
        for J in 1:NN
            ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
            # uu bending: frame stiffness with d₁,d₂
            K_uu = ∂NI1*∂NJ1*L₁₁ + ∂NI1*∂NJ2*L₁₂ + ∂NI2*∂NJ1*transpose(L₁₂) + ∂NI2*∂NJ2*L₂₂
            @views ke[5I-4:5I-2, 5J-4:5J-2] .+= K_uu * dΩ
            # shear material: 5×5 outer product over all DOFs of (I,J)
            b1J, b2J = Bγ[J]
            for p in 1:5, q in 1:5
                cbq = b1I[p]*(Cs[1,1]*b1J[q] + Cs[1,2]*b2J[q]) +
                      b2I[p]*(Cs[2,1]*b1J[q] + Cs[2,2]*b2J[q])
                ke[5(I-1)+p, 5(J-1)+q] += cbq * dΩ
            end
            # uφ/φu: bending material + bending and shear geometric; φφ: bending material
            g_IJ = ∂NI1*(M[1,1]*∂NJ1+M[1,2]*∂NJ2) + ∂NI2*(M[2,1]*∂NJ1+M[2,2]*∂NJ2)
            g₁, g₂ = shear_coupling(scv, qp, tie, I, J)
            g_IJ += Q₁*g₁ + Q₂*g₂
            for (l, dd_Jl) in ((1, rot[J][3]), (2, rot[J][4]))
                c₁ = dot(a₁, dd_Jl); c₂ = dot(a₂, dd_Jl)
                δκ = SymmetricTensor{2,2,T}((∂NJ1*c₁, 0.5*(∂NJ1*c₂+∂NJ2*c₁), ∂NJ2*c₂))
                δM = D ⊡ δκ
                col = 5J - 2 + l
                Kuφ = ∂NI1*(δM[1,1]*d₁+δM[1,2]*d₂) + ∂NI2*(δM[2,1]*d₁+δM[2,2]*d₂) + g_IJ*dd_Jl
                @views ke[5I-4:5I-2, col] .+= Kuφ * dΩ
                @views ke[col, 5I-4:5I-2] .+= Kuφ * dΩ
                δF_I = ∂NI1*(δM[1,1]*a₁ + δM[1,2]*a₂) + ∂NI2*(δM[2,1]*a₁ + δM[2,2]*a₂)
                ke[5I-1, col] += dot(δF_I, dd_I1) * dΩ
                ke[5I,   col] += dot(δF_I, dd_I2) * dΩ
            end
        end
        # φφ geometric (J = I): (F_I + Q^α W_α)·∂²d_I/∂φ_k∂φ_l
        # sccc = (-sinc - 3scc)/θ², Taylor at θ²→0: 1/15
        φ₁_I = u_e[5I-1]; φ₂_I = u_e[5I]; θ²_I = φ₁_I^2 + φ₂_I^2
        sccc_I = θ²_I < 1e-6 ? one(T)/15 : (-s_I - 3sc_I)/θ²_I
        d2_11, d2_12, d2_22 = rodrigues_hess(s_I, sc_I, sccc_I, φ₁_I, φ₂_I,
                                             scv.G₃_elem[I], scv.T₁_elem[I], scv.T₂_elem[I])
        W₁, W₂ = shear_director_weight(scv, qp, kin, tie, I)
        V_I = F_I + Q₁*W₁ + Q₂*W₂
        ke[5I-1, 5I-1] += dot(V_I, d2_11) * dΩ
        ke[5I-1, 5I  ] += dot(V_I, d2_12) * dΩ
        ke[5I,   5I-1] += dot(V_I, d2_12) * dΩ
        ke[5I,   5I  ] += dot(V_I, d2_22) * dΩ
    end
end

"""
    residuals_RM!(re, scv, u_e, mat)

Complete RM element residual (membrane + bending + transverse shear): the sum of
[`membrane_residuals_RM!`](@ref) and [`bending_residuals_RM!`](@ref) in a single quadrature
loop that evaluates the material once per point.
"""
residuals_RM!(re, scv::ShellCellValues, u_e, mat) = rm_element!(nothing, re, scv, u_e, mat, Val(:all))

"""
    tangent_RM!(ke, scv, u_e, mat)

Complete RM element tangent: the sum of [`membrane_tangent_RM!`](@ref) and
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

# RM element kernel shared by all explicit entry points. `parts` ∈ Val.((:membrane, :bending,
# :all)); `ke` or `re` may be `nothing`. All of these branches resolve at compile time.
function rm_element!(ke, re, scv::ShellCellValues, u_e::AbstractVector{T}, mat, parts::Val{P}) where {T,P}
    mem  = P !== :bending
    bend = P !== :membrane
    vn = Val(getnbasefunctions(scv.ip_shape))
    n_nodes = getnbasefunctions(scv.ip_shape)
    γ_k = bend ? tying_shear_strains(scv.mitc, u_e) : nothing
    tie = bend ? shear_tying_state(scv.mitc, u_e) : nothing
    rot = bend ? nodal_rotations(scv, u_e, vn) : nothing
    for qp in 1:getnquadpoints(scv)
        kin = bend ? rm_kinematics(scv, qp, u_e, γ_k, n_nodes) : covariant_basis(scv, qp, u_e, n_nodes)
        a₁, a₂ = kin
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        N, C, D, Cs = qp_material(mat, parts, c_ms, scv.A_metric[qp], scv.A₁[qp], scv.A₂[qp], scv.G₃[qp])
        if re !== nothing
            mem  && membrane_residual_qp!(re, scv, qp, a₁, a₂, N)
            bend && bending_residual_qp!(re, scv, qp, kin, D, Cs, rot, tie, vn)
        end
        if ke !== nothing
            mem  && membrane_tangent_qp!(ke, scv, qp, a₁, a₂, N, C)
            bend && bending_tangent_qp!(ke, scv, qp, u_e, kin, D, Cs, rot, tie, vn)
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

# Local positions of the three displacement components of node `I` within a cell's dof
# vector, read from the SubDofHandler's own `dof_range(:u)` rather than assumed to sit at
# `3I-2:3I`. That assumption holds only when `:u` is the first field; it silently writes
# into the wrong dofs when the fields are added in another order, when the DofHandler
# carries an extra field, or for the single-field interleaved `:u` as `ip^5` layout.
@inline function _u_dofs(ru, ncomp::Int, I::Int)
    o = ncomp * (I - 1)
    (ru[o+1], ru[o+2], ru[o+3])
end

# Components per node in the `:u` field: 3 for the two-field layout (`:u` as ip^3,
# `:θ` as ip^2), 5 for the single-field interleaved one.
@inline function _u_ncomp(ru, n_base::Int)
    ncomp, r = divrem(length(ru), n_base)
    (r == 0 && ncomp in (3, 5)) || throw(ArgumentError(
        ":u carries $(length(ru)) dofs over $n_base base functions; expected 3 per node " *
        "(two-field `:u` as ip^3) or 5 (single-field interleaved `:u` as ip^5)"))
    ncomp
end

# ∂ξ/∂t for the mapping t → 2D cell reference coord along facet `f`, where t is the
# parametrization implicit in `FacetQuadratureRule`: ξ(t) = A + t·(B-A) for the facet's
# vertex pair (A, B), with t ranging over an interval of length `weight_sum` (any valid
# quadrature rule integrates a constant exactly, so the rule's own weights sum to that
# length). Reads (A, B) from Ferrite's `reference_facets`/`reference_coordinates` rather
# than hardcoding per-shape vertex order — RefQuadrilateral's facets are parametrized
# over t∈[-1,1] (weight_sum=2) and RefTriangle's over t∈[0,1] (weight_sum=1); dividing
# by the measured weight_sum makes both cases (and any future RefShape) correct without
# needing to special-case the convention.
@inline function facet_dxi(::Interpolation{RefShape}, f::Int, weight_sum::Float64) where RefShape
    a, b = Ferrite.reference_facets(RefShape)[f]
    coords = Ferrite.reference_coordinates(Lagrange{RefShape,1}())
    return (coords[b] - coords[a]) / weight_sum
end

"""
    assemble_traction!(f, dh, facetset, ip::Interpolation, fqr::FacetQuadratureRule, traction)

Assemble external traction into force vector f for embedded shell elements (2D mesh in 3D).
`traction` is either a `Vec{3}` (uniform) or a callable `x::Vec{3} -> Vec{3}`.
Uses a `FacetQuadratureRule` and computes the edge length element directly from 3D node positions,
bypassing the sdim mismatch that prevents standard `FacetValues` from working on embedded meshes.
Works for RefQuadrilateral and RefTriangle of any interpolation order.
The `:u` dofs are located through each cell's SubDofHandler `dof_range`, so the two-field
(`:u` as ip³, `:θ` as ip²) layout, the single-field interleaved (`:u` as ip⁵) one, any field
order and any additional fields are all handled.
"""
function assemble_traction!(f, dh, facetset, ip::Interpolation, fqr::FacetQuadratureRule, traction)
    t_func = traction isa Function ? traction : (_ -> Vec{3}(traction))
    n_base = getnbasefunctions(ip)
    fe     = Float64[]
    for fc in FacetIterator(dh, facetset)
        cd    = celldofs(fc)
        sdh   = dh.subdofhandlers[dh.cell_to_subdofhandler[Ferrite.cellid(fc)]]
        ru    = Ferrite.dof_range(sdh, :u)
        ncomp = _u_ncomp(ru, n_base)
        resize!(fe, length(cd)); fill!(fe, 0.0)
        x        = getcoordinates(fc)
        facet_nr = fc.current_facet_id
        qr_f     = fqr.facet_rules[facet_nr]
        dxi      = facet_dxi(ip, facet_nr, sum(qr_f.weights))   # ∂ξ/∂t in reference coords
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
                N   = Ferrite.reference_shape_value(ip, ξ, I)
                Ndg = N * dΓ
                d1, d2, d3 = _u_dofs(ru, ncomp, I)
                fe[d1] += Ndg * t[1]; fe[d2] += Ndg * t[2]; fe[d3] += Ndg * t[3]
            end
        end
        f[cd] .+= fe
    end
end

"""
    assemble_pressure!(re, scv, u_e, p)

Follower pressure residual for embedded shell elements.
``\\mathrm{detJdV}[q] = \\|A_1 \\times A_2\\| \\cdot w`` (reference area times quadrature weight).
``\\mathrm{cross}(a_1, a_2)`` has magnitude ``\\|a_1 \\times a_2\\|`` (current area per parametric area).
"""
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
The `:u` dofs are located through the SubDofHandler's `dof_range`, so the two-field
(`:u` as ip³, `:θ` as ip²) layout, the single-field interleaved (`:u` as ip⁵) one, any
field order and any additional fields are all handled.
"""
function apply_pointload!(f, dh, nodeset_name::String, load::Vec{3})
    node_set  = getnodeset(dh.grid, nodeset_name)
    processed = Set{Int}()
    for cell in CellIterator(dh)
        nodes = getnodes(cell)
        any(gid -> gid ∈ node_set && gid ∉ processed, nodes) || continue
        cd    = celldofs(cell)
        sdh   = dh.subdofhandlers[dh.cell_to_subdofhandler[Ferrite.cellid(cell)]]
        ru    = Ferrite.dof_range(sdh, :u)
        ncomp = _u_ncomp(ru, length(nodes))
        for (I, gid) in enumerate(nodes)
            if gid ∈ node_set && gid ∉ processed
                push!(processed, gid)
                d1, d2, d3 = _u_dofs(ru, ncomp, I)
                f[cd[d1]] += load[1]; f[cd[d2]] += load[2]; f[cd[d3]] += load[3]
            end
        end
    end
end