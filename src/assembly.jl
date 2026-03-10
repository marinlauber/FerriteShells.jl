
using ForwardDiff

"""
    membrane_residuals_KL!(re, scv, x, u_e, mat)

Kirchhoff–Love membrane residual. `u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function membrane_residuals_KL!(re, scv, x, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        a₁, a₂, A_metric, a_metric = kinematics(scv, qp, x, u_e)
        E = 0.5 * (a_metric - A_metric)
        N = contravariant_elasticity(mat, A_metric) ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, I)
            v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
                ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
            @views re[3I-2:3I] .+= v * dΩ
        end
    end
end

"""
    membrane_tangent_KL!(ke, scv, x, u_e, mat)

Kirchhoff–Love membrane tangent. `u_e` is a flat vector of length 3·n_nodes.
"""
function membrane_tangent_KL!(ke, scv, x, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        a₁, a₂, A_metric, a_metric = kinematics(scv, qp, x, u_e)
        E = 0.5 * (a_metric - A_metric)
        C = contravariant_elasticity(mat, A_metric)
        N = C ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, I)
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, J)
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
Curvature change κ_{αβ} = b_{αβ} - B_{αβ} (current minus reference second fundamental form).
Requires Q2+ elements for full κ (Q4 only captures twist κ₁₂).
`u_e` is a flat vector of length 3·n_nodes.
"""
function bending_energy_KL(u_flat, scv, x, mat)
    T = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    u_e = [Vec{3,T}((u_flat[3i-2], u_flat[3i-1], u_flat[3i])) for i in 1:n_nodes]
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        A₁  = zero(Vec{3,Float64}); A₂  = zero(Vec{3,Float64})
        a₁  = zero(Vec{3,T});       a₂  = zero(Vec{3,T})
        A₁₁ = zero(Vec{3,Float64}); A₁₂ = zero(Vec{3,Float64}); A₂₂ = zero(Vec{3,Float64})
        a₁₁ = zero(Vec{3,T});       a₁₂ = zero(Vec{3,T});       a₂₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            d2N, dN, _ = Ferrite.reference_shape_hessian_gradient_and_value(scv.ip_shape, ξ, I)
            XI = x[I]; UI = u_e[I]
            A₁  += XI * dN[1];        A₂  += XI * dN[2]
            a₁  += (XI + UI) * dN[1]; a₂  += (XI + UI) * dN[2]
            A₁₁ += XI * d2N[1,1];    A₁₂ += XI * d2N[1,2];    A₂₂ += XI * d2N[2,2]
            a₁₁ += (XI + UI) * d2N[1,1]; a₁₂ += (XI + UI) * d2N[1,2]; a₂₂ += (XI + UI) * d2N[2,2]
        end
        A_metric = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
        N_n = (A₁ × A₂) / norm(A₁ × A₂)
        n_n = (a₁ × a₂) / norm(a₁ × a₂)
        B = SymmetricTensor{2,2,Float64}((dot(A₁₁,N_n), dot(A₁₂,N_n), dot(A₂₂,N_n)))
        b = SymmetricTensor{2,2,T}((dot(a₁₁,n_n), dot(a₁₂,n_n), dot(a₂₂,n_n)))
        κ = b - B
        D = contravariant_bending_stiffness(mat, A_metric)
        W += 0.5 * (κ ⊡ D ⊡ κ) * scv.detJdV[qp]
    end
    return W
end

function bending_residuals_KL!(re, scv, x, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy_KL(u, scv, x, mat), u_e)
end

function bending_tangent_KL!(ke, scv, x, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy_KL(u, scv, x, mat), u_e)
end

"""
Reissner–Mindlin membrane strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).
Only the displacement DOFs (indices 5I-4:5I-2) contribute to membrane energy.
"""
function rm_membrane_energy(u_flat, scv, x, mat)
    T = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        A₁ = zero(Vec{3,Float64}); A₂ = zero(Vec{3,Float64})
        a₁ = zero(Vec{3,T});       a₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, I)
            u_I = Vec{3,T}((u_flat[5I-4], u_flat[5I-3], u_flat[5I-2]))
            A₁ += x[I] * dN[1];         A₂ += x[I] * dN[2]
            a₁ += (x[I] + u_I) * dN[1]; a₂ += (x[I] + u_I) * dN[2]
        end
        A_metric = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E = 0.5 * (a_metric - A_metric)
        C = contravariant_elasticity(mat, A_metric)
        W += 0.5 * (E ⊡ C ⊡ E) * scv.detJdV[qp]
    end
    return W
end

"""
Reissner–Mindlin bending + transverse shear strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).

Director parametrization: d_I = G₃ + φ₁_I·T₁ + φ₂_I·T₂
where G₃ is the reference unit normal and T₁, T₂ are in-plane reference tangents.

Bending strain:  κ_{αβ} = ½(a_α·d,β + a_β·d,α) - B_{αβ}^ref
Transverse shear: γ_α = a_α·d - A_α·G₃

Shear correction factor κ_s = 5/6 is applied.
"""
function rm_bending_shear_energy(u_flat, scv, x, mat)
    T = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    # Buffers for cached shape function data (reused across second pass)
    dNs = Vector{Vec{2,Float64}}(undef, n_nodes)
    NIs = Vector{Float64}(undef, n_nodes)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        # First pass: reference and current tangent vectors + 2nd derivatives
        A₁  = zero(Vec{3,Float64}); A₂  = zero(Vec{3,Float64})
        A₁₁ = zero(Vec{3,Float64}); A₁₂ = zero(Vec{3,Float64}); A₂₂ = zero(Vec{3,Float64})
        a₁  = zero(Vec{3,T});       a₂  = zero(Vec{3,T})
        for I in 1:n_nodes
            d2N, dN, NI = Ferrite.reference_shape_hessian_gradient_and_value(scv.ip_shape, ξ, I)
            u_I = Vec{3,T}((u_flat[5I-4], u_flat[5I-3], u_flat[5I-2]))
            A₁  += x[I] * dN[1];         A₂  += x[I] * dN[2]
            a₁  += (x[I] + u_I) * dN[1]; a₂  += (x[I] + u_I) * dN[2]
            A₁₁ += x[I] * d2N[1,1]; A₁₂ += x[I] * d2N[1,2]; A₂₂ += x[I] * d2N[2,2]
            dNs[I] = dN
            NIs[I] = NI
        end
        # Reference frame
        J      = A₁ × A₂
        J_norm = norm(J)
        G₃     = J / J_norm
        T₁     = A₁ / norm(A₁)
        T₂_raw = G₃ × T₁
        T₂     = T₂_raw / norm(T₂_raw)
        # Reference curvature: B_{αβ}^ref = ½(A_α·G₃,β + A_β·G₃,α)
        # (equals zero for flat reference geometry)
        J₁  = A₁₁ × A₂ + A₁ × A₁₂
        J₂  = A₁₂ × A₂ + A₁ × A₂₂
        G₃₁ = (J₁ - G₃ * dot(G₃, J₁)) / J_norm
        G₃₂ = (J₂ - G₃ * dot(G₃, J₂)) / J_norm
        B₁₁ = dot(A₁, G₃₁)
        B₁₂ = 0.5 * (dot(A₁, G₃₂) + dot(A₂, G₃₁))
        B₂₂ = dot(A₂, G₃₂)
        # Second pass: director d and its parametric gradients d,α
        d  = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            φ₁ = u_flat[5I-1]; φ₂ = u_flat[5I]
            d_I = Vec{3,T}((G₃[1] + φ₁*T₁[1] + φ₂*T₂[1],
                            G₃[2] + φ₁*T₁[2] + φ₂*T₂[2],
                            G₃[3] + φ₁*T₁[3] + φ₂*T₂[3]))
            d  += NIs[I]    * d_I
            d₁ += dNs[I][1] * d_I
            d₂ += dNs[I][2] * d_I
        end
        # Bending strain κ_{αβ} = ½(a_α·d,β + a_β·d,α) - B_{αβ}^ref
        κ₁₁ = dot(a₁, d₁) - B₁₁
        κ₁₂ = 0.5 * (dot(a₁, d₂) + dot(a₂, d₁)) - B₁₂
        κ₂₂ = dot(a₂, d₂) - B₂₂
        κ   = SymmetricTensor{2,2,T}((κ₁₁, κ₁₂, κ₂₂))
        # Transverse shear γ_α = a_α·d - A_α·G₃
        γ₁ = dot(a₁, d) - dot(A₁, G₃)
        γ₂ = dot(a₂, d) - dot(A₂, G₃)
        # Stiffness tensors
        A_metric = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
        D    = contravariant_bending_stiffness(mat, A_metric)
        Aup  = inv(A_metric)
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
    membrane_residuals_RM!(re, scv, x, u_e, mat)

Reissner–Mindlin membrane residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_residuals_RM!(re, scv, x, u_e, mat)
    re .+= ForwardDiff.gradient(u -> rm_membrane_energy(u, scv, x, mat), u_e)
end

"""
    membrane_tangent_RM!(ke, scv, x, u_e, mat)

Reissner–Mindlin membrane tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_tangent_RM!(ke, scv, x, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> rm_membrane_energy(u, scv, x, mat), u_e)
end

"""
    bending_residuals_RM!(re, scv, x, u_e, mat)

Reissner–Mindlin bending + transverse shear residual. `u_e` is a flat vector of length 5·n_nodes.
Includes both bending (κ_{αβ}) and transverse shear (γ_α) contributions.
"""
function bending_residuals_RM!(re, scv, x, u_e, mat)
    re .+= ForwardDiff.gradient(u -> rm_bending_shear_energy(u, scv, x, mat), u_e)
end

"""
    bending_tangent_RM!(ke, scv, x, u_e, mat)

Reissner–Mindlin bending + transverse shear tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_tangent_RM!(ke, scv, x, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> rm_bending_shear_energy(u, scv, x, mat), u_e)
end
