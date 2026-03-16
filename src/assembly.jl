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

"""
    membrane_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love membrane residual. `u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function membrane_residuals_KL!(re, scv, u_e, mat)
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
function membrane_tangent_KL!(ke, scv, u_e, mat)
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
Curvature change κ_{αβ} = b_{αβ} - B_{αβ} (current minus reference second fundamental form).
Requires Q2+ elements for full κ (Q4 only captures twist κ₁₂).
`u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function bending_energy_KL(u_flat, scv, mat)
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

function bending_residuals_KL!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy_KL(u, scv, mat), u_e)
end

function bending_tangent_KL!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy_KL(u, scv, mat), u_e)
end

"""
Reissner–Mindlin membrane strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).
Only the displacement DOFs (indices 5I-4:5I-2) contribute to membrane energy.
"""
function rm_membrane_energy(u_flat, scv, mat)
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
Reissner–Mindlin bending + transverse shear strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).

Director parametrization: d_I = G₃ + φ₁_I·T₁ + φ₂_I·T₂
where G₃ is the reference unit normal and T₁, T₂ are reference tangents from scv.

Bending strain:  κ_{αβ} = ½(a_α·d,β + a_β·d,α) - B_{αβ}
Transverse shear: γ_α = a_α·d

Shear correction factor κ_s = 5/6 is applied.
"""
function rm_bending_shear_energy(u_flat, scv, mat)
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
    membrane_residuals_RM!(re, scv, u_e, mat)

Reissner–Mindlin membrane residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_residuals_RM!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> rm_membrane_energy(u, scv, mat), u_e)
end
function membrane_residuals_RM_impl!(re, scv, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁       = scv.A₁[qp] + Δa₁
        a₂       = scv.A₂[qp] + Δa₂
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E        = 0.5 * (a_metric - scv.A_metric[qp])
        N        = contravariant_elasticity(mat, scv.A_metric[qp]) ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
                ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
            @views re[5I-4:5I-2] .+= v * dΩ
        end
    end
end

"""
    membrane_tangent_RM!(ke, scv, u_e, mat)

Reissner–Mindlin membrane tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_tangent_RM!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> rm_membrane_energy(u, scv, mat), u_e)
end # 1050 μs (26 allocations: 115.21 KiB) on a 45x45 matrix
function membrane_tangent_RM_impl!(ke, scv, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_e[5I-4], u_e[5I-3], u_e[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁       = scv.A₁[qp] + Δa₁
        a₂       = scv.A₂[qp] + Δa₂
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E        = 0.5 * (a_metric - scv.A_metric[qp])
        C        = contravariant_elasticity(mat, scv.A_metric[qp])
        N        = C ⊡ E
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
                @views ke[5I-4:5I-2, 5J-4:5J-2] .+= (Kgeo + Kmat) * dΩ
            end
        end
    end
end # 19.969 μs (0 allocations: 0 bytes) on a 45x45 matrix (50x speedup)

"""
    bending_residuals_RM!(re, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_residuals_RM!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> rm_bending_shear_energy(u, scv, mat), u_e)
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_tangent_RM!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> rm_bending_shear_energy(u, scv, mat), u_e)
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
function assemble_pressure!(re, scv, u_e, p)
    T = eltype(u_e)
    n_nodes = getnbasefunctions(scv.ip_shape)
    ndofs_per_node = length(u_e) ÷ n_nodes
    @inline block(I) = ndofs_per_node == 5 ? (5I-4:5I-2) : (3I-2:3I)
    for qp in 1:getnquadpoints(scv)
        w = scv.qr.weights[qp]
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            u_I = Vec{3,T}(tuple(u_e[block(I)]...))
            Δa₁ += u_I * scv.dNdξ[I, qp][1]
            Δa₂ += u_I * scv.dNdξ[I, qp][2]
        end
        a₁ = scv.A₁[qp] + Δa₁
        a₂ = scv.A₂[qp] + Δa₂
        n_w = cross(a₁, a₂)
        for I in 1:n_nodes
            @views re[block(I)] .+= p * scv.N[I, qp] * n_w * w
        end
    end
end

"""
# Load-stiffness K_pres = ∂F_p/∂u via ForwardDiff (for unit pressure p=1).
# `K_IJ^p = p * ∂(a₁ × a₂)/∂u_J * N_I`
"""
function assemble_pressure_tangent!(ke, scv, u_e, p)
    pressure_residual(u) = (re = zeros(eltype(u), length(u)); assemble_pressure!(re, scv, u, p); re)
    ke .+= ForwardDiff.jacobian(pressure_residual, u_e)
end


"""
    apply_pointload!(f, dh, nodeset_name, load)

Add a concentrated force `load::Vec{3}` to the displacement DOFs of all nodes in `nodeset_name`.
Works for both single-field (:u only) and two-field (:u, :θ) DofHandlers; in both cases the
:u DOFs for node I in a cell occupy local positions 3I-2:3I.
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