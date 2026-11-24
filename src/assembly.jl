# Assembly routines for embedded shell elements (2D mesh in 3D)

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