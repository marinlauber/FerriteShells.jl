import Ferrite: Grid,Triangle,Quadrilateral,Nodes
using LinearAlgebra: cross

# maps the 2D nodes of a mesh onto the 3D coordinates
# by applying the `map` function to the nodes (default: flat z=0 plane)
function shell_grid(grid::Grid{2,P,T}; map::Function=(n)->(n.x[1], n.x[2], zero(T))) where {P<:Union{Triangle,Quadrilateral,QuadraticQuadrilateral},T}
    return Grid(grid.cells, [Node(Tensors.Vec{3}(map(n))) for n in grid.nodes])
end

# Assemble external traction into force vector f for embedded shell elements (2D mesh in 3D).
# `traction` is either a Vec{3} (uniform) or a callable x::Vec{3} -> Vec{3}.
# Uses a FacetQuadratureRule and computes the edge length element directly from 3D node positions,
# bypassing the sdim mismatch that prevents standard FacetValues from working on embedded meshes.
# For RefQuadrilateral: facets 1,3 (bottom/top) vary in ξ₁; facets 2,4 (right/left) vary in ξ₂.
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

# scv.detJdV[qp] = ‖A₁ × A₂‖ · w (reference area × weight).
# cross(a₁, a₂) already has magnitude ‖a₁ × a₂‖ (current area per parametric area).
# multiplying by w integrates over the parameter domain
# assemble_pressure!(re, scv, x, u_e, p::Number) = assemble_pressure!(re, scv, x, u_e, ()->p)
# @TODO should be agnostic of the number of dofs of u_e
function assemble_pressure!(re, scv, x, u_e, pfunc)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        w = scv.qr.weights[qp] # pure parametric weight — NOT detJdV
        a₁, a₂, _, _ = kinematics(scv, qp, x, u_e)
        # local oriented area vector
        n_weighted = cross(a₁, a₂)
        for I in 1:n_nodes
            NI = Ferrite.reference_shape_value(scv.ip_shape, ξ, I)
            # re[3I-2:3I] .+= pfunc(spatial_coordinate(scv, qp, x)) * NI * n_weighted * w
            re[3I-2:3I] .+= pfunc * NI * n_weighted * w
        end
    end
end

using ForwardDiff
# K_IJ^p = p * ∂(a₁ × a₂)/∂u_J * N_I
function assemble_pressure_tangent!(ke, scv, x, u_e, p)
    pressure_residual(u) = (re = zeros(eltype(u), length(u)); assemble_pressure!(re, scv, x, u, p); re)
    ke .+= ForwardDiff.jacobian(pressure_residual, u_e)
end

# write the strain at the quadrature points, does that work
function compute_membrane_strains(Es, scv, x, u_e)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        _,_A_metric,a_metric = kinematics(scv, qp, x, u_e)
        Es[qp] = 0.5 * (a_metric - A_metric)
    end
end


# K.J - Bath https://doi.org/10.1016/S0045-7949(03)00010-5
function s_norm(u, uₕ)
    nothing
end

import Ferrite: CellCache
"""
    shelldofs()

Reorder DOFs from a two-field DofHandler layout (:u as ip^3, :θ as ip^2)
to the interleaved 5-DOF-per-node layout expected by the RM assembly functions.
Input:  [u1x,u1y,u1z, u2x,...,unz | θ1₁,θ1₂, θ2₁,...,θn₂]
Output: [u1x,u1y,u1z,θ1₁,θ1₂, u2x,u2y,u2z,θ2₁,θ2₂, ...]
"""
function shelldofs(cell::CellCache)
    dofs = cell.dofs
    n = length(dofs) ÷ 5
    perm = similar(dofs)
    for I in 1:n
        @views perm[5I-4:5I-2] .= dofs[3I-2:3I]
        perm[5I-1] = dofs[3n + 2I-1]
        perm[5I  ] = dofs[3n + 2I]
    end
    return perm
end