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
                fe[3I-2] += N * t[1] * dΓ
                fe[3I-1] += N * t[2] * dΓ
                fe[3I  ] += N * t[3] * dΓ
            end
        end
        f[celldofs(fc)] .+= fe
    end
end

# scv.detJdV[qp] = ‖A₁ × A₂‖ · w (reference area × weight).
# cross(a₁, a₂) already has magnitude ‖a₁ × a₂‖ (current area per parametric area).
# multiplying by w integrates over the parameter domain
# assemble_pressure!(re, scv, x, u_e, p::Number) = assemble_pressure!(re, scv, x, u_e, ()->p)
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
    n_nodes = length(u_e)
    u_vec   = collect(reinterpret(Float64, u_e))   # flat Float64 input for ForwardDiff

    function pressure_residual(u)
        re     = zeros(eltype(u), 3*n_nodes)
        u_e_d  = [Vec{3}((u[3i-2], u[3i-1], u[3i])) for i in 1:n_nodes]
        assemble_pressure!(re, scv, x, u_e_d, p)
        return re
    end

    ke .+= ForwardDiff.jacobian(pressure_residual, u_vec)
end


# K.J - Bath https://doi.org/10.1016/S0045-7949(03)00010-5
function s_norm(u, uₕ)
    nothing
end