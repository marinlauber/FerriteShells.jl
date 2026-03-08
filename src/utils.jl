import Ferrite: Grid,Triangle,Quadrilateral,Nodes
using LinearAlgebra: cross

# maps the 2D nodes of a mesh onto the 3D coordinates
# by applying the `map` function to the nodes (default: flat z=0 plane)
function shell_grid(grid::Grid{2,P,T}; map::Function=(n)->(n.x[1], n.x[2], zero(T))) where {P<:Union{Triangle,Quadrilateral},T}
    return Grid(grid.cells, [Node(Tensors.Vec{3}(map(n))) for n in grid.nodes])
end

# Compute the contravariant metric A^{αβ} = inv(A_{αβ}) from the covariant metric A_{αβ}.
function contravariant(A_cov::SymmetricTensor{2,2,T}) where T
    det_A = A_cov[1,1]*A_cov[2,2] - A_cov[1,2]^2
    A11u  =  A_cov[2,2] / det_A
    A12u  = -A_cov[1,2] / det_A
    A22u  =  A_cov[1,1] / det_A
    SymmetricTensor{2,2,T}((A11u, A12u, A22u))
end

# Assemble external traction into force vector f.
# `traction` is either a Vec{3} (uniform) or a callable x::Vec{3} -> Vec{3}.
# `fv` is a FacetValues created for the element interpolation of the :u field.
function assemble_traction!(f, dh, facetset, fv::FacetValues, traction)
    t_func = traction isa Function ? traction : (_ -> Vec{3}(traction))
    n_base = getnbasefunctions(fv)
    fe     = zeros(ndofs_per_cell(dh))
    for fc in FacetIterator(dh, facetset)
        fill!(fe, 0.0)
        reinit!(fv, fc)
        x = getcoordinates(fc)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            t  = t_func(spatial_coordinate(fv, qp, x))
            for I in 1:n_base
                NI = shape_value(fv, qp, I)
                fe[3I-2] += NI * t[1] * dΓ
                fe[3I-1] += NI * t[2] * dΓ
                fe[3I  ] += NI * t[3] * dΓ
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