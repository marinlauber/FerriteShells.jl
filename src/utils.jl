import Ferrite: Grid,Triangle,Quadrilateral,Nodes

# maps the 2D nodes of a mesh onto the 3D coordinates
# by applying the `map` function to the nodes (default: flat z=0 plane)
function shell_grid(grid::Grid{2,P,T}; map::Function=(n)->(n.x[1], n.x[2], zero(T))) where {P<:Union{Triangle,Quadrilateral},T}
    return Grid(grid.cells, [Node(Tensors.Vec{3}(map(n))) for n in grid.nodes])
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

# K.J - Bath https://doi.org/10.1016/S0045-7949(03)00010-5
function s_norm(u, uₕ)
    nothing
end