using FerriteShells

function create_cook_grid(nx, ny; primitive=Quadrilateral)
    corners = [Tensors.Vec{2}((0.0, 0.0)),
               Tensors.Vec{2}((48.0, 44.0)),
               Tensors.Vec{2}((48.0, 60.0)),
               Tensors.Vec{2}((0.0, 44.0))]
    return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed in into a 3D space
end

# integrate edge traction force into f
# DOF ordering assumed: :u field first (3 DOFs per node, interleaved u,v,w)
function assemble_traction_force!(f, dh, facetset, traction)
    edge_local_nodes = Ferrite.reference_facets(RefQuadrilateral)
    n_dpc = ndofs_per_cell(dh)
    fe    = zeros(n_dpc)
    for fc in FacetIterator(dh, facetset)
        x  = getcoordinates(fc)
        fn = fc.current_facet_id             # local facet index: 1, 2, or 3
        ia, ib = edge_local_nodes[fn]        # local node indices on this edge
        edge_len = norm(x[ib] - x[ia])
        fill!(fe, 0.0)
        # 1-point midpoint quadrature: both edge nodes receive equal weight 0.5
        # (exact for the linear shape functions used here)
        for (node, N) in ((ia, 0.5), (ib, 0.5))
            for c in 1:3    # u, v, w components
                fe[3(node-1)+c] += N * traction[c] * edge_len
            end
        end
        f[celldofs(fc)] .+= fe
    end
end

function assemble_membrane!(K, r, dh, scv, u, mat)
    @assert length(u) == ndofs(dh) "u're not long enough mate!"
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re  = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        x = getcoordinates(cell)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell) # prepares reference geometry
        u_e = reinterpret(Vec{3,eltype(u)}, u[celldofs(cell)])
        membrane_tangent!(ke, scv, x, u_e, mat)
        membrane_residuals!(re, scv, x, u_e, mat)
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

# number of cells
grid = create_cook_grid(32, 16)

# facesets for boundary conditions
addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
addnodeset!(grid, "nodes", x -> true)

# interpolation order
ip = Lagrange{RefQuadrilateral,1}() #to define fields only
qr = QuadratureRule{RefQuadrilateral}(2) # avoid zero spurious modes

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)

# degrees of freedom for displacements (pure membrane test)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

# material model
mat = LinearElastic(1.0, 0.3)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"), x -> [0.0], [3]))
close!(dbc)

# stiffness matrix assembly (for visual inspection only, not used in a solve here)
Ke = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

# test traction force assembly
assemble_traction_force!(f, dh, getfacetset(grid, "traction"), (0.0, 1.0/16, 0.0))

# apply BCs
apply!(Ke, f, dbc)
@time ue = Ke\f

# write to vtk
VTKGridFile("cooks_membrane", dh) do vtk
    write_solution(vtk, dh, ue)
end