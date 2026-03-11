using FerriteShells

function create_cook_grid(nx, ny; primitive=Quadrilateral)
    corners = [Vec{2}(( 0.0,  0.0)), Vec{2}((48.0, 44.0)),
               Vec{2}((48.0, 60.0)), Vec{2}(( 0.0, 44.0))]
    return generate_grid(primitive, (nx, ny), corners) |> shell_grid # embed in into a 3D space
end

function assemble_membrane!(K, r, dh, scv, u, mat)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re  = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell) # prepares reference geometry
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, u_e, mat)
        membrane_residuals_KL!(re, scv, u_e, mat)
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

# number of cells
grid = create_cook_grid(32, 16; primitive=QuadraticQuadrilateral)

# facesets for boundary conditions
addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
addnodeset!(grid, "nodes", x -> true)

# interpolation order
ip = Lagrange{RefQuadrilateral, 2}()
qr = QuadratureRule{RefQuadrilateral}(3)

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# degrees of freedom for displacements (pure membrane test)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

# material model
mat = LinearElastic(1.0, 1/3)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"), x -> [0.0], [3])) # prevents the singular K if z unconstrained
close!(dbc)

# stiffness matrix and residuals vector construction and assembly
Ke = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

# traction force assembly, force of 1N on the face, split into 16 units (length of face)
assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1.0/16, 0.0))

# apply BCs and solve (\) figures out the best linear solver to use
apply!(Ke, f, dbc)
@time ue = Ke\f

# extract solution at point
ph     = PointEvalHandler(grid, [Vec{3}((48.0, 60.0, 0.0))])
u_eval = first(evaluate_at_points(ph, dh, ue, :u))
@show u_eval

# write to vtk
VTKGridFile("cooks_membrane", dh) do vtk
    write_solution(vtk, dh, ue)
end