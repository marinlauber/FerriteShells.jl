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
        u_e = u[shelldofs(cell)]
        FerriteShells.membrane_tangent_RM_explicit!(ke, scv, u_e, mat)
        FerriteShells.bending_tangent_RM_explicit!(ke, scv, u_e, mat)
        FerriteShells.membrane_residuals_RM_explicit!(re, scv, u_e, mat)
        FerriteShells.bending_residuals_RM_explicit!(re, scv, u_e, mat)
        assemble!(assembler, shelldofs(cell), ke, re)
    end
end

# number of cells
grid = create_cook_grid(32, 16; primitive=QuadraticQuadrilateral)

# facesets for boundary conditions
addfacetset!(grid,  "clamped", x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)

# interpolation order
# ip = Lagrange{RefQuadrilateral, 1}() # Q4
ip = Lagrange{RefQuadrilateral, 2}() # Q9
# ip = Lagrange{RefTriangle, 2}() # S3 TODO this requires different interpolation method
qr = QuadratureRule{RefQuadrilateral}(3)

# cell (shell) values
scv = ShellCellValues(qr, ip, ip)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)

# degrees of freedom for displacements (pure membrane test)
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# material model
mat = LinearElastic(1.0, 1/3, 1.0)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:θ, getfacetset(dh.grid, "clamped"), x -> [0.0,0.0], [1,2]))
close!(dbc)

# stiffness matrix and residuals vector construction and assembly
Ke = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_membrane!(Ke, f, dh, scv, zeros(ndofs(dh)), mat)

# traction force assembly, force of 1N on the face, split into 16 units (length of face)
assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 1/16, 0.0))

# apply BCs and solve (\) figures out the best linear solver to use
apply!(Ke, f, dbc)
@time ue = Ke \ f

# extract solution at point
ph     = PointEvalHandler(grid, [Vec{3}((48.0, 60.0, 0.0))])
u_eval = first(evaluate_at_points(ph, dh, ue, :u))
@show u_eval

# write to vtk
VTKGridFile("cooks_membrane_RM", dh) do vtk
    write_solution(vtk, dh, ue)
end