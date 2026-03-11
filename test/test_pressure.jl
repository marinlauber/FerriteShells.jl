using FerriteShells

# number of cells
grid = generate_grid(Quadrilateral, (10, 10), Vec((0.0,0.0)), Vec((1.0,1.0))) |> shell_grid
Area = 1.0
pressure = 1.0

# facesets for boundary conditions
addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 1.0)
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
mat = LinearElastic(1.0, 1/3)

# boundary conditions
dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1,2,3]))
add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "nodes"), x -> [0.0], [3]))
close!(dbc)

f = zeros(ndofs(dh))
for cell in CellIterator(dh)
    re = zeros(ndofs_per_cell(dh))
    x = getcoordinates(cell)
    u_e = zeros(ndofs_per_cell(dh)) # zero displacements
    assemble_pressure!(re, scv, u_e, pressure)
    f[celldofs(cell)] .+= re
end
@assert sum(f) ≈ pressure * Area