using FerriteShells

# interpolation space and quadrature
ip = Lagrange{RefQuadrilateral, 2}()
qr = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip, mitc=MITC9)

# make a grid
corners = [Vec{2}((0.0, 0.0)), Vec{2}((1.0, 0.0)), Vec{2}((1.0, 1.0)), Vec{2}((0.0, 1.0))]
grid = shell_grid(generate_grid(QuadraticQuadrilateral, (4, 4), corners))

# degrees of freedom of the problem
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# reinit
cell = first(CellIterator(dh))
reinit!(scv, cell)