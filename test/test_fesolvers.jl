using FerriteShells, FESolvers, Test, WriteVTK

grid = shell_grid(generate_grid(Quadrilateral, (32, 32)))

ip   = Lagrange{RefQuadrilateral, 1}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip)

mat = LinearElastic(1.0e6, 0.3, 1e-2)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "right"),  x -> 0.0, [3]))
close!(ch); Ferrite.update!(ch, 0.0)

pvd = paraview_collection("shell")

# A shell problem
rm_shell = ReissnerMindlinShellProblem(dh, ch, scv, mat; pvd=pvd)
# solver = QuasiStaticSolver(;nlsolver=LinearProblemSolver(),
                            # timestepper=FixedTimeStepper(collect(0.0:1.0:200)))
solver = LinearProblemSolver(;linsolver=BackslashSolver())
# solve it
solve_problem!(rm_shell, solver)