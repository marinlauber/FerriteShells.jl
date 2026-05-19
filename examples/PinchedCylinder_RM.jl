using FerriteShells

# Pinched cylinder — Reissner-Mindlin shell (1/8 symmetry model)
# Same geometry, loading, and BCs as PinchedCylinder.jl; see that file for full description.
# Two-field DofHandler (:u ip^3, :θ ip^2).  DOF reordering via shelldofs().

function pinched_cylinder_rm_grid(ns, na)
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (ns, na),
                      Vec{2}((0.0, 0.0)), Vec{2}((π/2, 600.0/2)));
        map = n -> (n.x[2], 300.0 * sin(n.x[1]), 300.0 * cos(n.x[1])))
    addnodeset!(g, "diaphragm",   x -> x[1] ≈ 0.0)
    addnodeset!(g, "sym_axial",   x -> x[1] ≈ 600.0/2)
    addnodeset!(g, "sym_theta0",  x -> abs(x[2]) < 1e-6)
    addnodeset!(g, "sym_theta90", x -> abs(x[3]) < 1e-6)
    addnodeset!(g, "load_point",
        x -> x[1] ≈ 600.0/2 && abs(x[2]) < 1e-6 && abs(x[3] - 300.0) < 1e-6)
    return g
end

# interplation space
ip  = Lagrange{RefQuadrilateral, 2}() # Q9
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)

# material
mat = LinearElastic(3.0e6, 0.3, 3.0)

# make grid
grid = pinched_cylinder_rm_grid(32, 32)

# degrees of freedom
dh   = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# assembly
n_base = getnbasefunctions(ip)
K  = allocate_matrix(dh)
f  = zeros(ndofs(dh))
asmb = start_assemble(K, zeros(ndofs(dh)))
ke = zeros(5n_base, 5n_base); re = zeros(5n_base)

for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(re, 0.0)
    reinit!(scv, cell)
    u0 = zeros(5n_base)
    membrane_tangent_RM!(ke, scv, u0, mat)
    bending_tangent_RM!(ke, scv, u0, mat)
    sd = shelldofs(cell)
    assemble!(asmb, sd, ke, re)
end

apply_pointload!(f, dh, "load_point", Vec{3}((0.0, 0.0, -1/4)))

dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"),   x -> zeros(2), [2, 3]))
add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_axial"),   x -> 0.0,      [1]))
add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta0"),  x -> 0.0,      [2]))
add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta90"), x -> 0.0,      [3]))
# Rotation symmetry BCs: director must also be symmetric at each symmetry plane.
# At θ=0  (T₁=e_y):   d_y = φ₁ = 0  →  fix :θ component 1
# At θ=π/2 (T₁=−e_z): d_z = −φ₁ = 0 → fix :θ component 1
# At x=L/2 (T₂=e_x):  d_x = φ₂ = 0  →  fix :θ component 2
add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta0"),  x -> 0.0, [1]))
add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_theta90"), x -> 0.0, [1]))
add!(dbc, Dirichlet(:θ, getnodeset(grid, "sym_axial"),   x -> 0.0, [2]))
close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

u_sol = K \ f

# write to vtk
VTKGridFile("pinched_cylinder", dh) do vtk
    write_solution(vtk, dh, u_sol)
end

# extract solution at point
ph     = PointEvalHandler(grid, [Vec{3}(([300.0, 0.0, 300.0]))])
u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))

println("Pinched cylinder (RM, 32×32): u_z at load point = $(u_eval[3]) (reference: -1.8248e-5)")
