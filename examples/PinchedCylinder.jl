using FerriteShells

# Pinched cylinder — Kirchhoff-Love shell (1/8 symmetry model)
#
# Full cylinder: R=300, L=600, t=3, E=3e6, ν=0.3.
# Two opposite inward point loads P=1 at the equator (x=L/2, θ=0 and θ=π).
# BCs: rigid diaphragm at both ends; symmetry planes at x=L/2, θ=0, θ=π/2.
# Reference radial deflection at load point: 1.8248e-5 (inward).
#
# 1/8 model: θ ∈ [0,π/2], axial ∈ [0,L/2].
# Coordinate map: (θ,axial) → (axial, R·sinθ, R·cosθ)
#   Load point (θ=0, axial=L/2):  X = (L/2, 0,   R)  — load P/4 in -z
#   Other corner (θ=π/2, axial=0): X = (0,   R,   0)
#
# Symmetry BCs (1/8 model):
#   x=0    (diaphragm end): u_y=u_z=0
#   x=L/2  (mid-plane):     u_x=0
#   θ=0    (y=0 plane):     u_y=0
#   θ=π/2  (z=0 plane):     u_z=0

function pinched_cylinder_grid(ns, na)
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
grid = pinched_cylinder_grid(16, 16)

# degrees of freedom
dh   = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

# assembly
n_el   = ndofs_per_cell(dh)
K  = allocate_matrix(dh)
f  = zeros(ndofs(dh))
asmb = start_assemble(K, zeros(ndofs(dh)))
ke = zeros(n_el, n_el); re = zeros(n_el)

for cell in CellIterator(dh)
    fill!(ke, 0.0); fill!(re, 0.0)
    reinit!(scv, cell)
    u0 = zeros(n_el)
    membrane_tangent_KL!(ke, scv, u0, mat)
    bending_tangent_KL!(ke, scv, u0, mat)
    assemble!(asmb, celldofs(cell), ke, re)
end

apply_pointload!(f, dh, "load_point", Vec{3}((0.0, 0.0, -1/4)))

dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getnodeset(grid, "diaphragm"),  x -> zeros(2), [2, 3]))
add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_axial"),  x -> 0.0,      [1]))
add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta0"), x -> 0.0,      [2]))
add!(dbc, Dirichlet(:u, getnodeset(grid, "sym_theta90"), x -> 0.0,     [3]))
close!(dbc); Ferrite.update!(dbc, 0.0); apply!(K, f, dbc)

u_sol = K \ f

# extract solution at point
ph     = PointEvalHandler(grid, [Vec{3}(([300.0, 0.0, 300.0]))])
u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))

println("Pinched cylinder (KL, 16×16): u_z at load point = $(u_eval[3])  (reference: -1.8248e-5)")
