using FerriteShells
using LinearAlgebra
using Printf

# Pinched hemispherical shell — Reissner–Mindlin (5 DOF/node) benchmark.
# Quarter symmetry model: polar angle θ ∈ [18°, 90°], azimuthal φ ∈ [0°, 90°].
# Parameters: R = 10, t = 0.04, E = 6.825×10⁷, ν = 0.3 (t/R = 0.004).
# Loads: P inward at A=(R,0,0); P outward at B=(0,R,0).
# Reference (linear, P=1): |u_x(A)| = 0.0924.
#
# Symmetry BCs. The displacement condition is the usual one (u_n = 0), but the director
# condition d·n = 0 must NOT be written as "fix φ₂": φ₁,φ₂ are components in the nodal
# frame, whose tangent vectors come from a heuristic that flips as the normal sweeps the
# sphere — on this geometry it flips right at the equator, where the load is applied, so
# fixing φ₂ silently clamps the shell there (u_x(A) comes out ~250× too small). Use
# `add_director_symmetry!`, which writes φ₁(T₁·n) + φ₂(T₂·n) = 0 in whatever frame each
# node carries. That requires per-node frames, so the assembly must reinit! with `nf`.
#
# NOTE: this benchmark is bending-dominated (t/R = 0.004), so MITC is needed. With MITC9,
# NodeFrames and the frame-independent symmetry BC it converges to the reference:
# 8×8 → 71% error, 16×16 → 14%, 32×32 → 0.2%.

function hemisphere_grid(n; R=10.0, θ_hole_deg=18.0)
    θ_min = θ_hole_deg * π / 180
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (n, n), Vec{2}((θ_min, 0.0)), Vec{2}((π/2, π/2)));
        map = nd -> (R*sin(nd.x[1])*cos(nd.x[2]), R*sin(nd.x[1])*sin(nd.x[2]), R*cos(nd.x[1])))
    addfacetset!(g, "sym_phi0",  x -> abs(x[2]) < 1e-10)
    addfacetset!(g, "sym_phi90", x -> abs(x[1]) < 1e-10)
    addnodeset!(g, "sym_phi0_n",  x -> abs(x[2]) < 1e-9)
    addnodeset!(g, "sym_phi90_n", x -> abs(x[1]) < 1e-9)
    addnodeset!(g, "load_A", x -> abs(x[3]) < 1e-6 && abs(x[2]) < 1e-6 && x[1] > 0.5R)
    addnodeset!(g, "load_B", x -> abs(x[3]) < 1e-6 && abs(x[1]) < 1e-6 && x[2] > 0.5R)
    return g
end

# material and grid
mat = LinearElastic(6.825e7, 0.3, 0.04)
grid = hemisphere_grid(32)

# interpolation space and shell with shear treatmens
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
nf = NodeFrames(grid, ip)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
add_director_symmetry!(ch, dh, nf, "sym_phi0_n",  Vec{3}((0.0, 1.0, 0.0)))
add_director_symmetry!(ch, dh, nf, "sym_phi90_n", Vec{3}((1.0, 0.0, 0.0)))
close!(ch); Ferrite.update!(ch, 0.0)

# allocate matrices and vectors
N      = ndofs(dh)
n_base = getnbasefunctions(ip)
K      = allocate_matrix(dh, ch)   # ch: the affine constraints add coupling entries
f      = zeros(N)
ke     = zeros(5n_base, 5n_base)
re     = zeros(5n_base)

# assemble once
asm = start_assemble(K, zeros(N))
for cell in CellIterator(dh)
    fill!(ke, 0.0)
    reinit!(scv, cell, nf)   # per-node frames — the frame the symmetry BC is written in
    u0 = zeros(5n_base)
    membrane_tangent_RM!(ke, scv, u0, mat)
    bending_tangent_RM!(ke, scv, u0, mat)
    assemble!(asm, shelldofs(cell), ke, re)
end

# apply loading
apply_pointload!(f, dh, "load_A", Vec{3}((-1.0, 0.0, 0.0)))
apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0, 1.0, 0.0)))
apply!(K, f, ch)

#solve and time it
@time u_sol = K \ f
apply!(u_sol, ch)   # recover the affine-constrained φ DOFs

# extract solution at point
ph     = PointEvalHandler(grid, [grid.nodes[first(grid.nodesets["load_A"])].x])
u_eval = first(evaluate_at_points(ph, dh, u_sol, :u))
println("FerriteShell.jl: ", round(u_eval[1],digits=4))
println("reference      : -0.0924")
# save
VTKGridFile("pinched_hemisphere", dh) do vtk
    write_solution(vtk, dh, u_sol)
end
