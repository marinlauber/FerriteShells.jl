using FerriteShells
using LinearAlgebra
using Printf

# Pinched hemispherical shell — Reissner–Mindlin (5 DOF/node) benchmark.
# Quarter symmetry model: polar angle θ ∈ [18°, 90°], azimuthal φ ∈ [0°, 90°].
# Parameters: R = 10, t = 0.04, E = 6.825×10⁷, ν = 0.3 (t/R = 0.004).
# Loads: P inward at A=(R,0,0); P outward at B=(0,R,0).
# Reference (linear, P=1): |u_x(A)| = 0.0924.
#
# Symmetry BCs (derived from director d = cosθ·G₃ + sincθ·(φ₁T₁ + φ₂T₂)):
#   φ=0  plane (x-z, y=0): T₂=ê_y   → d_y = sincθ·φ₂ = 0 → fix u_y, φ₂
#   φ=π/2 plane (y-z, x=0): T₂=−ê_x → d_x = −sincθ·φ₂ = 0 → fix u_x, φ₂
#
# NOTE: this benchmark is bending-dominated (t/R = 0.004). Standard displacement-based
# RM elements suffer from membrane locking here — convergence is monotone but slow.
# MITC or EAS elements are needed for practical accuracy on coarse meshes.

function hemisphere_grid(n; R=10.0, θ_hole_deg=18.0)
    θ_min = θ_hole_deg * π / 180
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (n, n), Vec{2}((θ_min, 0.0)), Vec{2}((π/2, π/2)));
        map = nd -> (R*sin(nd.x[1])*cos(nd.x[2]), R*sin(nd.x[1])*sin(nd.x[2]), R*cos(nd.x[1])))
    addfacetset!(g, "sym_phi0",  x -> abs(x[2]) < 1e-10)
    addfacetset!(g, "sym_phi90", x -> abs(x[1]) < 1e-10)
    addnodeset!(g, "load_A", x -> abs(x[3]) < 1e-6 && abs(x[2]) < 1e-6 && x[1] > 0.5R)
    addnodeset!(g, "load_B", x -> abs(x[3]) < 1e-6 && abs(x[1]) < 1e-6 && x[2] > 0.5R)
    return g
end

const mat = LinearElastic(6.825e7, 0.3, 0.04)
const ip  = Lagrange{RefQuadrilateral, 2}()
const qr  = QuadratureRule{RefQuadrilateral}(3)

println("Pinched hemisphere RM (Q9): mesh convergence, P=1")
println("   n | elements |   u_x(A)   |  ref=-0.0924 | error(%)")

for n in [2, 4, 8, 16, 32]
    grid = hemisphere_grid(n)
    scv  = ShellCellValues(qr, ip, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi90"), x -> 0.0, [2]))
    close!(ch); Ferrite.update!(ch, 0.0)

    N      = ndofs(dh)
    n_base = getnbasefunctions(ip)
    K      = allocate_matrix(dh)
    f      = zeros(N)
    ke     = zeros(5n_base, 5n_base)
    re     = zeros(5n_base)

    asm = start_assemble(K, zeros(N))
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        reinit!(scv, cell)
        u0 = zeros(5n_base)
        membrane_tangent_RM!(ke, scv, u0, mat)
        bending_tangent_RM!(ke, scv, u0, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end

    apply_pointload!(f, dh, "load_A", Vec{3}((-1.0, 0.0, 0.0)))
    apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0, 1.0, 0.0)))
    apply!(K, f, ch)
    u_sol = K \ f

    A_node = only(getnodeset(grid, "load_A"))
    u_x_A  = 0.0
    for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
        gid == A_node || continue
        u_x_A = u_sol[celldofs(cell)[3I-2]]
    end
    @printf("  %2d |    %4d  | %10.6f | %12.6f | %6.2f\n",
            n, getncells(grid), u_x_A, -0.0924, abs(u_x_A + 0.0924) / 0.0924 * 100)
end

# VTK output at n=8
let n = 8
    grid = hemisphere_grid(n)
    scv  = ShellCellValues(qr, ip, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sym_phi90"), x -> 0.0, [1]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi0"),  x -> 0.0, [2]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_phi90"), x -> 0.0, [2]))
    close!(ch); Ferrite.update!(ch, 0.0)

    N      = ndofs(dh)
    n_base = getnbasefunctions(ip)
    K      = allocate_matrix(dh)
    f      = zeros(N)
    ke     = zeros(5n_base, 5n_base)
    re     = zeros(5n_base)

    asm = start_assemble(K, zeros(N))
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        reinit!(scv, cell)
        u0 = zeros(5n_base)
        membrane_tangent_RM!(ke, scv, u0, mat)
        bending_tangent_RM!(ke, scv, u0, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end

    apply_pointload!(f, dh, "load_A", Vec{3}((-1.0, 0.0, 0.0)))
    apply_pointload!(f, dh, "load_B", Vec{3}(( 0.0, 1.0, 0.0)))
    apply!(K, f, ch)
    u_sol = K \ f

    VTKGridFile("pinched_hemisphere", dh) do vtk
        write_solution(vtk, dh, u_sol)
    end
    println("VTK written to pinched_hemisphere.vtu")
end
