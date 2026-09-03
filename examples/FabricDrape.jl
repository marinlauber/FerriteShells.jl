using FerriteShells, LinearAlgebra, Printf, WriteVTK

# Draping of a square fabric sheet over a smaller rigid square block.
#
#   ┌───────────────────────┐   ← fabric  [-L/2, L/2]²  (thin RM shell, Q9 + MITC9)
#   │        overhang       │
#   │     ┌───────────┐     │   ← rigid block footprint [-a/2, a/2]²  (a < L)
#   │     │  on-block │     │
#   │     │  (u_z=0)  │     │
#   │     └───────────┘     │
#   │        overhang       │
#   └───────────────────────┘
#
# The two squares are concentric. The block is NOT meshed: it enters only through
# boundary conditions on the shell — every node whose (x,y) lies inside the block
# footprint is held at u_z = 0 (rests flat on the block's top face). The parts of
# the sheet hanging past the block edges are free and drape down under gravity.
#
# Solved by DYNAMIC RELAXATION: integrate the transient equations of motion
#   M ü + C u̇ + R_int(u) = f_grav ,   C = α_damp · M  (mass-proportional damping)
# with HHT-α time integration until the kinetic energy dies out and the sheet
# settles into its steady static drape. Dynamic relaxation sails through the
# flat-reference membrane nonlinearity (a Newton step from flat injects large
# geometric membrane strains that defeat load-controlled statics) because inertia
# and damping regularise the otherwise near-singular transverse response.
#
# Gravity is a constant body force −g·ê_z, applied as a consistent load
#   f_grav_I = ∫ ρ t (−g) N_I dΩ , ramped 0→1 over T_ramp so the sheet accelerates
# gently. Rigid-body modes are removed consistently with the 4-fold symmetry:
#   u_z = 0 on the block footprint      → kills Tz, Rx, Ry
#   u_x = 0 on the x = 0 symmetry line  → kills Tx and Rz
#   u_y = 0 on the y = 0 symmetry line  → kills Ty

function make_fabric_grid(n; L=1.0, a=0.4)
    corners = [Vec{2}((-L/2, -L/2)), Vec{2}((L/2, -L/2)),
               Vec{2}(( L/2,  L/2)), Vec{2}((-L/2, L/2))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n, n), corners))
    tol  = 1e-8
    addnodeset!(grid, "block", x -> abs(x[1]) ≤ a/2 + tol && abs(x[2]) ≤ a/2 + tol)  # rests on block top
    addnodeset!(grid, "sym_x", x -> abs(x[1]) < tol)   # x = 0 symmetry plane
    addnodeset!(grid, "sym_y", x -> abs(x[2]) < tol)   # y = 0 symmetry plane
    addnodeset!(grid, "corner", x -> isapprox(abs(x[1]), L/2; atol=tol) &&
                                     isapprox(abs(x[2]), L/2; atol=tol))
    return grid
end

function assemble_all!(K_int, r_int, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e); re = zeros(n_e)
    asm = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        assemble!(asm, sd, ke, re)
    end
end

function assemble_mass!(M, dh, scv, ρ, mat)
    n_e = ndofs_per_cell(dh)
    me  = zeros(n_e, n_e)
    asm = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(me, 0.0)
        reinit!(scv, cell)
        mass_matrix!(me, scv, ρ, mat)
        assemble!(asm, shelldofs(cell), me)
    end
end

# Consistent gravity load: f_I[u_z] = ∫ ρ t g_z N_I dΩ (reference configuration).
function assemble_gravity!(f, dh, scv, ρ, g_z, mat)
    n_e     = ndofs_per_cell(dh)
    n_nodes = getnbasefunctions(scv.ip_shape)
    fe      = zeros(n_e)
    for cell in CellIterator(dh)
        fill!(fe, 0.0)
        reinit!(scv, cell)
        for qp in 1:getnquadpoints(scv)
            dΩ = scv.detJdV[qp]
            for I in 1:n_nodes
                fe[5I-2] += ρ * mat.thickness * g_z * scv.N[I, qp] * dΩ
            end
        end
        f[shelldofs(cell)] .+= fe
    end
end

# global u_z DOF of the fabric corner node (the point that drapes most)
function corner_uz_dof(dh, grid)
    cid = minimum(getnodeset(grid, "corner"))   # any one of the four corners (all equivalent by symmetry)
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            gid == cid && return celldofs(cell)[3I]
        end
    end
    return 0
end

n   = 20                        # elements per side (n even → block edge a/2 lands on a node line)
L   = 1.0                       # fabric side length [m]
a   = 0.4                       # rigid block footprint side length [m]  (a < L)
E   = 5.0e8                     # Young's modulus [Pa]  (lower → softer, more pronounced drape)
ν   = 0.3
t   = 1.0e-3                    # fabric thickness [m]
ρ   = 1.0e3                     # density [kg/m³]
g   = 9.81                      # gravitational acceleration [m/s²]

mat = LinearElastic(E, ν, t)

grid = make_fabric_grid(n; L, a)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "block"), x -> 0.0, [3]))   # on-block: no vertical penetration
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_x"), x -> 0.0, [1]))   # symmetry: u_x = 0 on x = 0
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_y"), x -> 0.0, [2]))   # symmetry: u_y = 0 on y = 0
close!(ch); Ferrite.update!(ch, 0.0)

N_dof = ndofs(dh)
free  = ch.free_dofs

K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
M     = allocate_matrix(dh)
r_int = zeros(N_dof)
g_old = zeros(N_dof)
rhs   = zeros(N_dof)
δu    = zeros(N_dof)

assemble_mass!(M, dh, scv, ρ, mat)

# full gravity load vector (scaled by the ramp each step)
f_grav = zeros(N_dof)
assemble_gravity!(f_grav, dh, scv, ρ, -g, mat)
f_step = zeros(N_dof)

# dynamic-relaxation parameters
T_ramp  = 1.0        # gravity ramp 0→1 [s]
T_sim   = 4.0        # total time — long enough for the drape to settle [s]
Δt      = 0.005      # time step [s]
n_steps = Int(round(T_sim / Δt))

# mass-proportional damping ≈ critical for the fundamental drape mode
# (overhang cantilever ω₁ ≈ 3.516/ℓ² · √(E t²/12ρ) ≈ 8 rad/s → c ≈ 2ω₁).
# Softer fabric (lower E) lowers ω₁ ∝ √E, so scale α_damp down accordingly.
α_damp  = 16.0       # [1/s]

# HHT-α integration (mild numerical damping of high frequencies)
α_hht   = -0.05
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
tol      = 1e-6
max_iter = 30

ramp(t) = t < T_ramp ? 0.5 * (1 - cos(π * t / T_ramp)) : 1.0
m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

# initial symbolic LU factorisation (reused each Newton step via lu!)
assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

w_dof = corner_uz_dof(dh, grid)

println("FerriShells.jl - runing")
println("  E=$E, t=$t, ρ=$ρ, α_damp=$α_damp, Δt=$Δt, T_sim=$T_sim")
@printf("%-6s  %-8s  %-6s  %-12s  %-11s  %-6s\n", "step", "t [s]", "λ", "u_z(corner)", "‖v‖", "iters")

u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
ũ = zeros(N_dof); ṽ = zeros(N_dof); a = zeros(N_dof)
u_new = zeros(N_dof); v_new = zeros(N_dof); a_new = zeros(N_dof)

pvd = paraview_collection("fabric_drape")
vtk_step = Ref(0)
VTKGridFile("fabric_drape-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

for step in 1:n_steps
    t_new = step * Δt
    f_step .= ramp(t_new) .* f_grav

    # predictor
    ũ .= u .+ Δt .* v .+ (Δt^2 * (0.5 - β_hht)) .* a
    ṽ .= v .+ (Δt * (1 - γ_hht)) .* a
    u_new .= ũ; apply!(u_new, ch)

    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat)

        a_new .= (u_new .- ũ) ./ (β_hht * Δt^2)
        v_new .= ṽ .+ (Δt * γ_hht) .* a_new

        # HHT residual: M ä + (1−α)[C v + r_int − f_grav] + α g_old = 0
        rhs .= -(M * a_new .+ (1 - α_hht) .* (α_damp .* (M * v_new) .+ r_int .- f_step) .+ α_hht .* g_old)
        apply_zero!(rhs, ch)
        norm(@views rhs[free]) < tol && (converged = true; break)

        K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        u_new .+= δu
        apply!(u_new, ch)
    end

    !converged && @warn "Step $step (t=$(round(t_new, digits=3)) s): no convergence in $max_iter iters"

    a .= (u_new .- ũ) ./ (β_hht * Δt^2)
    v .= ṽ .+ (Δt * γ_hht) .* a
    g_old .= α_damp .* (M * v) .+ r_int .- f_step
    u .= u_new

    if step % 10 == 0
        vtk_step[] += 1
        VTKGridFile("fabric_drape-$(vtk_step[])", dh) do vtk
            write_solution(vtk, dh, u); pvd[t_new] = vtk
        end
        @printf("%-6d  %-8.3f  %-6.3f  %-12.4e  %-11.3e  %-6d\n",
                step, t_new, ramp(t_new), u[w_dof], norm(@views v[free]), iters)
    end
end
vtk_save(pvd)
println("Steady drape reached: u_z(corner) = ", u[w_dof], " m,  residual velocity ‖v‖ = ", norm(@views v[free]))
