using FerriteShells, LinearAlgebra, SparseArrays, Printf, WriteVTK, Pardiso

# Dynamic inflation of a square airbag (quarter model) with a threaded element loop, the
# fused Reissner–Mindlin kernels and the threaded MKL Pardiso solver. Same problem as
# `SquareAirbagDynamic.jl`; here each Newton iteration evaluates internal forces, follower
# pressure and their tangents in a single threaded pass over the cells. Start Julia with
# threads, e.g. `julia -t auto`.

# MKL Pardiso behind the `lu!` / `ldiv!` interface. The sparsity pattern of K_eff is fixed,
# so the symbolic analysis runs once at construction and `lu!` only refactorizes
# numerically. Pardiso reads CSR; a CSC matrix is the CSR of its transpose, so
# `fix_iparm!(ps, :N)` makes the solve act on A itself. `A` is kept because iterative
# refinement reads its values: `lu!` must be called with that same matrix, updated in place.
# Every phase must run with the thread count of the analysis (fewer threads silently gives
# a wrong solution), and the MKL thread count is process-global, so it is re-set per call.
struct PardisoLU{M}
    ps::MKLPardisoSolver
    A::M
    colptr::Vector{Int32}
    rowval::Vector{Int32}
    nthreads::Int
end
function PardisoLU(A; nthreads, mtype=Pardiso.REAL_SYM)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, mtype)   # REAL_SYM: structurally symmetric pattern, unsymmetric values
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    F = PardisoLU(ps, A, convert(Vector{Int32}, A.colptr), convert(Vector{Int32}, A.rowval), nthreads)
    _pardiso!(F, Pardiso.ANALYSIS, Float64[], Float64[])
    return F
end
function _pardiso!(F::PardisoLU, phase, x, b)
    set_nprocs!(F.ps, F.nthreads)
    set_phase!(F.ps, phase)
    Pardiso.ccall_pardiso(F.ps, length(F.colptr) - 1, F.A.nzval, F.colptr, F.rowval, 1, b, x)
end
function LinearAlgebra.lu!(F::PardisoLU, A)
    A === F.A || throw(ArgumentError("PardisoLU refactorizes the matrix it was built on"))
    _pardiso!(F, Pardiso.NUM_FACT, Float64[], Float64[])
    return F
end
LinearAlgebra.ldiv!(x, F::PardisoLU, b) = (_pardiso!(F, Pardiso.SOLVE_ITERATIVE_REFINE, x, b); x)

function make_quarter_pillow_grid(n; L=1.0, primitive=QuadraticQuadrilateral)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid = shell_grid(generate_grid(primitive, (n, n), corners))
    addfacetset!(grid, "edge",  x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addfacetset!(grid, "sym_x", x -> isapprox(x[1], 0.0, atol=1e-10))
    addfacetset!(grid, "sym_y", x -> isapprox(x[2], 0.0, atol=1e-10))
    return grid
end

# Per-task buffers: a private `ShellCellValues` copy, cell cache, element arrays and an
# assembler. All assemblers write into the same global K and r; `fillzero=false` because
# the caller zeroes them once before the threaded loop.
struct Scratch{SCV,CC,A}
    scv::SCV
    cc::CC
    sd::Vector{Int}
    u_e::Vector{Float64}
    ke::Matrix{Float64}
    re::Vector{Float64}
    asm::A
end

function Scratch(dh, scv, K, r)
    n_e = ndofs_per_cell(dh)
    Scratch(copy(scv), CellCache(dh), Int[], zeros(n_e), zeros(n_e, n_e), zeros(n_e),
            start_assemble(K, r; fillzero=false))
end

# Cells of one colour share no dofs (Ferrite `create_coloring`), so the tasks working on a
# colour scatter into K and r without races. Tasks pull chunks of a colour from a shared
# counter, which balances the load across cores of unequal speed.
function foreach_colored!(kernel!, colors, scratch; chunk=8)
    for color in colors
        next = Threads.Atomic{Int}(1)
        @sync for s in scratch
            Threads.@spawn while (i = Threads.atomic_add!(next, chunk)) <= length(color)
                for j in i:min(i + chunk - 1, length(color))
                    kernel!(s, color[j])
                end
            end
        end
    end
end

# K = K_int − p K_p and r = r_int − p F_p in one pass. The fused kernel evaluates membrane,
# bending and shear with one kinematics and material evaluation per quadrature point; the
# pressure kernels accumulate into the same element arrays with load −p.
function assemble_all_threaded!(K, r, dh, colors, scratch, u, mat, p)
    fill!(K.nzval, 0.0); fill!(r, 0.0)
    sdh = dh.subdofhandlers[1]
    foreach_colored!(colors, scratch) do s, cid
        reinit!(s.cc, cid)
        reinit!(s.scv, s.cc)
        shelldofs!(s.sd, sdh, s.cc)
        s.u_e .= @view u[s.sd]
        fill!(s.ke, 0.0); fill!(s.re, 0.0)
        residuals_and_tangent_RM!(s.ke, s.re, s.scv, s.u_e, mat)
        assemble_pressure!(s.re, s.scv, s.u_e, -p)
        assemble_pressure_tangent!(s.ke, s.scv, s.u_e, -p)
        assemble!(s.asm, s.sd, s.ke, s.re)
    end
    return K, r
end

# Serial reference built from the split kernels, for checking the threaded assembly.
function assemble_all_serial!(K, r, dh, scv, u, mat, p)
    n_e = ndofs_per_cell(dh)
    ke = zeros(n_e, n_e); re = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re, scv, u_e, mat); bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat);   bending_tangent_RM!(ke, scv, u_e, mat)
        assemble_pressure!(re, scv, u_e, -p);      assemble_pressure_tangent!(ke, scv, u_e, -p)
        assemble!(asm, sd, ke, re)
    end
    return K, r
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

# material
mat = LinearElastic(1.0e6, 0.3, 0.002)
ρ   = 1200.0       # density [kg/m³]

# grid and interpolation space with shear treatment of the shell
grid = make_quarter_pillow_grid(32; L=1.0, primitive=QuadraticQuadrilateral)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "edge"),  x -> 0.0, [3]))          # u_z=0 at boundary
add!(ch, Dirichlet(:θ, getfacetset(grid, "edge"),  x -> zeros(2), [1,2]))   # φ=0 at boundary
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_x"), x -> 0.0, [1]))          # u_x=0 at x=0
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_x"), x -> 0.0, [1]))          # φ₁=0 at x=0
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_y"), x -> 0.0, [2]))          # u_y=0 at y=0
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_y"), x -> 0.0, [2]))          # φ₂=0 at y=0
close!(ch); Ferrite.update!(ch, 0.0)

# allocate
N_dof = ndofs(dh)
free  = ch.free_dofs
K_tan = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
M     = allocate_matrix(dh)
r     = zeros(N_dof)
g_old = zeros(N_dof)   # (1−α) term from previous step: C·v_n + r_int(u_n) − p_n·F_p(u_n)
rhs   = zeros(N_dof)
δu    = zeros(N_dof)

# one scratch per thread, cells grouped into colours with no shared dofs
colors  = create_coloring(grid)
scratch = [Scratch(dh, scv, K_tan, r) for _ in 1:Threads.nthreads()]
@printf("%d dofs, %d cells in %d colours, %d threads\n",
        N_dof, getncells(grid), length(colors), Threads.nthreads())

# check the threaded assembly against the serial one on a deformed, pressurised state
let u_test = 1e-3 .* randn(N_dof), K_s = allocate_matrix(dh), r_s = zeros(N_dof)
    t_s = @elapsed assemble_all_serial!(K_s, r_s, dh, scv, u_test, mat, 100.0)
    t_t = @elapsed assemble_all_threaded!(K_tan, r, dh, colors, scratch, u_test, mat, 100.0)
    K_tan ≈ K_s || error("threaded stiffness disagrees with serial assembly")
    r ≈ r_s     || error("threaded residual disagrees with serial assembly")
    t_s = @elapsed assemble_all_serial!(K_s, r_s, dh, scv, u_test, mat, 100.0)
    t_t = @elapsed assemble_all_threaded!(K_tan, r, dh, colors, scratch, u_test, mat, 100.0)
    @printf("assembly: serial split %.3f s, threaded fused %.3f s (%.1f×)\n", t_s, t_t, t_s / t_t)
end

# fill the mass terms
assemble_mass!(M, dh, scv, ρ, mat)

# sim params
T_sim     = 2.0     # total simulation  [s]
T_ramp    = 1.0     # ramp time for pressure load
Δt        = 0.001   # time step         [s]
n_steps   = Int(T_sim / Δt)
out_every = 10      # VTK output interval [steps]
outdir    = "pillow_dynamic_mt"

# HHT-α parameters
α_hht   = -0.05
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
α_damp  = 10.0    # mass-proportional Rayleigh damping coefficient [1/s]
tol      = 1e-6
max_iter = 50

# ramp the pressure slowly
ramp(t) = t < T_ramp ? 0.5 * (1 - cos(π * t / T_ramp)) : 1.0

# Effective mass factor (M coefficient in K_eff)
m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

# Linear solver: threaded MKL Pardiso on 8 physical (performance) cores.
pardiso_threads = min(8, Sys.CPU_THREADS)

# Initial symbolic factorisation (reused each Newton step via lu!)
assemble_all_threaded!(K_tan, r, dh, colors, scratch, zeros(N_dof), mat, 0.0)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_tan.nzval
apply_zero!(K_eff, r, ch)
F_lu = PardisoLU(K_eff; nthreads=pardiso_threads)

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
ũ = zeros(N_dof)
ṽ = zeros(N_dof)
a = zeros(N_dof)
a_new = zeros(N_dof)
v_new = zeros(N_dof)
u_new = zeros(N_dof)

mkpath(outdir)
pvd = paraview_collection(joinpath(outdir, "pillow_dynamic"))
VTKGridFile(joinpath(outdir, "pillow_dynamic-0"), dh) do vtk
    write_solution(vtk, dh, u)
    pvd[0.0] = vtk
end

@printf("%-6s  %-8s  %-8s  %-8s  %-6s\n", "step", "t [s]", "λ", "p [Pa]", "iters")

t_asm = 0.0; t_lin = 0.0
t_run = @elapsed for step in 1:n_steps
    global t_asm, t_lin
    t_new = step * Δt
    p_new = 2500 * ramp(t_new)
    # Predictor
    ũ .= u .+ Δt .* v .+ (Δt^2 * (0.5 - β_hht)) .* a
    ṽ .= v .+ (Δt * (1 - γ_hht)) .* a
    u_new .= ũ
    apply!(u_new, ch)
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        t_asm += @elapsed assemble_all_threaded!(K_tan, r, dh, colors, scratch, u_new, mat, p_new)

        a_new .= (u_new .- ũ) ./ (β_hht * Δt^2)
        v_new .= ṽ .+ (Δt * γ_hht) .* a_new

        # HHT residual: M ä + (1−α)[C v + r_int − p F_p] + α g_old = 0
        rhs .= -(M * a_new .+ (1 - α_hht) .* (α_damp .* (M * v_new) .+ r) .+ α_hht .* g_old)
        apply_zero!(rhs, ch)

        norm(@views rhs[free]) < tol && (converged = true; break)

        t_lin += @elapsed begin
            K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_tan.nzval
            apply_zero!(K_eff, rhs, ch)
            lu!(F_lu, K_eff)
            ldiv!(δu, F_lu, rhs)
        end
        u_new .+= δu
        apply!(u_new, ch)
    end

    !converged && @warn "Step $step (t=$(round(t_new, digits=3)) s): no convergence in $max_iter iters"

    a .= (u_new .- ũ) ./ (β_hht * Δt^2)
    v .= ṽ .+ (Δt * γ_hht) .* a
    # g_old = C·v + r_int(u) − p·F_p(u) for the α-offset in the next step; `r` holds the
    # last two terms at the converged state
    g_old .= α_damp .* (M * v) .+ r
    u .= u_new

    if step % out_every == 0
        VTKGridFile(joinpath(outdir, "pillow_dynamic-$(step ÷ out_every)"), dh) do vtk
            write_solution(vtk, dh, u)
            pvd[t_new] = vtk
        end
        @printf("%-6d  %-8.3f  %-8.4f  %-8.1f  %-6d\n", step, t_new, ramp(t_new), p_new, iters)
    end
end
close(pvd)

@printf("total %.1f s: assembly %.1f s, linear solve %.1f s\n", t_run, t_asm, t_lin)
ph = PointEvalHandler(grid, [Vec{3}((0.0, 0.0, 0.0))])
@printf("centre deflection u_z = %.4f m\n", first(evaluate_at_points(ph, dh, u, :u))[3])
