using FerriteShells,LinearAlgebra,Printf,WriteVTK

function make_quarter_pillow_grid(n; L=1.0, primitive=QuadraticQuadrilateral)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid = shell_grid(generate_grid(primitive, (n, n), corners))
    addfacetset!(grid, "edge",  x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addfacetset!(grid, "sym_x", x -> isapprox(x[1], 0.0, atol=1e-10))
    addfacetset!(grid, "sym_y", x -> isapprox(x[2], 0.0, atol=1e-10))
    addnodeset!(grid, "center", x -> norm(x) < 1e-10)
    return grid
end

function assemble_all!(K_int, r_int, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        membrane_tangent_RM!(ke_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        assemble!(asm_i, sd, ke_i, re_i)
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

function max_principal_stress_qp(dh, scv, u, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    n_qp    = getnquadpoints(scv)
    u_disp  = zeros(3 * n_nodes)
    out     = [zeros(n_qp) for _ in 1:getncells(dh.grid)]
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        @inbounds for I in 1:n_nodes
            u_disp[3I-2] = u_e[5I-4]; u_disp[3I-1] = u_e[5I-3]; u_disp[3I] = u_e[5I-2]
        end
        qp_vals = out[cellid(cell)]
        @inbounds for qp in 1:n_qp
            _, _, E = FerriteShells.kinematics_strains(scv, qp, u_disp)
            N  = FerriteShells.contravariant_elasticity(mat, scv.A_metric[qp]) ⊡ E
            N11, N12, N22 = N[1,1], N[1,2], N[2,2]
            qp_vals[qp] = (N11 + N22)/2 + sqrt(((N11 - N22)/2)^2 + N12^2)
        end
    end
    out
end

function assemble_pressure_all!(K_p, F_p, dh, scv, u)
    n_e  = ndofs_per_cell(dh)
    ke_p = zeros(n_e, n_e); re_p = zeros(n_e)
    asm  = start_assemble(K_p)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh)
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        assemble_pressure!(re_p, scv, u_e, 1.0)
        assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm, sd, ke_p)
        @views F_p[sd] .+= re_p
    end
end

# material
mat = LinearElastic(1.0e6, 0.3, 0.009)
ρ   = 1200.0       # density [kg/m³]

grid = make_quarter_pillow_grid(32; L=1.0, primitive=QuadraticQuadrilateral)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

proj = L2Projector(ip, grid)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "edge"),  x -> 0.0, [3]))          # u_z=0 at boundary
add!(ch, Dirichlet(:θ, getfacetset(grid, "edge"),  x -> zeros(2), [1,2]))   # φ=0 at boundary
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_x"), x -> 0.0, [1]))          # u_x=0 at x=0
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_x"), x -> 0.0, [1]))          # φ₁=0 at x=0 (∂w/∂x=0)
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_y"), x -> 0.0, [2]))          # u_y=0 at y=0
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_y"), x -> 0.0, [2]))          # φ₂=0 at y=0 (∂w/∂y=0)
close!(ch); Ferrite.update!(ch, 0.0)

# allocate
N_dof = ndofs(dh)
free  = ch.free_dofs
K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
K_plv = allocate_matrix(dh)
M     = allocate_matrix(dh)
r_int = zeros(N_dof)
F_plv = zeros(N_dof)
g_old = zeros(N_dof)   # (1−α) term from previous step: C·v_n + r_int(u_n) − p_n·F_p(u_n)
R     = zeros(N_dof)
δu    = zeros(N_dof)

# fill the mass terms
assemble_mass!(M, dh, scv, ρ, mat)

# sim params
T_sim   = 2.0   # total simulation  [s]
T_ramp  = 1.0    # ramp time for pressure load
Δt      = 0.01   # time step         [s]
n_steps = Int(T_sim / Δt)

# HHT-α parameters  (α = −0.3: strong high-frequency damping, still stable)
α_hht   = -0.05
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
α_damp  = 10.0    # mass-proportional Rayleigh damping coefficient [1/s]
tol      = 1e-6
max_iter = 50

ramp(t) = t < T_ramp ? 0.5 * (1 - cos(π * t / T_ramp)) : 1.0

# Effective mass factor (M coefficient in K_eff, updated when Δt changes)
m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

# Initial symbolic LU factorisation (reused each Newton step via lu!)
assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
a = zeros(N_dof)

pvd = paraview_collection("pillow_dynamic")
vtk_step = Ref(0)
VTKGridFile("pillow_dynamic-0", dh) do vtk
    write_solution(vtk, dh, u)
    write_projection(vtk, proj, project(proj, max_principal_stress_qp(dh, scv, u, mat), qr), "sigma_max")
    pvd[0.0] = vtk
end

@printf("%-6s  %-8s  %-8s  %-8s  %-6s\n", "step", "t [s]", "λ", "p [mmHg]", "iters")

for step in 1:n_steps
    t_new = step * Δt
    p_new = 2500 * ramp(t_new)

    # Predictor
    ũ = u .+ Δt .* v .+ (Δt^2 * (0.5 - β_hht)) .* a
    ṽ = v .+ (Δt * (1 - γ_hht)) .* a

    u_new = copy(ũ)
    apply!(u_new, ch)

    converged = false; iters = 0

    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat)
        assemble_pressure_all!(K_plv, F_plv, dh, scv, u_new)

        a_new = (u_new .- ũ) ./ (β_hht * Δt^2)
        v_new = ṽ .+ (Δt * γ_hht) .* a_new

        # HHT residual: M ä + (1−α)[C v + r_int − p F_p] + α g_old = 0
        R .= M * a_new .+ (1 - α_hht) .* (α_damp .* (M * v_new) .+ r_int .- p_new .* F_plv) .+ α_hht .* g_old
        apply_zero!(R, ch)

        norm(@views R[free]) < tol && (converged = true; break)

        K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* (K_int.nzval .- p_new .* K_plv.nzval)
        rhs = .-R
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        u_new .+= δu
        apply!(u_new, ch)
    end

    !converged && @warn "Step $step (t=$(round(t_new, digits=3)) s): no convergence in $max_iter iters"

    a .= (u_new .- ũ) ./ (β_hht * Δt^2)
    v .= ṽ .+ (Δt * γ_hht) .* a
    # store g_old = C·v + r_int(u) − p·F_p(u) for the α-offset in next step
    g_old .= α_damp .* (M * v) .+ r_int .- p_new .* F_plv
    p  = p_new
    u .= u_new

    vtk_step[] += 1
    VTKGridFile("pillow_dynamic-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u)
        write_projection(vtk, proj, project(proj, max_principal_stress_qp(dh, scv, u, mat), qr), "sigma_max")
        pvd[t_new] = vtk
    end

    step % 10 == 0 && @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d\n", step, t_new, ramp(t_new), p_new, iters)
end
close(pvd)
