using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK
using QuadGK

function bisect(f, θ_lo, θ_hi; tolerance=1e-8)
    θ_mid = (θ_lo + θ_hi) / 2
    while θ_hi - θ_lo > tolerance
        θ_mid = (θ_lo + θ_hi) / 2
        f(θ_mid) * f(θ_lo) < 0 ? (θ_hi = θ_mid) : (θ_lo = θ_mid)
    end
    return θ_mid
end

function find_points(x, y, A, B, L)
    N = length(x)
    x_new = similar(x); y_new = similar(y)
    x_min = minimum(x)
    for i in (1, N)
        θ = (x[i] - x_min) * π / L
        x_new[i] = -A * cos(θ); y_new[i] = -B * sin(θ)
    end
    lengths = @views sqrt.((x[2:end] .- x[1:end-1]).^2 .+ (y[2:end] .- y[1:end-1]).^2)
    θ0 = 0.0
    for i in 1:N-2
        x0, y0, d = x_new[N-i+1], y_new[N-i+1], lengths[N-i]
        θ0 = bisect(θ0, π) do θ
            sqrt((A*cos(θ)-x0)^2 + (B*sin(θ)-y0)^2) - d
        end
        x_new[N-i] = A*cos(θ0); y_new[N-i] = B*sin(θ0)
    end
    x_new, y_new
end

function map_initial(x, y, Ar)
    L = maximum(x) - minimum(x)
    ds(θ, a) = sqrt(a^2*sin(θ)^2 + (a/Ar)^2*cos(θ)^2)
    find_a(a) = quadgk(θ -> ds(θ, a), 0, π)[1] - L
    a0 = bisect(find_a, 0.0, L)
    a = bisect(0.98*a0, 1.08*a0) do a
        xi, yi = find_points(x, y, a, a/Ar, L)
        @views sum(sqrt.((xi[2:end].-xi[1:end-1]).^2 .+ (yi[2:end].-yi[1:end-1]).^2)) - L
    end
    find_points(x, y, a, a/Ar, L)
end

function make_quarter_pillow_grid(n; primitive=Quadrilateral)
    corners = [Vec{2}((-0.05058799, 0.000)), Vec{2}(( 0.05058799, 0.000)),
               Vec{2}(( 0.05058799, 0.109)), Vec{2}((-0.05058799, 0.109))]
    shell_grid(generate_grid(primitive, (n, n), corners))
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

mat  = LinearElastic(0.35e6, 0.3, 0.002)

# grid = make_quarter_pillow_grid(32)
fname = "/home/marin/Workspace/HHH/code/miniLIMO/p6/geom_julia.inp"
grid = get_ferrite_grid(fname)

addnodeset!(grid, "edge", x -> x[2] ≈ 0)
addfacetset!(grid, "sym",  x -> (x[2] ≈ 0.109) || (abs(x[1]) ≈ 0.05058799))

ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

function generate_boundary_function(grid, nodeset)
    top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(top_nodes)
    node_sorted = top_nodes[idx]
    Ar = 80.2 / 55.2
    x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
    x_new, y_new = map_initial(x, y, Ar)
    Xs  = vcat(x', y')
    dXs = vcat(x_new' .- x', y_new')
    (x, t) -> begin
        idx = findmin(dropdims(sum(abs2, Xs .- [x[1], x[2]], dims=1), dims=1))[2]
        min(t, 1) .* dXs[:, idx]
    end
end

prescribed_u = generate_boundary_function(grid, "edge")

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0,      [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0,      [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dof = ndofs(dh)
K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
r_int = zeros(N_dof)
f_ref = zeros(N_dof)
v     = zeros(N_dof)
v1    = zeros(N_dof)
v2    = zeros(N_dof)
u     = zeros(N_dof)
Δu    = zeros(N_dof)
rhs   = zeros(N_dof)

free = ch.free_dofs
# constrained DOF indices — used to zero f_ref without calling apply_zero! a second time
constrained_dofs = setdiff(1:N_dof, free)

# u_ref: the full prescribed displacement at λ=1 (constant throughout the run).
# f_ref = -(K_int · u_ref) gives the force on free DOFs from a unit BC increment.
# At free DOFs: (K_int · u_ref)[free] = K_fc · u_c_ref  →  f_ref[free] = -K_fc · u_c_ref.
u_ref = zeros(N_dof)
Ferrite.update!(ch, 1.0); apply!(u_ref, ch)
Ferrite.update!(ch, 0.0)

# Initial symbolic LU factorisation (pattern fixed by allocate_matrix).
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
let tmp = zeros(N_dof); apply_zero!(K_eff, tmp, ch); end
F_lu = lu(K_eff)

vtk_step = Ref(0)
pvd = paraview_collection("minilimo_morphing")
VTKGridFile("minilimo_morphing-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0] = vtk
end

let Δs_init = 0.01,
    max_step = 800,
    max_cor  = 20,
    tol      = 1e-6,
    verbose  = true

    λ  = 0.0
    Δs = Δs_init

    Δu_prev = zeros(N_dof)
    Δλ_prev = 0.0

    verbose && @printf("%-6s  %-8s  %-8s  %-8s\n", "step", "λ", "Δλ", "iters")

    for step in 1:max_step

        # Predictor — sensitivity direction v = -K_ff⁻¹ K_fc u_c_ref
        Ferrite.update!(ch, λ)
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        K_eff.nzval .= K_int.nzval
        mul!(f_ref, K_int, u_ref); f_ref .*= -1.0   # -(K · u_ref)
        apply_zero!(K_eff, f_ref, ch)               # BC rows/cols in K_eff; zero f_ref at constrained DOFs
        lu!(F_lu, K_eff); ldiv!(v, F_lu, f_ref)
        apply_zero!(v, ch)

        vn = norm(v[free])
        vn < 1e-15 && error("Singular predictor at step $step")

        δλ_pred = Δs / vn
        if step > 1
            dot(Δu_prev[free], v[free]) + Δλ_prev * δλ_pred < 0 && (δλ_pred = -δλ_pred)
        end

        Δu .= δλ_pred .* v
        Δλ  = δλ_pred
        converged = false

        iters = 0
        for cor in 1:max_cor
            iters = cor
            Ferrite.update!(ch, λ + Δλ)
            u_trial = u .+ Δu; apply!(u_trial, ch)

            assemble_all!(K_int, r_int, dh, scv, u_trial, mat)
            K_eff.nzval .= K_int.nzval
            rhs .= .-r_int                           # residual g = r_int (no external load)
            apply_zero!(K_eff, rhs, ch)
            norm(rhs[free]) < tol && (converged = true; break)

            lu!(F_lu, K_eff)
            ldiv!(v1, F_lu, rhs)                     # equilibrium correction

            # f_ref from current K_int (K_eff already in BC form — just zero constrained entries)
            mul!(f_ref, K_int, u_ref); f_ref .*= -1.0
            f_ref[constrained_dofs] .= 0.0
            ldiv!(v2, F_lu, f_ref)                   # load-sensitivity direction

            denom = dot(Δu[free], v2[free])
            abs(denom) < 1e-15 && error("Singular arc-length at step $step, cor $cor")
            δλ = -dot(Δu[free], v1[free]) / denom
            Δu[free] .+= v1[free] .+ δλ .* v2[free]
            Δλ += δλ
        end

        if !converged
            @warn "Step $step: no convergence in $max_cor iters; halving Δs"
            Δs /= 2
            continue
        end

        u .+= Δu
        λ  += Δλ
        Ferrite.update!(ch, λ); apply!(u, ch)
        Δu_prev .= Δu; Δλ_prev = Δλ
        Δs = min(Δs * 1.2, 0.1)

        vtk_step[] += 1
        VTKGridFile("minilimo_morphing-$(vtk_step[])", dh) do vtk
            write_solution(vtk, dh, u); pvd[vtk_step[]] = vtk
        end

        verbose && @printf("%-6d  %-8.4f  %-8.4g  %-8d\n", step, λ, Δλ, iters)
        λ ≥ 1.0 && break
    end
end
close(pvd)
