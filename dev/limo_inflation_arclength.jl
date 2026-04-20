using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# colors the surface in the mesh by their ID from the *.inp file
function color(vtk, grid, cellset)
    z = zeros(Ferrite.getncells(grid))
    z[collect(Ferrite.getcellset(grid, cellset))] .= 1.0
    write_cell_data(vtk, z, cellset)
end

using QuadGK
function bisect(f, θ_lo, θ_hi; tolerance=1e-8)
    # bisection
    θ_mid = (θ_lo + θ_hi) / 2 # initial guess
    while θ_hi - θ_lo > tolerance
        θ_mid = (θ_lo + θ_hi) / 2
        f(θ_mid) * f(θ_lo) < 0 ? (θ_hi = θ_mid) : (θ_lo = θ_mid)
    end
    return θ_mid
end
function find_points(x, y, A, B, L)
    N = length(x)
    x_new = similar(x)
    y_new = similar(y)

    x_min = minimum(x)
    for i in (1, N)
        θ = (x[i] - x_min) * π / L
        x_new[i] = -A * cos(θ)
        y_new[i] = -B * sin(θ)
    end
    lengths = @views sqrt.((x[2:end] .- x[1:end-1]) .^ 2 .+ (y[2:end] .- y[1:end-1]) .^ 2)
    θ0 = 0.0
    for i in 1:N-2
        x0, y0, d = x_new[N-i+1], y_new[N-i+1], lengths[N-i]
        θ0 = bisect(θ0, π) do θ
            sqrt((A * cos(θ) - x0)^2 + (B * sin(θ) - y0)^2) - d
        end
        x_new[N-i] = A * cos(θ0)
        y_new[N-i] = B * sin(θ0)
    end
    x_new, y_new
end
function map_initial(x, y, Ar)
    L = maximum(x) - minimum(x)
    # find the minor/major axis that result in this length
    ds(θ, a) = sqrt(a^2 * sin(θ)^2 + (a / Ar)^2 * cos(θ)^2)
    function find_a(a)
        quadgk(θ -> ds(θ, a), 0, π)[1] - L
    end
    a0 = bisect(find_a, 0.0, L)
    a = bisect(0.98 * a0, 1.08 * a0) do a
        xi, yi = find_points(x, y, a, a / Ar, L)
        @views sum(sqrt.((xi[2:end] .- xi[1:end-1]) .^ 2 .+ (yi[2:end] .- yi[1:end-1]) .^ 2)) - L
    end
    find_points(x, y, a, a / Ar, L)
end

function make_quarter_pillow_grid(n; primitive=Quadrilateral)
    corners = [Vec{2}((-0.05058799, 0.000)), Vec{2}(( 0.05058799, 0.000)),
               Vec{2}(( 0.05058799, 0.109)), Vec{2}((-0.05058799, 0.109))]
    grid = shell_grid(generate_grid(primitive, (n, n), corners))
    return grid
end

# Assemble K_int, R_int, K_plv and F_plv (all for unit pressure p=1) in one cell loop.
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

function assemble_pressure_region!(K_plv, F_plv, scv, u_vec, dh, region_name)
    n_e = ndofs_per_cell(dh)
    ke_p = zeros(n_e, n_e)
    re_p = zeros(n_e)
    asm_p = start_assemble(K_plv)
    fill!(F_plv, 0.0)
    for cell in CellIterator(dh, getcellset(grid, region_name))
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd = shelldofs(cell)
        u_e = u_vec[sd]
        assemble_pressure!(re_p, scv, u_e, 1.0) # unit pressure
        assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm_p, sd, ke_p)
        F_plv[sd] .+= re_p
    end
end

# material model
mat = LinearElastic(0.35e6, 0.3,  0.002)

grid = make_quarter_pillow_grid(32; primitive=Quadrilateral)

# fname = "/home/marin/Workspace/HHH/code/miniLIMO/p6/geom_julia.inp"
# grid = get_ferrite_grid(fname)

addnodeset!(grid, "edge", x -> x[2] ≈ 0)
addfacetset!(grid, "sym", x -> ((x[2] ≈ 0.109) || (abs(x[1]) ≈ 0.05058799)))
addcellset!(grid, "Plv", x -> true) # all the grid

# interpolation scape
ip   = Lagrange{RefQuadrilateral, 1}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

function generate_boundary_function(grid, nodeset)
    top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(top_nodes)
    node_sorted = top_nodes[idx]
    Ar = 80.2 / 55.2 # from Nienke
    x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
    x_new, y_new = map_initial(x, y, Ar)
    Xs = vcat(x', y'); dXs = vcat(x_new' .- x', y_new') # we map to z-displacements which are zero
    return function prescribed_u(x, t)
        idx = findmin(dropdims(sum(abs2, Xs .- [x[1], x[2]], dims=1), dims=1))[2]
        return min(t,1).*dXs[:, idx] # linear ramp
    end
end

# generate the function for the boundary conditions
prescribed_u = generate_boundary_function(grid, "edge")

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2])) # what happens when we rotate
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# Displacement steps
Pa2mmHg = 0.00750062 # Pa/mmHg
m3_to_ml = 1.0e6          # m³ to ml
p_max   = 6.0 / Pa2mmHg  # Pfill = 6 mmHg

N = ndofs(dh)
K_int  = allocate_matrix(dh)
K_plv  = allocate_matrix(dh)
K_eff  = allocate_matrix(dh)   # preallocated; values updated in-place each Newton step
r_int  = zeros(N)
F_plv  = zeros(N)
v      = zeros(N)
v1     = zeros(N)
v2     = zeros(N)
u      = zeros(N)
Δu     = zeros(N)
f_ref  = zeros(N)
rhs    = zeros(N)

pvd = paraview_collection("minilimo")
vtk_step = Ref(0)

# initialize the lu-decomposition
free   = ch.free_dofs
assemble_all!(K_int, r_int, dh, scv, u, mat)
apply_zero!(K_int, r_int, ch)
F_lu = lu(K_int)

VTKGridFile("minilimo-0", dh) do vtk
    vtk_step[] += 1; write_solution(vtk, dh, u); pvd[0.0] = vtk
end

# arc-length
let Δs_init  = 0.2,    # initial arc-length step
    max_step = 200,     # maximum arc-length steps
    max_cor  = 20,      # maximum corrector iterations
    tol      = 1e-6,   # convergence tolerance on ‖g[FREE]‖
    verbose  = true

    # initial load factor
    λ  = 0.0            # load factor
    Δλ = Δs_init
    Δs = Δs_init

    # Track previous converged increment for sign control
    Δu_prev = copy(v1)
    Δλ_prev = 0.0

    verbose && @printf("%-6s  %-10s  %-8s\n", "step", "λ", "iters")

    for step in 1:max_step

        # predictor
        Ferrite.update!(ch, λ)
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        assemble_pressure_region!(K_plv, F_plv, scv, u, dh, "Plv")
        K_eff.nzval .= K_int.nzval .- λ * p_max .* K_plv.nzval
        f_ref .= p_max .* F_plv
        apply_zero!(K_eff, f_ref, ch)
        # load direction, fill LU tangent to load-displacement path
        lu!(F_lu, K_eff); ldiv!(v, F_lu, f_ref) # sensitivity du/dλ at current state
        apply_zero!(v, ch)

        # cylindrical arc-length
        vn = norm(v[free])
        vn < 1e-15 && error("Singular predictor at step $step")

        # Sign control: choose direction consistent with previous step
        δλ_pred = Δs / vn
        if step > 1
            dot(Δu_prev[free], v[free]) + Δλ_prev * δλ_pred < 0 && (δλ_pred = -δλ_pred)
        end

        # corrections
        Δu      .= δλ_pred .* v
        Δλ       = δλ_pred
        converged = false

        # corrector
        iters = 0
        for cor in 1:max_cor
            iters = cor
            Ferrite.update!(ch, λ + Δλ)
            u_trial = u .+ Δu; apply!(u_trial, ch);

            assemble_all!(K_int, r_int, dh, scv, u_trial, mat)
            assemble_pressure_region!(K_plv, F_plv, scv, u_trial, dh, "Plv")
            K_eff.nzval .= K_int.nzval .- (λ + Δλ) * p_max .* K_plv.nzval
            rhs .= (λ + Δλ) * p_max .* F_plv .- r_int # equilibrium residual
            apply_zero!(K_eff, rhs, ch)
            # check for convergence
            norm(rhs[free]) < tol && (converged = true; break)
            # solve
            lu!(F_lu, K_eff)
            ldiv!(v1, F_lu, rhs)   # displacement correction
            ldiv!(v2, F_lu, f_ref) # load sensitivity direction
            # Arc-length corrector equation:
            #   dot(Δu_free, Δu_free + v1_free + δλ*(v2_free)) = Δs²
            # Linearised (omitting ‖v1‖² term):
            #   δλ = -dot(Δu_free, v1_free) / dot(Δu_free, v2_free)
            denom = dot(Δu[free], v2[free])
            abs(denom) < 1e-15 && error("Singular arc-length system at step $step, cor $cor")
            δλ = -dot(Δu[free], v1[free]) / denom
            Δu[free] .+= v1[free] .+ δλ .* v2[free]
            Δλ        += δλ
        end

        if !converged
            @warn "Step $step did not converge in $max_cor correctors; halving Δs"
            Δs /= 2
            continue
        end

        # ── Accept step ────────────────────────────────────────────────────
        u  .+= Δu
        λ   += Δλ
        Ferrite.update!(ch, λ)
        apply!(u, ch)   # update constrained DOFs to current λ (BC ramp)
        Δu_prev .= Δu
        Δλ_prev  = Δλ
        # increase factor if sucessfull.
        Δs = min(Δs * 1.2, 0.5)

        verbose && @printf("%-6d  %-10.4f  %-8d\n", step, λ, iters)

        λ ≥ 1.0 && break # once we reach the max load factor, exit
    end
end
VTKGridFile("minilimo-1", dh) do vtk
    vtk_step[] += 1; write_solution(vtk, dh, u); pvd[1.0] = vtk
end; close(pvd);
