using FerriteShells, LinearAlgebra, Printf, WriteVTK

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
    @show L
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

# Rectangular approximation of the miniLIMO geometry without rounded edges.
# SRF_1: outer endocardium (Plv only)
# SRF_2: inner endocardium at actuator footprint (Plv − Pact)
# SRF_3: actuator exterior shell (Pact only), double-layer with SRF_2.
#
# Each pouch (p = 1..Np) is a sealed inflatable cavity spanning nx_per_pouch coarse
# x-cells × ny_act coarse y-cells.  Only its outer perimeter (left edge, right edge,
# top row, bottom row of the pouch) is stitched to endo nodes; all interior nodes —
# including the column boundaries between coarse cells within the same pouch — are
# independent duplicate nodes so the pouch can inflate freely.  Adjacent pouches share
# their common boundary column through endo nodes (the dividing seam is attached).
function make_minilimo_grid(;
    nx_left=3, nx_act=10, nx_right=3,
    ny_bot=1, ny_act=14, ny_top=2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=1)

    @assert nx_act % Np == 0 "nx_act ($nx_act) must be divisible by Np ($Np)"
    nx_per_pouch = nx_act ÷ Np   # coarse x-cells per pouch

    Lx = W / 2
    xs = vcat(range(-Lx,   -x_act, nx_left + 1),
              range(-x_act,  x_act, nx_act  + 1)[2:end],
              range( x_act,    Lx,  nx_right + 1)[2:end])
    ys = vcat(range(0.0,  y_lo, ny_bot + 1),
              range(y_lo, y_hi, ny_act + 1)[2:end],
              range(y_hi,   H,  ny_top + 1)[2:end])
    nx = length(xs) - 1;  ny = length(ys) - 1

    ins(v) = (w = similar(v, 2length(v)-1); w[1:2:end]=v; w[2:2:end]=(v[1:end-1].+v[2:end])./2; w)
    xs_f = ins(xs);  ys_f = ins(ys)

    endo_node(px, py) = py * (2nx + 1) + px + 1
    n_endo = (2nx + 1) * (2ny + 1)
    endo_coords = [Vec{3}((xs_f[px+1], ys_f[py+1], 0.0)) for py in 0:2ny for px in 0:2nx]

    py_lo = 2ny_bot;  py_hi = 2(ny_bot + ny_act)

    # One act_nodes dict per pouch.  Perimeter = pouch outer x-edges + top/bottom → endo nodes.
    # Everything else (including inter-cell column boundaries within the pouch) → new nodes.
    act_nodes = [Dict{Tuple{Int,Int},Int}() for _ in 1:Np]
    act_coords = Vec{3,Float64}[]
    for p in 1:Np
        px_lo_p = 2(nx_left + (p-1)*nx_per_pouch)
        px_hi_p = 2(nx_left + p*nx_per_pouch)
        for py in py_lo:py_hi, px in px_lo_p:px_hi_p
            if px == px_lo_p || px == px_hi_p || py == py_lo || py == py_hi
                act_nodes[p][(px, py)] = endo_node(px, py)
            else
                push!(act_coords, Vec{3}((xs_f[px+1], ys_f[py+1], 0.0)))
                act_nodes[p][(px, py)] = n_endo + length(act_coords)
            end
        end
    end

    function q9_endo(px, py)
        QuadraticQuadrilateral((
            endo_node(px,   py),   endo_node(px+2, py),
            endo_node(px+2, py+2), endo_node(px,   py+2),
            endo_node(px+1, py),   endo_node(px+2, py+1),
            endo_node(px+1, py+2), endo_node(px,   py+1),
            endo_node(px+1, py+1)))
    end
    function q9_act(p, px, py)
        QuadraticQuadrilateral((
            act_nodes[p][(px,   py)],   act_nodes[p][(px+2, py)],
            act_nodes[p][(px+2, py+2)], act_nodes[p][(px,   py+2)],
            act_nodes[p][(px+1, py)],   act_nodes[p][(px+2, py+1)],
            act_nodes[p][(px+1, py+2)], act_nodes[p][(px,   py+1)],
            act_nodes[p][(px+1, py+1)]))
    end

    srf1   = Int[]
    srf2_k = [Int[] for _ in 1:Np]
    endo_cells = QuadraticQuadrilateral[]
    for iy in 0:ny-1, ix in 0:nx-1
        push!(endo_cells, q9_endo(2ix, 2iy))
        cid    = length(endo_cells)
        ix_rel = ix - nx_left
        if iy >= ny_bot && iy < ny_bot + ny_act && ix_rel >= 0 && ix_rel < nx_act
            push!(srf2_k[ix_rel ÷ nx_per_pouch + 1], cid)
        else
            push!(srf1, cid)
        end
    end
    n_ec = length(endo_cells)

    srf3_k = [Int[] for _ in 1:Np]
    act_cells = QuadraticQuadrilateral[]
    for p in 1:Np
        for k in 1:nx_per_pouch
            ix = nx_left + (p-1)*nx_per_pouch + k - 1
            px = 2ix
            for iy in ny_bot:ny_bot+ny_act-1
                push!(act_cells, q9_act(p, px, 2iy))
                push!(srf3_k[p], n_ec + length(act_cells))
            end
        end
    end

    grid = Grid(vcat(endo_cells, act_cells), Node.(vcat(endo_coords, act_coords)))
    addcellset!(grid, "SRF_1", Set(srf1))
    srf2_all = Int[];  srf3_all = Int[]
    for k in 1:Np
        addcellset!(grid, "SRF_2_$k", Set(srf2_k[k]))
        addcellset!(grid, "SRF_3_$k", Set(srf3_k[k]))
        append!(srf2_all, srf2_k[k]);  append!(srf3_all, srf3_k[k])
    end
    addcellset!(grid, "SRF_2", Set(srf2_all))
    addcellset!(grid, "SRF_3", Set(srf3_all))
    addnodeset!(grid, "edge", x -> x[2] ≈ 0.0)
    addfacetset!(grid, "sym", x -> x[2] ≈ H || abs(x[1]) ≈ Lx)
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

function assemble_pressure_region!(K_plv, F_plv, scv, u_vec, dh, cellset; Pᵢ=1)
    n_e = ndofs_per_cell(dh)
    ke_p = zeros(n_e, n_e)
    re_p = zeros(n_e)
    asm_p = start_assemble(K_plv)
    for cell in CellIterator(dh, cellset)
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd = shelldofs(cell)
        u_e = u_vec[sd]
        assemble_pressure!(re_p, scv, u_e, Pᵢ) # unit pressure
        assemble_pressure_tangent!(ke_p, scv, u_e, Pᵢ)
        assemble!(asm_p, sd, ke_p)
        F_plv[sd] .+= re_p
    end
end

# material model
mat = LinearElastic(0.35e6, 0.3,  0.0018)

# make the mesh
grid = make_minilimo_grid(;
    nx_left=3, nx_act=10, nx_right=3,
    ny_bot=1, ny_act=14, ny_top=2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=1
)


# interpolation scape
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

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
n_steps = 50
tol     = 1e-6
max_iter = 20

N = ndofs(dh)
K_int  = allocate_matrix(dh)
K_plv  = allocate_matrix(dh)
K_pact = allocate_matrix(dh)
K_plvpact = allocate_matrix(dh)
K_eff  = allocate_matrix(dh)   # preallocated; values updated in-place each Newton step
r_int  = zeros(N)
F_plv  = zeros(N)
F_pact = zeros(N)
F_plvpact = zeros(N)
v      = zeros(N)
v1     = zeros(N)
v2     = zeros(N)
u      = zeros(N)
Δu     = zeros(N)
un     = zeros(N)

pvd = paraview_collection("minilimo-inflation")
vtk_step = Ref(0)

# initialize the lu-decomposition
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)
free   = ch.free_dofs

tol_nl = 1e-6
n_pre  = 30          # NR steps
println("  step |    λ    | iters")
for step in 1:n_pre
    λ = step / n_pre
    Ferrite.update!(ch, λ)
    converged_pre = false; n_iter_pre = 0
    for iter in 1:max_iter
        fill!(F_plv, 0.0)
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        # external loading — must match coupling loop: SRF_1 ∪ SRF_2 is endocardium
        Plv = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")
        assemble_pressure_region!(K_plv, F_plv, scv, u, dh, Plv)
        K_eff.nzval .= K_int.nzval .- λ * p_max .* K_plv.nzval
        rhs1 = λ * p_max .* F_plv .- r_int
        apply_zero!(K_eff, rhs1, ch)
        norm(rhs1[free]) < tol_nl && (converged_pre = true; n_iter_pre = iter - 1; break)
        n_iter_pre = iter
        lu!(F_lu, K_eff); ldiv!(v1, F_lu, rhs1)
        u .+= v1; apply!(u, ch)
    end
    !converged_pre && (@warn "NR warm-up step $step did not converge"; break)
    VTKGridFile("minilimo-inflation-$(vtk_step[])", dh) do vtk
        vtk_step[] += 1
        write_solution(vtk, dh, u)
        Ferrite.write_constraints(vtk, ch)
        for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
        pvd[vtk_step[]] = vtk
    end
    un .= u
    @printf("  %4d |   %.4f | %d   | %4f\n", step, λ, n_iter_pre, λ * p_max)
end

# the three different surface where different pressures are assembled
# SRF_1: endocardium, Plv only (outward)
# SRF_2: endocardium + actuator, Plv (outward) and Pact (inward, opposing Plv)
# SRF_3: actuator exterior, Pact only (outward)
Plv_srf     = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")  # Plv acts here
Pact_srf    = getcellset(grid, "SRF_3")                                # +Pact
PlvPact_srf = getcellset(grid, "SRF_2")                                # −Pact (opposes Plv)

# what's the volume in this configuration
vol = -2compute_volume(dh, scv, un; cellset=Plv_srf) * m3_to_ml
println("Initial volume of the device: ", round(vol; digits=4), " ml")

import OrdinaryDiffEq as ODE
using Plots

# open-loop windkessel
function Windkessel!(du,u,p,t)
    # unpack
    (Vlv,Pa,Pv,Plv) = u
    (Ra,Ca,Rv,Cv,Rp)  = p

    # flow at the two vales
    Qmv = Pv ≥ Plv ? (Pv - Plv)/Rv : (Plv - Pv)/1e10
    Qao = Plv ≥ Pa ? (Plv - Pa)/Ra : (Pa - Plv)/1e10

    # rates
    du[1] = Qmv - Qao                 # dVlv/dt=Qmv-Qao
    du[2] = Qao/Ca + (Pv-Pa)/(Rp*Ca)  # dPa/dt
    du[3] = (Pa-Pv)/(Rp*Cv) - Qmv/Cv  # dPv/dt
    du[4] = 0.0                       # un-used u[4] hold the ventricular pressure
end

# actuation waveform (normalized to [0,1])
ϕᵢ(t;tC=0.10,tR=0.25,TC=0.15,TR=0.45) = 0.0<=(t-tC)%1<=TC ? 0.5*(1-cos(π*((t-tC)%1)/TC)) : (0.0<=(t-tR)%1<=TR ? 0.5*(1+cos(π*((t-tR)%1)/TR)) : 0)

# Kasra's parameters
Ra = 8.0e6*Pa2mmHg/m3_to_ml     # Pa.s/m³ -> mmHg.s/ml
Rp = 1.0e8*Pa2mmHg/m3_to_ml     # Pa.s/m³
Rv = 5.0e5*Pa2mmHg/m3_to_ml     # Pa.s/m³
Ca = 8.0e-9*m3_to_ml/Pa2mmHg    # m³/Pa
Cv = 5.0e-8*m3_to_ml/Pa2mmHg    # m³/Pa not used in openloop
Pv = p_max * Pa2mmHg

# setup
u₀ = [vol, 60, Pv, Pv]              # initial conditions
tspan = (0.0, 4.0)
params = (Ra,Ca,Rv,Cv,Rp)

# generate a problem to solve
prob = ODE.ODEProblem(Windkessel!, u₀, tspan, params)

# full control over iterations by making an iterator
integrator = ODE.init(prob, ODE.Tsit5(), reltol=1e-6,
                      abstol=1e-9, save_everystep=false)

# Reset ODE, I don;t think it's really necessary
ODE.reinit!(integrator, [vol, 60, Pv])

# coupling tolerances
tol      = 1e-4
max_iter = 20
dt_cpl   = 0.005

# storages
vols = Float64[]
pres = Float64[]
pact = Float64[]
paos = Float64[]
pvns = Float64[]
vtarget = []

# new FE arrays
dVdu = zeros(N)

# start with the initial condition from the morphing step
@time let u = copy(un), p = p_max, k₀ = length(pvd.timeSteps)
    println("3D-0D Lie–Trotter coupling (dt_cpl=$(dt_cpl) s)")
    println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters")
    step = 0
    while integrator.t < tspan[2] - dt_cpl / 2
        step += 1

        # advance Windkessel by dt_cpl; Plv = integrator.u[3] is held fixed.
        ODE.step!(integrator, dt_cpl, true)

        # full-LV volume (ml)
        V_target = 0.5 * integrator.u[1] / m3_to_ml # in m³
        push!(vtarget, integrator.u[1])

        # pressure at this step, meaning at t [mmHg], converted to Pa for 3D model
        Pact_mmHg = 100 * ϕᵢ(integrator.t;tC=0.1,tR=0.4,TC=0.3,TR=0.3) # in mmHg
        Pact = Pact_mmHg / Pa2mmHg # Pa

        # Schur Complement Newton-Raphson solve for the volume
        converged = false; n_iter = 0; V₃D = 0.0; S₀ = NaN
        for iter in 1:max_iter
            # assembly
            assemble_all!(K_int, r_int, dh, scv, u, mat)
            fill!(F_plv, 0.0); fill!(F_pact, 0.0); fill!(F_plvpact, 0.0) # reset here
            assemble_pressure_region!(K_plv, F_plv, scv, u, dh, Plv_srf)
            assemble_pressure_region!(K_pact, F_pact, scv, u, dh, Pact_srf)
            assemble_pressure_region!(K_plvpact, F_plvpact, scv, u, dh, PlvPact_srf)
            # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
            V₃D = -compute_volume(dh, scv, u; cellset=Plv_srf) # in m³
            volume_gradient!(dVdu, dh, scv, u; cellset=Plv_srf)
            dVdu[ch.prescribed_dofs] .= 0.0   # zero BC DOFs in gradient
            # Lagrange term of the coupled problem
            r_V  = V₃D - V_target
            # F_ext = p*F_plv + Pact*F_pact - Pact*F_plvpact
            # K_eff = K_int - ∂F_ext/∂u
            K_eff.nzval .= K_int.nzval .- p .* K_plv.nzval .- Pact .* K_pact.nzval .+ Pact .* K_plvpact.nzval
            rhs1 = p .* F_plv .+ Pact .* F_pact .- Pact .* F_plvpact .- r_int
            apply_zero!(K_eff, rhs1, ch)
            if norm(rhs1) < tol && abs(r_V) < tol * max(1.0, abs(V_target)) && iter!=1
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter
            # linear solve
            lu!(F_lu, K_eff)        # factorize
            ldiv!(v1, F_lu, rhs1)   # equilibrium correction
            ldiv!(v2, F_lu, F_plv)
            # Schur complement (dVdu = ∂(compute_volume)/∂u = −∂V₃D/∂u):
            S  = -dot(dVdu, v2)    # > 0: dVdu[u_z]<0, v2[u_z]>0
            δp = (-r_V + dot(dVdu, v1)) / S
            # correction
            u .+= v1 .+ δp .* v2
            p  += δp
            apply!(u, ch)
        end

        if mod(step, 1) == 0
            VTKGridFile("minilimo-inflation-$(vtk_step[])", dh) do vtk
                vtk_step[] += 1
                write_solution(vtk, dh, u)
                Ferrite.write_constraints(vtk, ch)
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                # per-node residual fields for debugging
                rhs_dbg = p .* F_plv .+ Pact .* F_pact .- Pact .* F_plvpact .- r_int
                rhs_dbg[ch.prescribed_dofs] .= 0.0
                u_range = dof_range(dh, :u); θ_range = dof_range(dh, :θ)
                n_nc    = length(grid.cells[1].nodes)
                res_u   = zeros(3, getnnodes(grid))
                res_θ   = zeros(2, getnnodes(grid))
                cnt     = zeros(Int, getnnodes(grid))
                for cell in CellIterator(dh)
                    dofs = celldofs(cell)
                    nids = grid.cells[Ferrite.cellid(cell)].nodes
                    for k in 1:n_nc
                        nid = nids[k]
                        res_u[:, nid] .+= rhs_dbg[dofs[u_range[3k-2:3k]]]
                        res_θ[:, nid] .+= rhs_dbg[dofs[θ_range[2k-1:2k]]]
                        cnt[nid] += 1
                    end
                end
                res_u ./= reshape(max.(cnt, 1), 1, :)
                res_θ ./= reshape(max.(cnt, 1), 1, :)
                write_node_data(vtk, res_u, "residual_u")
                write_node_data(vtk, res_θ, "residual_theta")
                pvd[k₀+integrator.t] = vtk
            end
            @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n", integrator.t, p * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter)
        end

        !converged && (@warn "step $step (t=$(integrator.t)) did not converge"; break)

        # feed new LV pressure back into ODE state.
        integrator.u[4] = p * Pa2mmHg # back in mmHg for the ODE
        ODE.u_modified!(integrator, true)

        push!(vols, 2V₃D * m3_to_ml)   # full volume [ml]
        push!(pres, p * Pa2mmHg)       # pressure [mmHg]
        push!(pact, Pact_mmHg)
        push!(paos, integrator.u[2])
        push!(pvns, integrator.u[3])
    end
end
vtk_save(pvd);

times = collect(0:dt_cpl:integrator.t)[1:length(pres)]
p1=plot(times, [vols, pres, pact, paos, pvns], xlabel="Time [s]",
        label=["Vlv" "Plv" "Pact" "Pao" "Pv"], lw=2, legend=:right)
p2=plot(vols, pres, label=:none, xlim=extrema(vols).+(-10,10), ylims=(0, 100),
        xlabel="Volume [ml]", ylabel="Pressure [mmHg]", lw=2,
        linez=times./maximum(times))
plot(p1, p2)
savefig("3D0D_limo_ferriteshells_N1.png")