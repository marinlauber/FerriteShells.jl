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
addcellset!(grid, "Pact", x -> (abs(x[1]) < 0.020235196 && abs(x[2]) < 0.020235196) )

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

# using Plots
# nodeset = "edge"
# top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
# idx = sortperm(top_nodes)
# node_sorted = top_nodes[idx]
# Ar = 80.2 / 55.2 # from Nienke
# x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
# x_new, y_new = map_initial(x, y, Ar)
# Xs = vcat(x', y'); dXs = vcat(x_new' .- x', y_new')
# scatter(x, y); scatter!(x_new, y_new, aspect_ratio=:equal)

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
K_eff  = allocate_matrix(dh)   # preallocated; values updated in-place each Newton step
r_int  = zeros(N)
F_plv  = zeros(N)
F_pact = zeros(N)
v      = zeros(N)
v1     = zeros(N)
v2     = zeros(N)
u      = zeros(N)
Δu     = zeros(N)
un     = zeros(N)

pvd = paraview_collection("minilimo")
vtk_step = Ref(0)
p_final  = Ref(0.0)

# initialize the lu-decomposition
assemble_all!(K_int, r_int, dh, scv, u_final, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)
free   = ch.free_dofs

# tol_nl = 1e-6
# n_pre   = 30          # NR steps before arc-length
# println("Phase 1: NR warm-up (λ = 0 → $λ_pre in $n_pre steps)")
# println("  step |    λ    | iters")
# for step in 1:n_pre
#     λ = step / n_pre
#     Ferrite.update!(ch, λ)
#     converged_pre = false; n_iter_pre = 0
#     for iter in 1:max_iter
#         assemble_all!(K_int, r_int, dh, scv, u, mat)
#         assemble_pressure_region!(K_plv, F_plv, scv, u, dh, "Plv")
#         K_eff.nzval .= K_int.nzval .- λ * p_max .* K_plv.nzval
#         rhs1 = λ * p_max .* F_plv .- r_int
#         apply_zero!(K_eff, rhs1, ch)
#         norm(rhs1[free]) < tol_nl && (converged_pre = true; n_iter_pre = iter - 1; break)
#         n_iter_pre = iter
#         lu!(F_lu, K_eff); ldiv!(v1, F_lu, rhs1)
#         u .+= v1; apply!(u, ch)
#     end
#     !converged_pre && @warn "NR warm-up step $step did not converge"
#     un .= u
#     @printf("  %4d |   %.4f | %d   | %4f\n", step, λ, n_iter_pre, λ * p_max)
# end

using JLD2
# jldsave("inflation.jld2"; u_final=u, p_final=p_max)
data = jldopen("inflation.jld2", "r")
u_final = data["u_final"]
p_final = data["p_final"]

# what's the volume in this configuration
vol = -2compute_volume(dh, scv, u_final) * m3_to_ml

import OrdinaryDiffEq as ODE
using Plots

# open-loop windkessel
function Windkessel!(du,u,p,t)
    # unpack
    (Vlv,Pa,Plv) = u
    (Ra,Ca,Rv,Cv,Rp,Pv)  = p

    # flow at the two vales
    Qmv = Pv ≥ Plv ? (Pv - Plv)/Rv : (Plv - Pv)/1e10
    Qao = Plv ≥ Pa ? (Plv - Pa)/Ra : (Pa - Plv)/1e10

    # rates
    du[1] = Qmv - Qao             # dVlv/dt=Qmv-Qao
    du[2] = Qao/Ca - Pa/(Rp*Ca)   # dPa/dt=Qao/C-Pao/RC
    du[3] = 0.0                   # un-used u[3] hold the ventricular pressure
end

# actuation waveform (normalized to [0,1])
ϕᵢ(t;tC=0.10,tR=0.25,TC=0.15,TR=0.45) = 0.0<=(t-tC)%1<=TC ? 0.5*(1-cos(π*((t-tC)%1)/TC)) : (0.0<=(t-tR)%1<=TR ? 0.5*(1+cos(π*((t-tR)%1)/TR)) : 0)

# Kasra's parameters
Ra = 8.0e6*Pa2mmHg/m3_to_ml     # Pa.s/m³ -> mmHg.s/ml
Rp = 1.0e8*Pa2mmHg/m3_to_ml     # Pa.s/m³
Rv = 5.0e5*Pa2mmHg/m3_to_ml     # Pa.s/m³
Ca = 8.0e-9*m3_to_ml/Pa2mmHg    # m³/Pa
Cv = 5.0e-8*m3_to_ml/Pa2mmHg    # m³/Pa not used in openloop
Pv = p_final[] * Pa2mmHg

# setup
u₀ = [vol, 60, Pv]              # initial conditions
tspan = (0.0, 10.0)
params = (Ra,Ca,Rv,Cv,Rp,Pv)

# generate a problem to solve
prob = ODE.ODEProblem(Windkessel!, u₀, tspan, params)

# full control over iterations
integrator = ODE.init(prob, ODE.Tsit5(), reltol=1e-6,
                      abstol=1e-9, save_everystep=false)

# Reset ODE.
ODE.reinit!(integrator, [vol, 60, Pv])

# coupling tolerances
tol      = 1e-6
max_iter = 20
dt_cpl   = 0.01

# storages
vols = Float64[]
pres = Float64[]
pact = Float64[]
vtarget = []

# new FE arrays
dVdu = zeros(N)

# start with the initial condition from the morphing step
@time let u = copy(u_final), p = p_final[], k₀ = length(pvd.timeSteps)
    println("3D-0D Lie–Trotter coupling (RM Q4, dt_cpl=$(dt_cpl) s)")
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
        Pact_mmHg = 80 * ϕᵢ(integrator.t;tC=0.1,tR=0.4,TC=0.3,TR=0.3) # in mmHg
        Pact = Pact_mmHg / Pa2mmHg # Pa

        # Schur Complement Newton-Raphson solve for the volume
        converged = false; n_iter = 0; V₃D = 0.0
        for iter in 1:max_iter
            # assembly
            assemble_all!(K_int, r_int, dh, scv, u, mat)
            assemble_pressure_region!(K_plv, F_plv, scv, u, dh, "Plv")
            assemble_pressure_region!(K_pact, F_pact, scv, u, dh, "Pact")
            # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
            V₃D = -compute_volume(dh, scv, u) # in m³
            volume_gradient!(dVdu, dh, scv, u)
            dVdu[ch.prescribed_dofs] .= 0.0   # zero BC DOFs in gradient
            # Lagrange term of the coupled problem
            r_V  = V₃D - V_target
            K_eff.nzval .= K_int.nzval .- p .* K_plv.nzval .+ Pact .* K_pact.nzval
            rhs1 = p .* F_plv .- Pact .* F_pact .- r_int
            apply_zero!(K_eff, rhs1, ch)
            if norm(rhs1) < tol && abs(r_V) < tol * max(1.0, abs(V_target))
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter
            # linear solve
            lu!(F_lu, K_eff)        # factorize
            ldiv!(v1, F_lu, rhs1)   # equilibrium correction
            ldiv!(v2, F_lu, F_plv)    # load-direction vector
            # Schur complement (dVdu = ∂(compute_volume)/∂u = −∂V₃D/∂u):
            S  = -dot(dVdu, v2)                       # > 0: dVdu[u_z]<0, v2[u_z]>0
            δp = (-r_V + dot(dVdu, v1)) / S
            u .+= v1 .+ δp .* v2
            p  += δp
            apply!(u, ch) # make sure BCs are correctly imposed
        end

        !converged && (@warn "step $step (t=$(integrator.t)) did not converge"; break)

        # feed new LV pressure back into ODE state.
        integrator.u[3] = p * Pa2mmHg # back in mmHg for the ODE
        ODE.u_modified!(integrator, true)

        push!(vols, 2V₃D * m3_to_ml)   # full volume [ml]
        push!(pres, p * Pa2mmHg)            # pressure [mmHg]
        push!(pact, Pact_mmHg)

        if mod(step, 5) == 0
            VTKGridFile("minilimo-$(vtk_step[])", dh) do vtk
                vtk_step[] += 1
                write_solution(vtk, dh, u)
                Ferrite.write_constraints(vtk, ch)
                color(vtk, grid, "Pact")
                color(vtk, grid, "Plv")
                pvd[k₀+integrator.t] = vtk
            end
            @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n", integrator.t, p * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter)
        end
    end
end
vtk_save(pvd);

# times = collect(0:dt_cpl:integrator.t)
# p1=plot(times, [vols, pres, pact], xlabel="Time [s]",
#         label=["Vlv" "Plv" "Pact"], lw=2)
# p2=plot(vols, pres, label=:none, xlim=(200,300),ylims=(0, 100),
#         xlabel="Volume [ml]", ylabel="Pressure [mmHg]", lw=2,
#         linez=times./maximum(times))
# plot(p1, p2)
# savefig("3D0D_limo_ferriteshells.png")