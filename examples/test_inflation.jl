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

function make_quarter_pillow_grid(n; L=1.0)
    corners = [Vec{2}((-L/2, -L/2)), Vec{2}((L/2, -L/2)), Vec{2}((L/2, L/2)), Vec{2}((-L/2, L/2))]
    grid = shell_grid(generate_grid(Quadrilateral, (n, n), corners))

    addnodeset!(grid, "x_low", x -> isapprox(x[1], -L/2, atol=1e-10))

    addcellset!(grid, "actuation", x -> (abs(x[1]) < 0.4L && abs(x[2]) < 0.4L) )

    addfacetset!(grid, "x_high", x -> isapprox(x[1],  L/2, atol=1e-10))
    addfacetset!(grid, "y_low", x -> isapprox(x[2], -L/2, atol=1e-10))
    addfacetset!(grid, "y_high", x -> isapprox(x[2],  L/2, atol=1e-10))
    addnodeset!(grid, "center", x -> norm(x) < 1e-10)
    addnodeset!(grid, "origin", x -> isapprox(x[1], -L/2, atol=1e-10) && isapprox(x[2], L/2, atol=1e-10))
    return grid
end

# Assemble K_int, R_int, K_pres and F_p (all for unit pressure p=1) in one cell loop.
function assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    ke_p = zeros(n_e, n_e); re_p = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    asm_p = start_assemble(K_pres)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        assemble_pressure!(re_p, scv, u_e, 1.0)
        membrane_tangent_RM!(ke_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm_i, sd, ke_i, re_i)
        assemble!(asm_p, sd, ke_p)
        F_p[sd] .+= re_p
    end
end

# device dimensions
L   = 0.1 # Length in meters
n   = 32
mat = LinearElastic(0.35e6, 0.3,  0.002)

grid = make_quarter_pillow_grid(n; L)
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
    x, y = getindex.(node_sorted, 2), getindex.(node_sorted, 1)
    x_new, y_new = map_initial(x, y, Ar)
    Xs = vcat(x', y'); dXs = vcat(x_new' .- x', y_new') # we map to z-displacements which are zero
    return function prescribed_u(x, t)
        # carefull with the Xs, these are (y,x) points that map to (y,z) displacements
        idx = findmin(dropdims(sum(abs2, Xs .- [x[2], x[1]], dims=1), dims=1))[2]
        return min(t,1).*dXs[:, idx] # linear ramp
    end
end

# generate the function for the boundary conditions
prescribed_u = generate_boundary_function(grid, "x_low")

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "x_low"), (x,t) -> prescribed_u(x, t), [2,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "x_low"), x -> 0.0, [1]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "x_low"), x -> zeros(2), [1,2])) # what happens when we rotate
add!(ch, Dirichlet(:u, getfacetset(grid, "y_low"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "y_low"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "x_high"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "x_high"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "y_high"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "y_high"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# Global DOF index for u_z at the center node (free DOF — used as displacement control).
center_node = only(getnodeset(grid, "center"))
w_center_dof = let
    dof = 0
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            if gid == center_node
                dof = celldofs(cell)[3I]; break
            end
        end
        dof > 0 && break
    end
    dof
end
@assert w_center_dof > 0 "center u_z DOF not found"

# Displacement steps
Pa2mmHg = 0.00750062 # Pa/mmHg
m3_to_ml = 1.0e6          # m³ to ml
p_max   = 6.0 / Pa2mmHg  # Pfill = 6 mmHg
w_max   = 0.02         # upper bound for w_center (membrane theory at p_max ≈ 0.63 m)
n_steps = 50
Δw      = 2w_max / n_steps   # = 0.005 m = 5·t per step
tol     = 1e-6
max_iter = 20

N = ndofs(dh)
K_int  = allocate_matrix(dh)
K_pres = allocate_matrix(dh)
K_eff  = allocate_matrix(dh)   # preallocated; values updated in-place each Newton step
r_int  = zeros(N)
F_p    = zeros(N)
v1     = zeros(N)
v2     = zeros(N)
u_final= zeros(N)

pvd = paraview_collection("minilimo")
vtk_step = Ref(0)
p_final  = Ref(0.0)

# Initialise symbolic LU factorisation from the linearised system at u=0, p=0
assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u_final, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

# TODO the one I am using
# println("miniLIMO RM (Q4, n=$n)")
# println("  step |    λ    | iters")
# let u = zeros(N), p = 0.0
#     VTKGridFile("minilimo-0", dh) do vtk
#         vtk_step[] += 1
#         write_solution(vtk, dh, u)
#         pvd[0.0] = vtk
#     end
#     δp = p_max / n_steps
#     for step in 1:n_steps
#         λ = step/n_steps
#         Ferrite.update!(ch, λ)
#         converged = false; n_iter = 0
#         for iter in 1:max_iter
#             assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
#             # Update K_eff values in-place (same sparsity pattern as K_int and K_pres).
#             K_eff.nzval .= K_int.nzval .- p .* K_pres.nzval
#             rhs1 = p .* F_p .- r_int           # −R(u,p): negative equilibrium residual
#             apply_zero!(K_eff, rhs1, ch)        # zero BC rows/cols of K_eff, BC entries of rhs1
#             if norm(rhs1) < tol
#                 converged = true; n_iter = iter - 1; break
#             end
#             n_iter = iter
#             lu!(F_lu, K_eff)        # numeric refactorisation only; reuses symbolic analysis
#             ldiv!(v1, F_lu, rhs1)   # equilibrium correction
#             u .+= v1
#             apply!(u, ch)          # reset BC DOFs to zero
#         end
#         p  += δp # increment pressure loading

#         if !converged
#             @warn "step $step did not converge after $max_iter iters (p=$p)"
#             break
#         end

#         VTKGridFile("minilimo-$(vtk_step[])", dh) do vtk
#             vtk_step[] += 1
#             write_solution(vtk, dh, u)
#             Ferrite.write_constraints(vtk, ch)
#             color(vtk, grid, "actuation")
#             pvd[float(step)] = vtk
#         end
#         p_final[] = p
#         @printf("  %4d |   %.3f | %d\n", step, λ, n_iter)
#     end
#     u_final .= u
# end

# # first step, go into initial configuration by applying BC on valve support and P_fill in ventricle with
# # load displacement-controlled Newton-Raphson.
# let u = zeros(N), p = 0.0
#     VTKGridFile("minilimo-0", dh) do vtk
#         vtk_step[] += 1
#         write_solution(vtk, dh, u); pvd[0.0] = vtk
#     end

#     for step in 1:n_steps
#         w_target = step * Δw
#         Ferrite.update!(ch, step/20) # zero to one ramp quicker than the pressure, maxes at t=1 in function
#         converged = false; n_iter = 0
#         for iter in 1:max_iter
#             assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
#             # Update K_eff values in-place (same sparsity pattern as K_int and K_pres).
#             K_eff.nzval .= K_int.nzval .- p .* K_pres.nzval
#             rhs1 = p .* F_p .- r_int           # −R(u,p): negative equilibrium residual
#             apply_zero!(K_eff, rhs1, ch)        # zero BC rows/cols of K_eff, BC entries of rhs1
#             if norm(rhs1) < tol && abs(u[w_center_dof] - w_target) < tol
#                 converged = true; n_iter = iter - 1; break
#             end
#             n_iter = iter
#             lu!(F_lu, K_eff)        # numeric refactorisation only; reuses symbolic analysis
#             ldiv!(v1, F_lu, rhs1)   # equilibrium correction
#             ldiv!(v2, F_lu, F_p)    # load-direction vector
#             δp = (w_target - u[w_center_dof] - v1[w_center_dof]) / v2[w_center_dof]
#             u .+= v1 .+ δp .* v2
#             p  += δp
#             apply!(u, ch)                       # reset BC DOFs to zero
#         end

#         if !converged
#             @warn "step $step (w_target=$w_target) did not converge after $max_iter iters (p=$p)"
#             break
#         end

#         VTKGridFile("minilimo-$(vtk_step[])", dh) do vtk
#             vtk_step[] += 1
#             write_solution(vtk, dh, u)
#             Ferrite.write_constraints(vtk, ch)
#             color(vtk, grid, "actuation")
#             pvd[float(step)] = vtk
#         end
#         p_final[] = p
#         @printf("  %4d | %4f | %8.4e | %d\n", step, p, u[w_center_dof], n_iter)
#         p ≥ p_max && (@printf("  Reached p_max = %.4f at step %d (w = %.4f m).\n", p_max, step, u[w_center_dof]); break)
#     end
#     u_final .= u
# end

using JLD2
# jldsave("inflation.jld2"; u_final, p_final)
data = jldopen("inflation.jld2", "r")
u_final = data["u_final"]
p_final = data["p_final"]

Ferrite.update!(ch, 1.0)
assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u_final, mat)
K_eff.nzval .= K_int.nzval .- p_final[] .* K_pres.nzval
rhs1 = p_final[] .* F_p .- r_int
apply_zero!(K_eff, rhs1, ch)
norm(rhs1)

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
let u = copy(u_final), p = p_final[], k₀ = length(pvd.timeSteps)
    println("3D-0D Lie–Trotter coupling (RM Q9, n=$n, dt_cpl=$(dt_cpl) s)")
    println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters")

    step = 0
    while integrator.t < tspan[2] - dt_cpl / 2
        step += 1

        # advance Windkessel by dt_cpl; Plv = integrator.u[3] is held fixed.
        ODE.step!(integrator, dt_cpl, true)

        # full-LV volume (ml)
        V_target = 0.5 * integrator.u[1] / m3_to_ml # in m³
        push!(vtarget, V_target)

        # pressure at this step, meaning at t [mmHg], converted to Pa for 3D model
        Pact_mmHg = 80 * ϕᵢ(integrator.t;tC=0.1,tR=0.4,TC=0.3,TR=0.3) # in mmHg
        Pact = Pact_mmHg / Pa2mmHg # Pa
        @show V_target, p, Pact_mmHg

        # Schur Complement Newton-Raphson solve for the volume
        converged = false; n_iter = 0; V₃D = 0.0
        for iter in 1:max_iter
            assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)

            # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
            V₃D = -compute_volume(dh, scv, u) # in m³
            volume_gradient!(dVdu, dh, scv, u)
            dVdu[ch.prescribed_dofs] .= 0.0   # zero BC DOFs in gradient
            @show norm(dVdu)

            r_V  = V₃D - V_target
            K_eff.nzval .= K_int.nzval .- (p-Pact) .* K_pres.nzval
            rhs1 = (p-Pact) .* F_p .- r_int
            apply_zero!(K_eff, rhs1, ch)
            @show V₃D, p
            if norm(rhs1) < tol && abs(r_V) < tol * max(1.0, abs(V_target))
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter

            lu!(F_lu, K_eff)        # factorize
            ldiv!(v1, F_lu, rhs1)   # equilibrium correction
            ldiv!(v2, F_lu, F_p)    # load-direction vector

            # Schur complement (dVdu = ∂(compute_volume)/∂u = −∂V₃D/∂u):
            S  = -dot(dVdu, v2)                       # > 0: dVdu[u_z]<0, v2[u_z]>0
            δp = (-r_V + dot(dVdu, v1)) / S
            u .+= v1 .+ δp .* v2
            p  += δp
            apply!(u, ch)
        end

        !converged && (@warn "step $step (t=$(integrator.t)) did not converge"; break)

        # feed new LV pressure back into ODE state.
        integrator.u[3] = p * Pa2mmHg # back in mmHg for the ODE
        ODE.u_modified!(integrator, true)

        push!(vols, 2V₃D * m3_to_ml)   # full volume [ml]
        push!(pres, p * Pa2mmHg)            # pressure [mmHg]
        push!(pact, Pact_mmHg)

        if mod(step, 10) == 0
            VTKGridFile("minilimo-$(vtk_step[])", dh) do vtk
                vtk_step[] += 1
                write_solution(vtk, dh, u)
                pvd[k₀+integrator.t] = vtk
            end
            @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n",
                    integrator.t, p * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter)
        end
    end
end
vtk_save(pvd);

# # plot([vols, pres, pact, vtarget* 4 * m3_to_ml], xlabel="Volume [ml]", ylabel="Pressure [mmHg]",
#     #  label=["Vlv" "Plv" "Pact" "Vtarget"])