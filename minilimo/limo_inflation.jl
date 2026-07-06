using FerriteShells, LinearAlgebra, Printf, WriteVTK
include(joinpath(@__DIR__, "util.jl"))

using QuadGK
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
# mat = LinearElastic(0.35e9, 0.3, 0.0002) # nylon-cpated TPU
mat = LinearElastic(20e6, 0.3, 0.001) # soft TPU
# grid = make_quarter_pillow_grid(32; primitive=Quadrilateral)

# fname = "/home/marin/Workspace/HHH/code/miniLIMO/p6/geom_julia_single.inp"
# grid = get_ferrite_grid(fname)
# addnodeset!(grid, "edge", x -> x[2] ≈ 0)
# addfacetset!(grid, "sym", x -> (x[2] ≈ 0.109) || (abs(x[1]) ≈ 0.05058799))
# let r_arc = 0.109 - 0.0951134 # add the arc nodes
#     addfacetset!(grid, "sym_arcs", x -> begin
#         d1 = sqrt((x[1] - 0.03670139)^2 + (x[2] - 0.0951134)^2)
#         d2 = sqrt((x[1] + 0.03670139)^2 + (x[2] - 0.0951134)^2)
#         abs(d1 - r_arc) < 1e-5 || abs(d2 - r_arc) < 1e-5
#     end)
#     union!(grid.facetsets["sym"], grid.facetsets["sym_arcs"])
# end

Np = 3
grid = make_minilimo_grid(;
    nx_left=3*3, nx_act=3*10, nx_right=3*3,
    ny_bot=3*1, ny_act=3*14, ny_top=3*2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=Np
)

# interpolation scape
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip; mitc=MITC9)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# generate the function for the boundary conditions
prescribed_u = generate_boundary_function(grid, "edge"; ramp = t -> min(t, 1))

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2])) # what happens when we rotate
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
# add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
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
rhs1   = zeros(N)   # preallocated Newton RHS, filled in-place each iteration
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
        @. rhs1 = λ * p_max * F_plv - r_int
        apply_zero!(K_eff, rhs1, ch)
        res_pre = norm(@view rhs1[free])
        @show res_pre
        res_pre < tol_nl && (converged_pre = true; n_iter_pre = iter - 1; break)
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

using JLD2
jldsave("minilimo_inflation.jld2"; u=un)

# load the initial displacements
un .= load("minilimo_inflation.jld2")["u"]
Ferrite.update!(ch, 1.0)

# # the three different surface where different pressures are assembled
# # SRF_1: endocardium, Plv only (outward)
# # SRF_2: endocardium + actuator, Plv (outward) and Pact (inward, opposing Plv)
# # SRF_3: actuator exterior, Pact only (outward)
# Plv_srf     = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")  # Plv acts here
# Pact_srf    = getcellset(grid, "SRF_3")                                # +Pact
# PlvPact_srf = getcellset(grid, "SRF_2")                                # −Pact (opposes Plv)

# # what's the volume in this configuration
# vol = -2compute_volume(dh, scv, un; cellset=Plv_srf) * m3_to_ml
# println("Initial volume of the device: ", round(vol; digits=4), " ml")
# # vtk_save(pvd);

# import OrdinaryDiffEq as ODE
# using Plots

# # open-loop windkessel
# function Windkessel!(du,u,p,t)
#     # unpack
#     (Vlv,Pa,Pv,Plv) = u
#     (Ra,Ca,Rv,Cv,Rp)  = p

#     # flow at the two vales
#     Qmv = Pv ≥ Plv ? (Pv - Plv)/Rv : (Plv - Pv)/1e10
#     Qao = Plv ≥ Pa ? (Plv - Pa)/Ra : (Pa - Plv)/1e10

#     # rates
#     du[1] = Qmv - Qao                 # dVlv/dt=Qmv-Qao
#     du[2] = Qao/Ca + (Pv-Pa)/(Rp*Ca)  # dPa/dt
#     du[3] = (Pa-Pv)/(Rp*Cv) - Qmv/Cv  # dPv/dt
#     du[4] = 0.0                       # un-used u[4] hold the ventricular pressure
# end;

# # actuation waveform (normalized to [0,1])
# ϕᵢ(t;tC=0.10,tR=0.25,TC=0.15,TR=0.45) = 0.0<=(t-tC)%1<=TC ? 0.5*(1-cos(π*((t-tC)%1)/TC)) : (0.0<=(t-tR)%1<=TR ? 0.5*(1+cos(π*((t-tR)%1)/TR)) : 0)

# # Kasra's parameters
# Ra = 8.0e6*Pa2mmHg/m3_to_ml     # Pa.s/m³ -> mmHg.s/ml
# Rp = 1.0e8*Pa2mmHg/m3_to_ml     # Pa.s/m³
# Rv = 5.0e5*Pa2mmHg/m3_to_ml     # Pa.s/m³
# Ca = 8.0e-9*m3_to_ml/Pa2mmHg    # m³/Pa
# Cv = 5.0e-8*m3_to_ml/Pa2mmHg    # m³/Pa not used in openloop
# Pv = p_max * Pa2mmHg

# # setup
# u₀ = [vol, 80, Pv, Pv]              # initial conditions
# tspan = (0.0, 4.0)
# params = (Ra,Ca,Rv,Cv,Rp)

# # generate a problem to solve
# prob = ODE.ODEProblem(Windkessel!, u₀, tspan, params)

# # full control over iterations
# integrator = ODE.init(prob, ODE.Tsit5(), reltol=1e-6,
#                       abstol=1e-9, save_everystep=false)

# # coupling tolerances
# tol      = 1e-4
# max_iter = 20
# dt_cpl   = 0.01

# # storages
# vols = Float64[]
# pres = Float64[]
# pact = Float64[]
# paos = Float64[]
# pvns = Float64[]
# vtarget = []

# # new FE arrays
# dVdu = zeros(N)

# # Newton + Schur-complement solve for one coupling step. Every FE buffer is
# # passed in via `bufs`, so the inner loop touches no non-const globals and is
# # type-stable and allocation-free. `u` is updated in place; returns the updated
# # pressure together with iteration/convergence info.
# function solve_step!(u, p, Pact, V_target, dh, scv, mat, ch,
#                      Plv_srf, Pact_srf, PlvPact_srf, bufs; max_iter=20, tol=1e-4, verbose=false)
#     (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact,
#        K_eff, rhs1, v1, v2, dVdu, F_lu) = bufs
#     converged = false; n_iter = 0; V₃D = 0.0
#     for iter in 1:max_iter
#         assemble_all!(K_int, r_int, dh, scv, u, mat)
#         fill!(F_plv, 0.0); fill!(F_pact, 0.0); fill!(F_plvpact, 0.0)
#         assemble_pressure_region!(K_plv, F_plv, scv, u, dh, Plv_srf)
#         assemble_pressure_region!(K_pact, F_pact, scv, u, dh, Pact_srf)
#         assemble_pressure_region!(K_plvpact, F_plvpact, scv, u, dh, PlvPact_srf)
#         # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
#         V₃D = -compute_volume(dh, scv, u; cellset=Plv_srf)
#         volume_gradient!(dVdu, dh, scv, u; cellset=Plv_srf)
#         dVdu[ch.prescribed_dofs] .= 0.0
#         r_V = V₃D - V_target
#         # F_ext = p*F_plv + Pact*F_pact - Pact*F_plvpact;  K_eff = K_int - ∂F_ext/∂u
#         K_eff.nzval .= K_int.nzval .- p .* K_plv.nzval .- Pact .* K_pact.nzval .+ Pact .* K_plvpact.nzval
#         @. rhs1 = p * F_plv + Pact * F_pact - Pact * F_plvpact - r_int
#         apply_zero!(K_eff, rhs1, ch)
#         verbose && @printf("    iter %2d | r_V=%+.3e | |rhs|=%.3e\n", iter, r_V, norm(rhs1))
#         if norm(rhs1) < tol && abs(r_V) < tol * max(1.0, abs(V_target)) && iter != 1
#             converged = true; n_iter = iter - 1; break
#         end
#         n_iter = iter
#         lu!(F_lu, K_eff)
#         ldiv!(v1, F_lu, rhs1)
#         ldiv!(v2, F_lu, F_plv)
#         # Schur complement (dVdu = ∂(compute_volume)/∂u = −∂V₃D/∂u):
#         S  = -dot(dVdu, v2)
#         δp = (-r_V + dot(dVdu, v1)) / S
#         u .+= v1 .+ δp .* v2
#         p  += δp
#         apply!(u, ch)
#     end
#     return p, n_iter, converged, V₃D
# end

# bufs = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact,
#           K_eff, rhs1, v1, v2, dVdu, F_lu)

# # start with the initial condition from the morphing step
# @time let u = copy(un), p = p_max, k₀ = length(pvd.timeSteps)
#     println("3D-0D Lie–Trotter coupling (dt_cpl=$(dt_cpl) s)")
#     println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters")

#     step = 0
#     while integrator.t < tspan[2] - dt_cpl / 2
#         step += 1

#         # advance Windkessel by dt_cpl; Plv = integrator.u[3] is held fixed.
#         ODE.step!(integrator, dt_cpl, true)

#         # full-LV volume (ml)
#         V_target = 0.5 * integrator.u[1] / m3_to_ml # in m³
#         push!(vtarget, integrator.u[1])

#         # pressure at this step, meaning at t [mmHg], converted to Pa for 3D model
#         Pact_mmHg = 200 * ϕᵢ(integrator.t;tC=0.1,tR=0.4,TC=0.3,TR=0.3) # in mmHg
#         Pact = Pact_mmHg / Pa2mmHg # Pa

#         # Schur Complement Newton-Raphson solve for the volume
#         @show step
#         p, n_iter, converged, V₃D = solve_step!(u, p, Pact, V_target, dh, scv, mat, ch,
#                                                 Plv_srf, Pact_srf, PlvPact_srf, bufs;
#                                                 max_iter=max_iter, tol=tol, verbose=true)

#         if mod(step, 1) == 0
#             VTKGridFile("minilimo-inflation-$(vtk_step[])", dh) do vtk
#                 vtk_step[] += 1
#                 write_solution(vtk, dh, u)
#                 Ferrite.write_constraints(vtk, ch)
#                 for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
#                 # per-node residual fields for debugging
#                 rhs_dbg = p .* F_plv .+ Pact .* F_pact .- Pact .* F_plvpact .- r_int
#                 rhs_dbg[ch.prescribed_dofs] .= 0.0
#                 u_range = dof_range(dh, :u); θ_range = dof_range(dh, :θ)
#                 n_nc    = length(grid.cells[1].nodes)
#                 res_u   = zeros(3, getnnodes(grid))
#                 res_θ   = zeros(2, getnnodes(grid))
#                 cnt     = zeros(Int, getnnodes(grid))
#                 for cell in CellIterator(dh)
#                     dofs = celldofs(cell)
#                     nids = grid.cells[Ferrite.cellid(cell)].nodes
#                     for k in 1:n_nc
#                         nid = nids[k]
#                         res_u[:, nid] .+= rhs_dbg[dofs[u_range[3k-2:3k]]]
#                         res_θ[:, nid] .+= rhs_dbg[dofs[θ_range[2k-1:2k]]]
#                         cnt[nid] += 1
#                     end
#                 end
#                 res_u ./= reshape(max.(cnt, 1), 1, :)
#                 res_θ ./= reshape(max.(cnt, 1), 1, :)
#                 write_node_data(vtk, res_u, "residual_u")
#                 write_node_data(vtk, res_θ, "residual_theta")
#                 # membrane stress resultants N₁₁, N₂₂, N₁₂ for buckling diagnostic
#                 N11 = zeros(getnnodes(grid))
#                 N22 = zeros(getnnodes(grid))
#                 N12 = zeros(getnnodes(grid))
#                 cnt_N = zeros(Int, getnnodes(grid))
#                 n_qp  = getnquadpoints(scv)
#                 n_nodes_e = getnbasefunctions(scv.ip_shape)
#                 for cell in CellIterator(dh)
#                     reinit!(scv, cell)
#                     u_e = u[shelldofs(cell)]
#                     G₃  = scv.G₃_elem[1]
#                     N_avg = zero(SymmetricTensor{2,2,Float64})
#                     for qp in 1:n_qp
#                         a₁, a₂ = FerriteShells.covariant_basis(scv, qp, u_e, n_nodes_e)
#                         c_ms = SymmetricTensor{2,2}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
#                         Nq, _ = membrane_stress_and_tangent(mat, c_ms, scv.A_metric[qp],
#                                                             Vec{3}(Tuple(scv.A₁[qp])), Vec{3}(Tuple(scv.A₂[qp])), G₃)
#                         N_avg += Nq
#                     end
#                     N_avg /= n_qp
#                     nids = grid.cells[Ferrite.cellid(cell)].nodes
#                     for k in 1:n_nc
#                         nid = nids[k]
#                         N11[nid] += N_avg[1,1]
#                         N22[nid] += N_avg[2,2]
#                         N12[nid] += N_avg[1,2]
#                         cnt_N[nid] += 1
#                     end
#                 end
#                 N11 ./= max.(cnt_N, 1)
#                 N22 ./= max.(cnt_N, 1)
#                 N12 ./= max.(cnt_N, 1)
#                 write_node_data(vtk, N11, "N11")
#                 write_node_data(vtk, N22, "N22")
#                 write_node_data(vtk, N12, "N12")
#                 pvd[k₀+integrator.t] = vtk
#             end
#             @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n", integrator.t, p * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter)
#         end

#         !converged && (@warn "step $step (t=$(integrator.t)) did not converge"; break)

#         # feed new LV pressure back into ODE state.
#         integrator.u[4] = p * Pa2mmHg # back in mmHg for the ODE
#         ODE.u_modified!(integrator, true)

#         push!(vols, 2V₃D * m3_to_ml)   # full volume [ml]
#         push!(pres, p * Pa2mmHg)       # pressure [mmHg]
#         push!(pact, Pact_mmHg)
#         push!(paos, integrator.u[2])
#         push!(pvns, integrator.u[3])
#     end
# end
vtk_save(pvd);

# times = collect(0:dt_cpl:integrator.t)[1:length(pres)]
# p1=plot(times, [vols, pres, pact, paos, pvns], xlabel="Time [s]",
#         label=["Vlv" "Plv" "Pact" "Pao" "Pv"], lw=2, legend=:right)
# p2=plot(vols, pres, label=:none, xlim=extrema(vols).+(-10,10), ylims=(0, 100),
#         xlabel="Volume [ml]", ylabel="Pressure [mmHg]", lw=2,
#         linez=times./maximum(times))
# plot(p1, p2)
# savefig("3D0D_limo_ferriteshells_N$Np.png")
