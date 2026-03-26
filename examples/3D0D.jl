using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK
import OrdinaryDiffEq as ODE
using Plots
# 3D-0D hemodynamic coupling: RM shell "ventricle" coupled to a 0D pressure-volume model.
#
# Geometry: quarter of a flat square membrane [0,L/2]×[0,L/2] (the "ventricle wall"),
# simply-supported at the two outer edges with symmetry BCs on the two interior edges.
# The shell inflates under internal follower pressure p in the +z direction.
#
# 3D model: Reissner–Mindlin shell, 5 DOF/node, Q9 elements.
#   Equilibrium:      R_int(u) − p·F_p(u) = 0
#   Volume (quarter): V₃D(u) = compute_volume(dh, scv, u)
#
# 0D model: linear compliance  V₀D(p) = C₀D · p  (reference volume V_ref = 0 since flat).
#   This models a linear elastic chamber: pressure proportional to enclosed volume.
#
# Coupled system (residuals = 0):
#   R_u(u, p) = R_int(u) − p·F_p(u) = 0          (N_dof equations)
#   R_p(u, p) = V₃D(u)  − V₀D(p)   = 0          (1 equation)
#
# Newton linearisation:
#   [ K_eff   −F_p  ] [ δu ] = [ −R_u = p·F_p − R_int ]
#   [ ∂V/∂u  −C₀D  ] [ δp ]   [ −R_p = V₀D(p) − V₃D  ]
#
# where K_eff = K_int − p·K_pres and C₀D = dV₀D/dp.
#
# Bordering solve (two back-substitutions + scalar Schur complement):
#   v₁ = K_eff⁻¹ · (p·F_p − R_int)         equilibrium correction
#   v₂ = K_eff⁻¹ · F_p                      load-direction vector
#   S  = −C₀D − (∂V/∂u) · v₂               Schur complement (scalar)
#   δp = ((V₀D − V₃D) − (∂V/∂u) · v₁) / S
#   δu = v₁ + δp · v₂
#
# Volume control: prescribe V_target = step · ΔV, set V₀D = V_target and C₀D = 0.
# This recovers the pure displacement-controlled limit: S = −(∂V/∂u) · v₂.
# Setting C₀D > 0 enables genuine 3D-0D compliance coupling.

function make_quarter_pillow_grid(n; L=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n, n), corners))
    addfacetset!(grid, "edge",  x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addfacetset!(grid, "sym_x", x -> isapprox(x[1], 0.0, atol=1e-10))
    addfacetset!(grid, "sym_y", x -> isapprox(x[2], 0.0, atol=1e-10))
    return grid
end

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
        FerriteShells.reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        FerriteShells.membrane_residuals_RM_explicit!(re_i, scv, u_e, mat)
        FerriteShells.bending_residuals_RM_explicit!(re_i, scv, u_e, mat)
        assemble_pressure!(re_p, scv, u_e, 1.0)
        FerriteShells.membrane_tangent_RM_explicit!(ke_i, scv, u_e, mat)
        FerriteShells.bending_tangent_RM_explicit!(ke_i, scv, u_e, mat)
        assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm_i, sd, ke_i, re_i)
        assemble!(asm_p, sd, ke_p)
        F_p[sd] .+= re_p
    end
end

# open-loop windkessel
function Windkessel_init!(du,u,p,t)
    # unpack
    (Vlv,Pa,Plv) = u
    (Ra,Ca,Rv,Cv,Rp,Pv)  = p
    # Plv
    V0 = 20; Emin = 0.05; Emax = 2.0
    Plv = (Emin+(Emax-Emin)*ϕᵢ(t;tC=0.0,tR=0.40,TC=0.40,TR=0.2))*(Vlv-V0)
    u[3] = Plv
    # flow at the two vales
    Qmv = Pv ≥ Plv ? (Pv - Plv)/Rv : (Plv - Pv)/1e10
    Qao = Plv ≥ Pa ? (Plv - Pa)/Ra : (Pa - Plv)/1e10

    # rates
    du[1] = Qmv - Qao             # dVlv/dt=Qmv-Qao
    du[2] = Qao/Ca - Pa/(Rp*Ca)   # dPa/dt=Qao/C-Pao/RC
end

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
mmHg2kPa = 0.133322387415
Pa2mmHg = 0.00750062 # Pa/mmHg
m3_to_ml = 1.0e6          # m³ to ml
Ra = 8.0e6*Pa2mmHg/m3_to_ml     # Pa.s/m³ -> mmHg.s/ml
Rp = 1.0e8*Pa2mmHg/m3_to_ml     # Pa.s/m³
Rv = 5.0e5*Pa2mmHg/m3_to_ml     # Pa.s/m³
Ca = 8.0e-9*m3_to_ml/Pa2mmHg    # m³/Pa
Cv = 5.0e-8*m3_to_ml/Pa2mmHg    # m³/Pa not used in openloop
Pv = 6.01                       # venous pressure in mmHg
# setup
EDV = 207.05                    #ml; end-diastolic volume.
u₀ = [160, 80, 60]              # initial conditions
tspan = (0.0, 10.0)
params = (Ra,Ca,Rv,Cv,Rp,Pv)

# generate a problem to solve
prob = ODE.ODEProblem(Windkessel!, u₀, tspan, params)

# full control over iterations
integrator = ODE.init(prob, ODE.Tsit5(), reltol=1e-6,
                      abstol=1e-9, save_everystep=false)

# solve the ODE oroblem to get the reference solution for the uncoupled 0D model
# sol = ODE.solve(ODE.ODEProblem(Windkessel_init!, [100, 80, 60], tspan, params),
#                 ODE.Tsit5(), dtmax=0.001)
# V0 = 20; Emin = 0.05; Emax = 2.0
# PLV = @. (Emin+(Emax-Emin)*ϕᵢ(sol.t;tC=0.0,tR=0.40,TC=0.40,TR=0.2))*(sol[1,:]-V0)
# p1 = plot(sol.t, PLV,label="P_\\ LV",lw=2)
# plot!(p1, sol, idxs=[1,2], linewidth=2, xaxis="Time (t/T)", yaxis="Pressure (mmHg)",
#       label=["V_\\ LV" "P_\\ AO"], ylims=(0,160))
# plot!(p1, sol.t, Pv.*ones(length(sol.t)), label="P_\\ Fill", lw=2)
# p2 = plot(getindex.(sol.u, 1), PLV, alpha=0.5,
#           label=:none, lw=2, xlims=(0,180), ylims=(0,100), xlabel="Volume")
# plot(p1,p2;layout=(1,2),size=(1200,400))

# Geometry scaled to match the Windkessel ODE units (mmHg, ml):
#
#   Target: V_quarter ≈ 52 ml = 5.2e-5 m³ (→ V_full = 4×V_quarter ≈ 208 ml ≈ EDV)
#           p ≈ 800 Pa ≈ 6 mmHg at end-diastole (≈ Pv → mitral valve closes naturally)
#
#   Large-deflection membrane scaling: p ≈ C·E·t·w²/L³, V_quarter ≈ κ·(L/2)²·w
#   Calibrated from the SquareAirbag reference (C≈0.126, κ≈0.16).
#   Solving for the target operating point with E=1 MPa gives L≈0.15 m, t≈8 mm.
n   = 16
L   = 0.1
mat = LinearElastic(4.0e4, 0.3, 8e-3)

grid = make_quarter_pillow_grid(n; L)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "edge"),  x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "edge"),  x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_x"), x -> 0.0, [1]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_x"), x -> 0.0, [1]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_y"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_y"), x -> 0.0, [2]))
close!(ch); Ferrite.update!(ch, 0.0)

N = ndofs(dh)
K_int  = allocate_matrix(dh)
K_pres = allocate_matrix(dh)
K_eff  = allocate_matrix(dh)
r_int  = zeros(N)
F_p    = zeros(N)
dVdu   = zeros(N)
v1     = zeros(N)
v2     = zeros(N)

tol      = 1e-6
max_iter = 20
dt_cpl   = 0.01   # coupling (macro) time step [s]; ODE may take smaller internal steps

# Displacement steps: trace p vs w_center from w=0 up to p=p_max.
# V_target   = sol.u[end][1] / m3_to_ml / 4 # target is last solution
tol     = 1e-6
max_iter = 20

# V_offset = 0.0

# print("3D initial condition V_target=",round(V_target*m3_to_ml,digits=3), " ml")
# println(" p_target=",PLV[end], " mmHg")

# # get to EDV configuration
# let u = zeros(N), p = 0.0
#     assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
#     K_eff.nzval .= K_int.nzval
#     let rhs_dummy = zeros(N); apply_zero!(K_eff, rhs_dummy, ch); end
#     F_lu = lu(K_eff)
#     for iter in 1:max_iter
#         assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)

#         # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
#         V₃D = -compute_volume(dh, scv, u)
#         @show V₃D, p * Pa2mmHg, iter
#         volume_gradient!(dVdu, dh, scv, u)
#         dVdu[ch.prescribed_dofs] .= 0.0   # zero BC DOFs in gradient
#         Pact = 0
#         r_V  = V₃D - V_target - V_offset
#         K_eff.nzval .= K_int.nzval .- (p-Pact) .* K_pres.nzval
#         rhs1 = (p-Pact) .* F_p .- r_int
#         apply_zero!(K_eff, rhs1, ch)

#         if norm(rhs1) < tol && abs(r_V) < tol * max(1.0, abs(V_target))
#             converged = true; n_iter = iter - 1; break
#         end
#         n_iter = iter

#         lu!(F_lu, K_eff)
#         ldiv!(v1, F_lu, rhs1)   # equilibrium correction
#         ldiv!(v2, F_lu, F_p)    # load-direction vector

#         # Schur complement (dVdu = ∂(compute_volume)/∂u = −∂V₃D/∂u):
#         S  = -dot(dVdu, v2)                       # > 0: dVdu[u_z]<0, v2[u_z]>0
#         δp = (-r_V + dot(dVdu, v1)) / S
#         u .+= v1 .+ δp .* v2
#         p  += δp
#         apply!(u, ch)
#     end
#     VTKGridFile("3d0d_ventricle_initial", dh) do vtk
#         write_solution(vtk, dh, u)
#     end
# end


pvd  = paraview_collection("3d0d_ventricle")
vols = Float64[]
pres = Float64[]
pact = Float64[]
vtarget = []

# Lie–Trotter operator splitting:
#
#   At each macro step [t, t+dt_cpl]:
#     (1) FLUID step  : advance Windkessel by dt_cpl with Plv fixed → new Vlv, Pa
#     (2) SOLID step  : bordering Newton to find (u, p) with V₃D(u) = V_target
#                       where V_target = Vlv_full / 4  (quarter-domain)
#     (3) SYNCHRONISE : update Plv in ODE state with pressure p from solid
#
# Note: the toy square membrane (L=1 m, E=1 MPa) has volume O(0.01 m³) ≈ O(10⁴ ml),
# while the Windkessel parameters are tuned for a ~200 ml cardiac LV.  For a
# physiologically realistic coupled simulation, scale L and/or the Windkessel
# resistances/compliances so that the two models operate in the same volume range.

# Reset ODE to a flat-plate–consistent initial state: Vlv=0, Pa=80 mmHg, Plv=0.
ODE.reinit!(integrator, [0.0, 80.0, 0.0]) # flat-plate consistent: Vlv=0, Pa=80 mmHg, Plv=0

let u = zeros(N), p = 0.0
    assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
    K_eff.nzval .= K_int.nzval
    let rhs_dummy = zeros(N); apply_zero!(K_eff, rhs_dummy, ch); end
    F_lu = lu(K_eff)

    VTKGridFile("3d0d_ventricle-0", dh) do vtk
        write_solution(vtk, dh, u); pvd[0.0] = vtk
    end

    println("3D-0D Lie–Trotter coupling (RM Q9, n=$n, dt_cpl=$(dt_cpl) s)")
    println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters")

    step = 0
    while integrator.t < tspan[2] - dt_cpl / 2
        step += 1

        # (1) FLUID STEP: advance Windkessel by dt_cpl; Plv = integrator.u[3] is held fixed.
        ODE.step!(integrator, dt_cpl, true)

        # Convert full-LV volume (ml) to quarter-domain target (m³).
        V_target = integrator.u[1] / m3_to_ml / 4
        push!(vtarget, V_target)

        # pressure at this step, meaning at t [mmHg], converted to Pa for 3D model
        Pact_mmHg = 80 * ϕᵢ(integrator.t;tC=0.1,tR=0.4,TC=0.3,TR=0.3)
        Pact = Pact_mmHg / Pa2mmHg

        # (2) SOLID STEP: bordering Newton.
        converged = false; n_iter = 0; V₃D = 0.0
        for iter in 1:max_iter
            assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)

            # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
            V₃D = -compute_volume(dh, scv, u)
            volume_gradient!(dVdu, dh, scv, u)
            dVdu[ch.prescribed_dofs] .= 0.0   # zero BC DOFs in gradient

            r_V  = V₃D - V_target
            K_eff.nzval .= K_int.nzval .- (p-Pact) .* K_pres.nzval
            rhs1 = (p-Pact) .* F_p .- r_int
            apply_zero!(K_eff, rhs1, ch)

            if norm(rhs1) < tol && abs(r_V) < tol * max(1.0, abs(V_target))
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter

            lu!(F_lu, K_eff)
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

        # (3) SYNCHRONISE: feed new LV pressure back into ODE state.
        integrator.u[3] = p * Pa2mmHg
        ODE.u_modified!(integrator, true)

        push!(vols, V₃D * 4 * m3_to_ml)   # full volume [ml]
        push!(pres, p * Pa2mmHg)            # pressure [mmHg]
        push!(pact, Pact_mmHg)

        if mod(step, 10) == 0
            VTKGridFile("3d0d_ventricle-$step", dh) do vtk
                write_solution(vtk, dh, u); pvd[integrator.t] = vtk
            end
            @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n",
                    integrator.t, p * Pa2mmHg, V₃D * 4 * m3_to_ml, Pact_mmHg, n_iter)
        end
    end
end
vtk_save(pvd)
plot([vols, pres, pact, vtarget* 4 * m3_to_ml], xlabel="Volume [ml]", ylabel="Pressure [mmHg]",
     label=["Vlv" "Plv" "Pact" "Vtarget"])